import argparse
import json
import time
from pathlib import Path
from threading import Thread

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextIteratorStreamer

from diffusion_hf import STOP_STRINGS, TASK_PROMPTS
from speed_compare.config import MINERU_OUTPUT_PATH, resolve_mineru_model_path


MAX_NEW_TOKENS = 4096
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 2048 * 28 * 28


def _trim_text_at_stop(text: str) -> tuple[str, bool]:
    stop_positions = [text.find(stop) for stop in STOP_STRINGS if stop in text]
    if not stop_positions:
        return text, False
    stop_index = min(stop_positions)
    return text[:stop_index], True


def _trim_records(records: list[dict]) -> list[dict]:
    trimmed_records = []
    for record in records:
        text, hit_stop = _trim_text_at_stop(record["text"] or "")
        trimmed_records.append({"time": record["time"], "text": text})
        if hit_stop:
            break
    return trimmed_records


def infer_mineru(
    *,
    model_path: str | Path,
    image_path: str | Path,
    prompt_type: str = "text",
    prompt: str | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    prompt_text_value = prompt or TASK_PROMPTS[prompt_type].lstrip("\n")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        use_fast=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt_text_value,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert (
        "<|vision_start|>" in text
        or "<|image_pad|>" in text
        or "<image>" in text
    ), f"chat template did not insert image token. text=\n{text}"

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer=processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=False,
    )

    records = []
    start = time.perf_counter()
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        records.append(
            {
                "time": time.perf_counter() - start,
                "text": new_text,
            }
        )

    thread.join()
    elapsed = time.perf_counter() - start

    try:
        trimmed_records = _trim_records(records)
        full_output = "".join(record["text"] for record in trimmed_records)
        token_count = len(processor.tokenizer(full_output, add_special_tokens=False).input_ids)
        first_token_time = next((record["time"] for record in trimmed_records if record["text"]), 0.0)
        return {
            "model_name": "MinerU 2.5",
            "prompt_text": prompt_text_value,
            "records": trimmed_records,
            "response": full_output.strip(),
            "elapsed": elapsed,
            "first_token_time": first_token_time,
            "token_count": token_count,
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU 2.5 OCR inference.")
    parser.add_argument("--model-path", default=None, help="MinerU 2.5 model directory.")
    parser.add_argument("--image-path", required=True, help="Input image path.")
    parser.add_argument("--prompt-type", choices=sorted(TASK_PROMPTS.keys()), default="text")
    parser.add_argument("--prompt", default=None, help="Optional custom prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else resolve_mineru_model_path()
    image_path = Path(args.image_path).expanduser().resolve()

    result = infer_mineru(
        model_path=model_path,
        image_path=image_path,
        prompt_type=args.prompt_type,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    with MINERU_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result["records"], f, indent=2)

    print("\n===== elapsed =====")
    print(result["elapsed"])
