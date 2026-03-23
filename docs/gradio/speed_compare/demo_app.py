import time

import gradio as gr

from speed_compare.config import PROMPT_CHOICES, REPLAY_STEP_SECONDS, TASK_PROMPTS
from speed_compare.inference import run_diffusion_model, run_mineru_model
from speed_compare.parsers import make_diffusion_events, make_mineru_events
from speed_compare.render import (
    APP_CSS,
    build_initial_view,
    build_result_panel,
    build_waiting_panel,
)


def _log_generation_result(label: str, result: dict) -> None:
    print(f"\n===== {label} Result =====")
    print(result.get("response", "").strip() or "(empty response)")
    print(f"===== End {label} Result =====\n")


def run_compare(image_path: str | None, prompt_type: str):
    initial_mineru, initial_diffusion = build_initial_view()

    if not image_path:
        yield (initial_mineru, initial_diffusion)
        return

    if prompt_type not in TASK_PROMPTS:
        yield (initial_mineru, initial_diffusion)
        return

    yield (
        build_waiting_panel("MinerU 2.5", "Running", "append"),
        build_waiting_panel("MinerU-Diffusion", "Running", "positional"),
    )

    try:
        mineru_result = run_mineru_model(image_path, prompt_type)
        mineru_events, mineru_duration = make_mineru_events(mineru_result)
        diffusion_result = run_diffusion_model(image_path, prompt_type)
        _log_generation_result("MinerU 2.5", mineru_result)
        _log_generation_result("MinerU-Diffusion", diffusion_result)
    except Exception as exc:
        print(exc)
        yield (initial_mineru, initial_diffusion)
        return

    diffusion_events, diffusion_duration = make_diffusion_events(diffusion_result)
    max_time = max(mineru_duration, diffusion_duration, 0.0)

    yield (
        build_result_panel("MinerU 2.5", mineru_result, mineru_events, mineru_duration, "append", 0.0, "Working"),
        build_result_panel(
            "MinerU-Diffusion",
            diffusion_result,
            diffusion_events,
            diffusion_duration,
            "positional",
            0.0,
            "Working",
        ),
    )

    replay_start = time.perf_counter()
    last_replay_time = -1.0
    while True:
        replay_time = min(time.perf_counter() - replay_start, max_time)
        if replay_time <= last_replay_time and replay_time < max_time:
            time.sleep(REPLAY_STEP_SECONDS)
            continue
        last_replay_time = replay_time
        mineru_status = "Done!" if replay_time >= mineru_duration else "Working"
        diffusion_status = "Done!" if replay_time >= diffusion_duration else "Working"

        yield (
            build_result_panel(
                "MinerU 2.5",
                mineru_result,
                mineru_events,
                mineru_duration,
                "append",
                replay_time,
                mineru_status,
            ),
            build_result_panel(
                "MinerU-Diffusion",
                diffusion_result,
                diffusion_events,
                diffusion_duration,
                "positional",
                replay_time,
                diffusion_status,
            ),
        )
        if replay_time >= max_time:
            break
        if max_time - replay_time > 0:
            time.sleep(REPLAY_STEP_SECONDS)

    yield (
        build_result_panel(
            "MinerU 2.5",
            mineru_result,
            mineru_events,
            mineru_duration,
            "append",
            mineru_duration,
            "Done!",
        ),
        build_result_panel(
            "MinerU-Diffusion",
            diffusion_result,
            diffusion_events,
            diffusion_duration,
            "positional",
            diffusion_duration,
            "Done!",
        ),
    )


def build_demo() -> gr.Blocks:
    initial_mineru, initial_diffusion = build_initial_view()

    with gr.Blocks(title="MinerU Speed Compare") as demo:
        gr.Markdown("## MinerU Speed Compare", elem_classes=["panel-title"])
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=5, elem_id="left-col"):
                image_input = gr.Image(label="Input Image", type="filepath", sources=["upload"])
                prompt_input = gr.Radio(label="OCR Type", choices=PROMPT_CHOICES, value="text", elem_id="prompt-grid")
                run_button = gr.Button("Start Comparison", variant="primary", elem_id="start-button")

            with gr.Column(scale=6, elem_id="mineru-col"):
                mineru_panel = gr.HTML(initial_mineru)

            with gr.Column(scale=6, elem_id="diffusion-col"):
                diffusion_panel = gr.HTML(initial_diffusion)

        run_button.click(
            fn=run_compare,
            inputs=[image_input, prompt_input],
            outputs=[mineru_panel, diffusion_panel],
            show_progress="hidden",
        )

    demo.queue(default_concurrency_limit=1)
    return demo


demo = build_demo()


def launch() -> None:
    demo.launch(server_name="0.0.0.0", css=APP_CSS)
