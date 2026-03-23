from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


def parse_bbox_prompt(prompt_string):
    bboxes_data = []

    start_box_tag = "<|box_start|>"
    end_box_tag = "<|box_end|>"
    start_ref_tag = "<|ref_start|>"
    end_ref_tag = "<|ref_end|>"

    element_list = prompt_string.split("\n")

    for element in element_list:
        element = element.strip()
        if not element.startswith(start_box_tag):
            continue

        box_start = element.find(start_box_tag)
        box_end = element.find(end_box_tag)
        if box_start == -1 or box_end == -1:
            continue

        coords_str = element[box_start + len(start_box_tag) : box_end].strip()
        try:
            x1, y1, x2, y2 = map(int, coords_str.split())
            box_coords = (x1, y1, x2, y2)
        except ValueError:
            print(f"Warning: skip malformed coords string: '{coords_str}'")
            continue

        ref_start = element.find(start_ref_tag)
        ref_end = element.find(end_ref_tag)
        if ref_start == -1 or ref_end == -1:
            continue

        label = element[ref_start + len(start_ref_tag) : ref_end].strip()

        rotate = "up"
        if "<|rotate_left|>" in element:
            rotate = "left"
        elif "<|rotate_right|>" in element:
            rotate = "right"
        elif "<|rotate_down|>" in element:
            rotate = "down"

        bboxes_data.append(
            {
                "box_coords": box_coords,
                "label": label,
                "rotate": rotate,
            }
        )

    return bboxes_data


def _load_font(font_size):
    font_path = Path("mineru2qwen/package/arial.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def draw_bbox(img_path, prompt, out_path=None):
    parsed_data = parse_bbox_prompt(prompt)

    if not parsed_data:
        print("No bounding box data was parsed.")
        return parsed_data

    try:
        image = Image.open(img_path)
        img_height = image.height
        draw = ImageDraw.Draw(image)

        for idx, item in enumerate(parsed_data):
            x1, y1, x2, y2 = item["box_coords"]
            x1 = x1 / 1000 * image.width
            y1 = y1 / 1000 * image.height
            x2 = x2 / 1000 * image.width
            y2 = y2 / 1000 * image.height
            pil_box = (int(x1), int(y1), int(x2), int(y2))

            label = item["label"]
            rotate = item["rotate"]

            outline_color = "dodgerblue"
            if label == "text":
                outline_color = "crimson"
            elif label == "table":
                outline_color = "dodgerblue"
            elif label == "title":
                outline_color = "darkorchid"
            elif label == "equation":
                outline_color = "forestgreen"
            elif "caption" in label:
                outline_color = "orange"
            elif "footnote" in label:
                outline_color = "magenta"
            elif label == "image":
                outline_color = "yellowgreen"

            draw.rectangle(pil_box, outline=outline_color, width=4)
            text = f"{idx + 1}: {label} | {rotate}"
            font_size = max(12, img_height // 100)
            font = _load_font(font_size)
            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

            text_y = y1 - text_height if y1 > text_height else y1 + text_height
            text_box = (x1, text_y, x1 + text_width + 6, text_y + text_height + 4)

            draw.rectangle(text_box, fill=outline_color)
            draw.text((x1 + 3, text_y + 2), text, fill="white", font=font)

        if out_path is None:
            out_path = "./playground/output_with_bboxes.png"

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

    except FileNotFoundError:
        print(f"Error: image file not found at '{img_path}'.")
    except UnidentifiedImageError:
        print(f"Error: cannot identify image file at '{img_path}'.")
    except Exception as exc:
        print(f"An error occurred during image processing: {exc}")

    return parsed_data
