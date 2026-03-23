import re


_AUTOREGRESSIVE_SPLIT_RE = re.compile(r"<[^>]+>\s*|[^\s<]+(?:\s+)?|\s+")
_NBSP_MARK = "__NBSP__"
_NBHY_MARK = "__NBHY__"


def format_clock(seconds: float) -> str:
    return f"{max(0.0, seconds):.2f}s"


def format_stats(result: dict | None) -> str:
    if not result:
        return "-- tokens / -- tok/s"
    token_count = int(result.get("token_count") or 0)
    elapsed = float(result.get("elapsed") or 0.0)
    rate = (token_count / elapsed) if elapsed > 0 else 0.0
    return f"{token_count} tokens / {rate:.1f} tok/s"


def placeholder_text_for(text: str) -> str:
    output = []
    for char in text:
        if char == "\n":
            output.append("\n")
        elif char == "\t":
            output.append("  ")
        else:
            output.append("·")
    return "".join(output)


def decode_diffusion_token(token: str) -> str:
    return (
        token.replace("Ġ", " ")
        .replace("▁", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\t")
    )


def split_autoregressive_chunk(text: str) -> list[str]:
    protected = text.replace("\xa0", _NBSP_MARK).replace("\u2011", _NBHY_MARK)
    pieces = _AUTOREGRESSIVE_SPLIT_RE.findall(protected)
    return [piece.replace(_NBSP_MARK, "\xa0").replace(_NBHY_MARK, "\u2011") for piece in pieces]


def make_mineru_events(result: dict) -> tuple[list[dict], float]:
    events = []
    prev_time = 0.0

    for entry in result.get("records", []):
        time_value = float(entry.get("time") or 0.0)
        pieces = split_autoregressive_chunk(entry.get("text") or "")
        if not pieces:
            prev_time = time_value
            continue

        window = max(time_value - prev_time, 0.0)
        for piece_index, piece in enumerate(pieces):
            piece_time = (
                time_value
                if len(pieces) == 1
                else prev_time + window * ((piece_index + 1) / len(pieces))
            )
            events.append(
                {
                    "aligned_time": piece_time,
                    "text": piece,
                }
            )
        prev_time = time_value

    active_duration = max(
        (event["aligned_time"] for event in events if event["text"]),
        default=float(result.get("elapsed") or 0.0),
    )
    return events, active_duration


def make_diffusion_events(result: dict) -> tuple[list[dict], float]:
    raw_step_time = result.get("step_time") or []
    step_times = raw_step_time[0] if raw_step_time and isinstance(raw_step_time[0], list) else raw_step_time
    visible_text_pieces = result.get("visible_text_pieces")
    visible_tokens = result.get("visible_tokens") or []
    display_pieces = visible_text_pieces or [decode_diffusion_token(token) for token in visible_tokens]

    events = []
    for index, piece in enumerate(display_pieces):
        time_value = float(step_times[index]) if index < len(step_times) else float(result.get("elapsed") or 0.0)
        events.append(
            {
                "index": index,
                "aligned_time": time_value,
                "text": piece,
            }
        )

    active_duration = max(
        (event["aligned_time"] for event in events if event["text"]),
        default=float(result.get("elapsed") or 0.0),
    )
    return events, active_duration


def build_panel_state(
    *,
    events: list[dict] | None = None,
    active_duration: float = 0.0,
    local_time: float = 0.0,
    status: str = "Ready",
    stats: str = "-- tokens / -- tok/s",
    mode: str = "append",
) -> dict:
    active_duration = max(0.0, active_duration)
    local_time = min(max(0.0, local_time), active_duration) if active_duration else 0.0
    if active_duration <= 0:
        progress = 100.0 if status == "Done!" else 0.0
    else:
        progress = min((local_time / active_duration) * 100.0, 100.0)

    if mode == "positional":
        output = render_positional_output(events or [], local_time)
    else:
        output = render_append_output(events or [], local_time)

    return {
        "status": status,
        "stats": stats,
        "progress": progress,
        "clock": format_clock(local_time),
        "output": output,
        "done": status == "Done!",
    }


def render_append_output(events: list[dict], local_time: float) -> str:
    fragments = []
    visible = False
    for event in events:
        if event["aligned_time"] > local_time:
            break
        text = event["text"]
        if not text:
            continue
        visible = True
        fragments.append(("chunk", text))

    return {"visible": visible, "fragments": fragments}


def render_positional_output(events: list[dict], local_time: float) -> dict:
    fragments = []
    visible = False
    for event in events:
        text = event["text"]
        if event["aligned_time"] > local_time:
            placeholder = placeholder_text_for(text)
            if placeholder:
                fragments.append(("placeholder", placeholder))
            continue

        if not text:
            continue
        visible = True
        fragments.append(("chunk", text))

    return {"visible": visible, "fragments": fragments}
