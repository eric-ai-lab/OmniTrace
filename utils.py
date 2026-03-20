from typing import Any, Dict, List
import json
from pathlib import Path

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def resolve_media_path(path_str: str, repo_root: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    return str(path)


def detect_modality(question_blocks: List[Dict[str, Any]]) -> str:
    has_image = any("image" in x for x in question_blocks)
    has_audio = any("audio" in x for x in question_blocks)
    has_video = any("video" in x for x in question_blocks)

    active = sum([has_image, has_audio, has_video])
    if active == 0:
        raise ValueError("Could not detect modality: no image/audio/video field found.")
    if active > 1:
        raise ValueError("Mixed media types in one sample are not supported by this demo runner.")

    if has_image:
        return "image_text"
    if has_audio:
        return "audio"
    return "video"