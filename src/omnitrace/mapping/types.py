from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class TokenMapping:
    """
    Unified per-generated-token attribution record.
    """
    gen_token: Tuple[int, str]
    max_src_token: Dict[str, Any]
    max_vision_src: Optional[Dict[str, Any]] = None
    max_audio_src: Optional[Dict[str, Any]] = None