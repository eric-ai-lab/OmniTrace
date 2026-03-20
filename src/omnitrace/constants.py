from dataclasses import dataclass
from typing import Dict
# =========================
# Model paths (optional)
# =========================

QWEN_MODEL_PATH ="Qwen/Qwen2.5-Omni-7B"
MINICPM_MODEL_PATH = "openbmb/MiniCPM-o-4_5"

# =========================
# Special tokens
# =========================

QWEN_SPECIAL_TOKENS = set([
    151644,   # <|im_start|>
    151645,   # <|im_end|>
    151647,   # <|audio_bos|>
    151648,   # <|audio_eos|>
    151652,   # <|vision_bos|>  
    151653,   # <|vision_eos|> 
    198,      # \n
    *range(15),
])

MINICPM_SPECIAL_TOKENS = set([
    151644,   # <|im_start|>
    151645,   # <|im_end|>
    151652,   # <|vision_start|> 
    151653,   # <|vision_end|>   
    151669,   # <image>
    151670,   # </image>
    151679,   # <slice>
    151680,   # </slice>
    151697,   # <|audio_start|>
    151699,   # <|audio_end|>
    198,      # \n
    *range(15),
])

# -------------------------
# spaCy chunking constants
# -------------------------
SPACY_MODEL = "en_core_web_sm"
COORD_SPLIT = {}

# =========================
# Attribution methods
# =========================

SUPPORTED_METHODS = {
    "attmean",
    "attraw",
    "attgrads",
}

GRADIENT_METHODS = {
    "attgrads",
}

# =========================
# Audio / Video defaults
# =========================

DEFAULT_BIN_SIZE = 1.0  # seconds

# video temporal mapping
TEMPORAL_PATCH_SIZE = 2 
DEFAULT_VIDEO_FPS = 1.0

# -------------------------
# POS weighting (attribution)
# -------------------------
POS_W: Dict[str, float] = {
    "NOUN": 1.0, "PROPN": 1.0, "NUM": 1.0,
    "VERB": 0.9, "ADJ": 0.8,
    "ADV": 0.5,
    "AUX": 0.1, "PRON": 0.1, "DET": 0.1, "ADP": 0.1,
    "CCONJ": 0.1, "SCONJ": 0.1, "PART": 0.1,
    "PUNCT": 0.0,
}

@dataclass(frozen=True)
class SourceCurationConfig:
    coverage: float = 0.85
    p_min: float = 0.07
    run_min: float = 0.10
    alpha: float = 0.8
    gamma: float = 2.0

DEFAULT_CURATE_CFG = SourceCurationConfig()
