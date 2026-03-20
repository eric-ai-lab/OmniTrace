This is the official repo for paper: "OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs".

# OmniTrace

OmniTrace is a plug-and-play framework for attribution in multimodal generative models (text, image, audio, video).

---

## 🚀 Installation

### Option 1: Install from GitHub (recommended)

```bash
git clone https://github.com/Jackie-2000/OmniTrace.git
cd OmniTrace
pip install -e .
```

### Option 2: (Coming soon)
```bash
pip install omnitrace
```

### ⚡ Quick Start (Python API)
```python
from omnitrace import OmniTracer

tracer = OmniTracer(
    model_name="qwen",      # or "minicpm"
    method="attmean"        # attribution method, choose from "attmean", "attraw", "attgrads"
)

# visual-text input
sample = {
    "prompt": "Answer the question based on the images provided. Explain your reasoning step by step.",
    "question": [
        {
            "text": "Is the time shown in clock or watch in both <image> and <image> the same?\n(A) Yes, they are both at 9 o'clock\n(B) Yes, they are both at 12 o'clock\n(C) No, they show different time"
        },
        {
            "image": "examples/media/262_0.jpg"
        },
        {
            "image": "examples/media/262_1.jpg"
        }
    ],
}

# audio input
sample = {
    "prompt": "Answer the question based on the audio provided. Explain your reasoning step by step.\n",
    "question": [
        {
            "text": "What was the last sound in the sequence?\nA. footsteps\nB. dog_barking\nC. camera_shutter_clicking\nD. tapping_on_glass"
        },
        {
            "audio": "examples/media/b7701ab1-c37e-49f2-8ad9-7177fe0465e9.wav"
        }
    ],
}

# video input
sample = {
    "prompt": "Answer the question based on the video provided. Explain your reasoning step by step.\n",
    "question": [
        {
            "text": "What type of weapon does the slain legend retrieve?\nA. Sword\nB. Axe\nC. Gun\nD. Spear"
        },
        {
            "video": "examples/media/6Z_XNM_iT4g.mp4"
        }
    ],
}

result = tracer.trace(sample)

print(result)
```

### 🖥️ Command Line Usage
Run OmniTrace on a dataset file:
```bash
python scripts/run_demo.py \
  --questions_path examples/question_visual_text.json \
  --model_name qwen \
  --method attmean
```

### 📂 Input Format
The input file should be a JSON list of samples:
```json
[
  {
    "id": 0,
    "prompt": "Answer the question based on the images provided. Explain your reasoning step by step.\n",
    "question": [
        {
            "text": "Is the time shown in clock or watch in both <image> and <image> the same?\n(A) Yes, they are both at 9 o'clock\n(B) Yes, they are both at 12 o'clock\n(C) No, they show different time"
        },
        {
            "image": "examples/media/262_0.jpg"
        },
        {
            "image": "examples/media/262_1.jpg"
        }
    ],
  },
  {
    "id": 1,
    "prompt": "Answer the question based on the audio provided. Explain your reasoning step by step.\n",
    "question": [
        {
            "text": "What was the last sound in the sequence?\nA. footsteps\nB. dog_barking\nC. camera_shutter_clicking\nD. tapping_on_glass"
        },
        {
            "audio": "examples/media/b7701ab1-c37e-49f2-8ad9-7177fe0465e9.wav"
        }
    ],
  },
  {
    "id": 2,
    "prompt": "Answer the question based on the video provided. Explain your reasoning step by step.\n",
    "question": [
        {
            "text": "What type of weapon does the slain legend retrieve?\nA. Sword\nB. Axe\nC. Gun\nD. Spear"
        },
        {
            "video": "examples/media/6Z_XNM_iT4g.mp4"
        }
    ],
  },
]
```