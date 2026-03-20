# OmniTrace
This is the official repository for the paper:
**"OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs"**

[📄 Paper](#) | [🌐 Project Page](#) | [🤗 Demo](#) | [📦 PyPI](https://pypi.org/project/omnitrace/0.1.0/)

OmniTrace is a **plug-and-play framework for generation-time attribution** in multimodal large language models (text, image, audio, video).

## 🧠 Overview

<p align="center">
  <img src="docs/overview.png" width="100%" />
</p>
**Figure:** OmniTrace performs generation-time token tracing in decoder-only multimodal LLMs, producing unified span-level attribution across text, image, audio, and video inputs.

- Works for **decoder-only multimodal LLMs**
- Supports **text, image, audio, and video**
- Provides **generation-time attribution**
- Plug-and-play across different backends (Qwen, MiniCPM)
- No retraining required

---

## 🚀 Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install omnitrace
```


### Option 2: Install from GitHub (recommended)

```bash
git clone https://github.com/Jackie-2000/OmniTrace.git
cd OmniTrace
pip install -e .
```

## ⚡ Quick Start (Python API)
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
        {"text": "Is the time shown in clock or watch in both <image> and <image> the same?\n(A) Yes, they are both at 9 o'clock\n(B) Yes, they are both at 12 o'clock\n(C) No, they show different time"},
        {"image": "examples/media/262_0.jpg"},
        {"image": "examples/media/262_1.jpg"}
    ],
}

# audio input
sample = {
    "prompt": "Answer the question based on the audio provided. Explain your reasoning step by step.\n",
    "question": [
        {"text": "What was the last sound in the sequence?\nA. footsteps\nB. dog_barking\nC. camera_shutter_clicking\nD. tapping_on_glass"},
        {"audio": "examples/media/b7701ab1-c37e-49f2-8ad9-7177fe0465e9.wav"}
    ],
}

# video input
sample = {
    "prompt": "Answer the question based on the video provided. Explain your reasoning step by step.\n",
    "question": [
        {"text": "What type of weapon does the slain legend retrieve?\nA. Sword\nB. Axe\nC. Gun\nD. Spear"},
        {"video": "examples/media/6Z_XNM_iT4g.mp4"}
    ],
}

result = tracer.trace(sample)
print(result)
```

### 🖥️ Command Line Usage
Run OmniTrace on a dataset file:
```bash
python scripts/run_demo.py trace \
  --questions_path examples/question_visual_text.json \
  --model_name qwen \
  --method attmean
```

#### 📂 Input Format
The input file should be a JSON list of samples:
```json
[
  {
    "id": 0,
    "prompt": "Answer the question based on the images provided. Explain your reasoning step by step.\n",
    "question": [
        {"text": "Is the time shown in clock or watch in both <image> and <image> the same?\n(A) Yes, they are both at 9 o'clock\n(B) Yes, they are both at 12 o'clock\n(C) No, they show different time"},
        {"image": "examples/media/262_0.jpg"},
        {"image": "examples/media/262_1.jpg"}
    ],
  },
  {
    "id": 1,
    "prompt": "Answer the question based on the audio provided. Explain your reasoning step by step.\n",
    "question": [
        {"text": "What was the last sound in the sequence?\nA. footsteps\nB. dog_barking\nC. camera_shutter_clicking\nD. tapping_on_glass"},
        {"audio": "examples/media/b7701ab1-c37e-49f2-8ad9-7177fe0465e9.wav"}
    ],
  },
  {
    "id": 2,
    "prompt": "Answer the question based on the video provided. Explain your reasoning step by step.\n",
    "question": [
        {"text": "What type of weapon does the slain legend retrieve?\nA. Sword\nB. Axe\nC. Gun\nD. Spear"},
        {"video": "examples/media/6Z_XNM_iT4g.mp4"}
    ],
  },
]
```

## 🧩 Supported Modalities

OmniTrace supports multimodal inputs with the following structure:

### 🔹 Text + Image (Interleaved)
You can provide **multiple text and image inputs**, interleaved:

| Field    | Description                  |
|----------|------------------------------|
| `text`   | Input text (string or list)  |
| `image`  | Path(s) to image(s)          |

Example:
```json
{
    "prompt": "Summarize the conversation.\n",
    "question": [
        { "text": "<TURN> \"I have most enjoyed painting poor, delicate children. I didn't know whether that will interest anyone.\" - Helene Schjerfbeck (1862-1946). The Convalescent (1888) is her most famous example of this. It shows the girl getting her energy back."},
        {"image": "examples/media/-288980723939800020.jpg"},
        {"text": "<TURN> Thank you for sharing this. 'The Wounded Angel' is my favourite painting in AteneumMuseum"},
        {"image": "examples/media/8846049217870534914.jpg"},
        {"image": "examples/media/-4402135406098345009.jpg"},
        {"text": "<TURN> That's a very nice indeed!"}
    ],
}
```

---

### 🔹 Audio/Video

| Field    | Description                  |
|----------|------------------------------|
| `audio`  | Path to a single audio/video file  |
| `question` / `text` | Prompt related to the audio/video |

Example:
```json
{
    "question": [
        {"text": "What was the last sound in the sequence?\nA. footsteps\nB. dog_barking\nC. camera_shutter_clicking\nD. tapping_on_glass"},
        {"audio": "examples/media/b7701ab1-c37e-49f2-8ad9-7177fe0465e9.wav"}
    ],
}
{
    "question": [
        {"text": "What type of weapon does the slain legend retrieve?\nA. Sword\nB. Axe\nC. Gun\nD. Spear"},
        {"video": "examples/media/6Z_XNM_iT4g.mp4"}
    ],
}
```

### ⚠️ Notes

- **Text + Image** supports multiple inputs and interleaving.
- **Audio and Video** currently support **only one file per sample**.
- Each sample should include a prompt (`text` or `question`) describing the task.

---

## ⚙️ Arguments

### `--questions_path`
Path to input JSON file.

### `--model_name`
Supported:
- `qwen`
- `minicpm`

### `--method`
Attribution method:
- `attmean`
- `attraw`
- `attgrads`

---

## 📁 Example Files

We provide ready-to-run examples:

```bash
examples/question_visual_text.json
examples/question_audio.json
examples/question_video.json
```

---

## 🧪 Minimal Test

Run this to verify everything works:

```bash
python scripts/run_demo.py trace \
  --questions_path examples/question_visual_text.json \
  --model_name qwen \
  --method attmean
```

---

## 💡 Tips

- Always run from the repo root
- Use relative paths for media files
- `attgrads` may require high-memory GPUs (e.g., H100/H200)


---

## 📌 Citation

(To be added)