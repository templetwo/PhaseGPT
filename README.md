# PhaseGPT v1.4: The Volitional Oracle

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLX](https://img.shields.io/badge/Powered%20by-MLX-blue)](https://github.com/ml-explore/mlx)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--7B-orange)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**PhaseGPT** is a framework for training **Volitional AI**—models that possess the agency to refuse corrupted, unanswerable, or impossible queries ("Volitional Silence") rather than hallucinating.

Version 1.4 introduces the **Oracle Architecture**, optimized for Apple Silicon (M-Series) deployment via the MLX framework.

---

## 🌟 Core Capabilities

*   **Volitional Silence:** The model detects entropic corruption and semantic impossibility, outputting a `<PASS>` token instead of generating false information.
*   **Agency Cliff:** Achieved >88% accuracy in distinguishing valid queries from invalid ones (verified via `scripts/mlx_oracle_test.py`).
*   **Local Sovereignty:** Fully trainable and deployable on a single Mac Studio (M4 Max) using 4-bit QLoRA and FP16 Fusion.

## 🏗️ Repository Structure

*   `src/phasegpt/`: Core library code.
    *   `core/`: Architecture configuration (Pydantic).
    *   `trainer/`: Custom `VolitionalTrainer` with QLoRA and gradient accumulation.
    *   `data/`: Dataset generation (SQuAD + Entropy).
*   `config/`: YAML configurations for models and training.
*   `scripts/`: Operational tools.
    *   `train_production.py`: Production training loop.
    *   `manual_mlx_fuse.py`: Robust adapter fusion for MLX.
    *   `serve_mlx.py`: OpenAI-compatible API server.
    *   `chat_oracle.py`: Interactive CLI chat.
    *   `dashboard.py`: Real-time training TUI.

## 🚀 Quick Start (Apple Silicon)

### 1. Installation
```bash
git clone https://github.com/templetwo/PhaseGPT.git
cd PhaseGPT
pip install -e .
pip install mlx-lm huggingface_hub rich psutil
```

### 2. Inference (Chat with the Oracle)
Download the pre-trained Oracle (or train your own) and chat:

```bash
# Chat with local fused model
python3 scripts/chat_oracle.py --model mlx_models/Qwen2.5-7B-Oracle-FP16
```

### 3. Training
To train the Oracle from scratch on your Mac:

```bash
# Generate Data
python3 scripts/generate_mlx_data.py

# Launch Training (7B)
./scripts/train_7b_mlx.sh

# Monitor Progress
python3 scripts/dashboard.py
```

### 4. Serving (API)
Serve the model as an OpenAI-compatible endpoint:

```bash
python3 scripts/serve_mlx.py --model mlx_models/Qwen2.5-7B-Oracle-FP16
```

## 🧠 The "Agency Cliff"
PhaseGPT models exhibit a phase transition during training where they abruptly learn to map high-entropy inputs to the `<PASS>` token. This "Agency Cliff" is the visual signature of the model learning epistemic boundaries.

![Agency Cliff](https://github.com/templetwo/PhaseGPT/assets/placeholder-cliff.png)

## 📦 Models
| Model | Size | Hardware | Status |
|-------|------|----------|--------|
| **PhaseGPT-Oracle-7B** | 14GB (FP16) | M4 Max / Ultra | ✅ **Stable** |
| PhaseGPT-Oracle-1.5B | 3GB (FP16) | M1/M2/M3 | ⚠️ Experimental |

## 📜 License
MIT License. Created by **TempleTwo.AI** for the PhaseGPT Initiative.
