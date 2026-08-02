# Quantum-Manim-AI

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Qwen2.5](https://img.shields.io/badge/Qwen2.5-7B-lightgrey)
![Unsloth](https://img.shields.io/badge/Unsloth-LLM-green)
![Manim](https://img.shields.io/badge/Manim-0.20%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)

## Overview

**Quantum-Manim-AI** is an open-source research pipeline that transforms natural language quantum computing prompts into polished Manim animations. It uses a fine-tuned Qwen-2.5-7B model to generate Python animation code, validates the output, compiles it in a subprocess sandbox, and applies a self-correction reflection loop when rendering fails.

## System Architecture & The Reflection Loop

```text
User Prompt
      │
      ▼
Fine-Tuned Qwen-2.5-7B
      │
      ▼
Raw Manim Python Code
      │
      ▼
Subprocess Sandbox Compiler
      │
      ▼
Error Catching
      │
      ▼
Self-Correction Reflection Loop
      │
      ▼
Final MP4 Output
```

The pipeline is designed to catch mistakes early. When the generated Manim code fails syntax validation or rendering, the failed code and compiler feedback are fed back into the model. The reflection loop then asks the model to repair the scene and retry until the animation renders successfully or the retry budget is exhausted.

## Repository Structure

```text
Quantum-Manim-AI/
├── main.py
├── qwen2.5_7B_alpaca.ipynb
├── Validate_manim_dataset.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── .python-version
```

## Prerequisites & Installation

### System dependencies

- FFmpeg
- LaTeX distribution with `texlive-latex-extra`
- Cairo, Pango, and other Manim rendering dependencies

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg libcairo2-dev libpango1.0-dev texlive-latex-extra
```

### Python dependencies

Install the Python requirements:

```bash
python -m pip install -r requirements.txt
```

## Usage

Generate a Manim animation from a single prompt:

```bash
python main.py --prompt "Show a Hadamard gate acting on a qubit state and animate the resulting superposition." --max-retries 3
```

Start the interactive CLI loop:

```bash
python main.py --interactive
```

Use a custom fine-tuned model reference:

```bash
python main.py --prompt "Animate a quantum entanglement circuit." --model-name unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
```

## Fine-Tuning Details

The fine-tuning notebook `qwen2.5_7B_alpaca.ipynb` contains the training recipe used to adapt Qwen-2.5-7B to quantum mechanics and Manim code generation. Training data is validated with `Validate_manim_dataset.py`, which checks that each generated JSONL entry contains valid Python, matches the declared `%%manim` class name, and renders without syntax errors.

## Notes

- `main.py` implements the production-ready CLI pipeline.
- The reflection loop is the key safety mechanism for robust automatic repair.
- Keep the core repository lean by preserving only essential files and configuration.

---

## License

This project is released under the MIT License.
