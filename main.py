#!/usr/bin/env python3
"""Quantum-Manim-AI CLI

Streamlined pipeline without heavy prompt instructions.
Order of operations: Syntax/Clean -> Render Scene -> VLM Visual Review -> Reflection Loop.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import json_repair
from pathlib import Path
from typing import Optional

# Mount Google Drive in Colab if needed
if os.path.exists("/content") and not os.path.exists("/content/drive/MyDrive"):
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
    except ImportError:
        pass

try:
    from IPython.display import Video, display
except ImportError:
    Video = None
    display = None

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

CODE_BLOCK_RE = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)
MAGIC_LINE_RE = re.compile(r"^%%manim\s+(?:-\S+\s+)*(\S+)\s*$", re.MULTILINE)
CLASS_RE = re.compile(r"class\s+([A-Za-z_]\w*)\s*(?:\([^)]*\))?:")
DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
QWEN_WEIGHTS_PATH = Path("/content/drive/MyDrive/unsloth_outputs/final_lora_adapter")
DATASET_PATH = Path("/content/drive/MyDrive/input.jsonl")
OUTPUT_DIR = Path("output")

MODEL_CACHE = {"model": None, "tokenizer": None}
VLM_CACHE = {"model": None, "tokenizer": None, "process_vision_info": None}
DEFAULT_VLM_MODEL = DEFAULT_MODEL_NAME


def show_banner() -> None:
    banner = Panel(
        "[bold cyan]Quantum-Manim-AI[/bold cyan]\n"
        "[white]Generate, compile, render, and visually audit Manim scenes via Qwen-2.5-7B.[/white]",
        title="[bold green]Quantum-Manim-AI[/bold green]",
        border_style="bright_blue",
    )
    console.print(banner)


def save_to_dataset(prompt: str, code: str, filepath: Path = DATASET_PATH) -> None:
    """Appends verified prompt-code pairs to JSONL file with duplicate prevention."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        existing_prompts = set()

        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            existing_prompts.add(data.get("prompt", "").strip())
                        except json.JSONDecodeError:
                            continue

        if prompt.strip() in existing_prompts:
            console.print("[dim yellow]Entry already exists in dataset. Skipping duplicate append.[/dim yellow]")
            return

        record = {"prompt": prompt.strip(), "code": code.strip()}
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        console.print(f"[bold green][+][/bold green] Saved verified code pair to dataset: {filepath}")

    except Exception as exc:
        console.print(f"[red]Failed to write to dataset path ({filepath}):[/red] {exc}")


def load_qwen_model() -> tuple[object, object]:
    if MODEL_CACHE["model"] is not None and MODEL_CACHE["tokenizer"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install unsloth to run Qwen inference.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(QWEN_WEIGHTS_PATH),
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def serialize_token_inputs(tokens: dict, model: object) -> dict:
    device = next(model.parameters()).device
    result = {}
    for key, value in tokens.items():
        if hasattr(value, "to"):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


MINIMAL_SYSTEM_PREFIX = (
    "Write a complete, runnable Manim Python script inside a single ```python ``` code block. "
    "Do not include explanation outside the block."
)

def generate_manim_code(prompt_text: str, attempt: int = 1) -> str:
    model, tokenizer = load_qwen_model()

    # If it's a reflection retry, pass the prompt directly. Otherwise, add minimal instruction.
    if "Failed Code:" in prompt_text:
        content = prompt_text
    else:
        content = f"{MINIMAL_SYSTEM_PREFIX}\n\nTask:\n{prompt_text}"

    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True)
    inputs = serialize_token_inputs(inputs, model=model)

    temperature = 0.2 if attempt == 1 else min(0.3 + (attempt - 1) * 0.25, 0.8)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=temperature,
        repetition_penalty=1.15 if attempt > 1 else 1.0,
        do_sample=(temperature > 0.0),
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded.strip()

def load_vlm_model(model_name: str = DEFAULT_VLM_MODEL):
    if VLM_CACHE["model"] is not None and VLM_CACHE["tokenizer"] is not None:
        return VLM_CACHE["model"], VLM_CACHE["tokenizer"]

    try:
        from unsloth import FastVisionModel
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install unsloth and qwen_vl_utils to enable VLM review."
        ) from exc

    vlm_model, vlm_tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(vlm_model)
    VLM_CACHE["model"] = vlm_model
    VLM_CACHE["tokenizer"] = vlm_tokenizer
    VLM_CACHE["process_vision_info"] = process_vision_info
    return vlm_model, vlm_tokenizer


def parse_vlm_output(output_text: str) -> dict:
    cleaned_text = (output_text or "").strip()
    if not cleaned_text:
        return {"status": "ERROR", "valid": False, "feedback": "", "error": "Empty VLM response."}

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned_text = fenced_match.group(1).strip()

    try:
        data = json_repair.loads(cleaned_text)
        if isinstance(data, dict) and "valid" in data:
            return {
                "status": "SUCCESS",
                "valid": bool(data["valid"]),
                "feedback": str(data.get("feedback", "")),
            }
    except Exception:
        pass

    return {
        "status": "ERROR",
        "valid": False,
        "feedback": "",
        "error": f"Failed to parse JSON from VLM output: {output_text}",
    }


def run_vlm(video_path: str, prompt: str, model_name: str = DEFAULT_VLM_MODEL) -> dict:
    if not os.path.exists(video_path):
        return {"status": "ERROR", "valid": False, "feedback": "", "error": f"Video file missing: {video_path}"}

    vlm_model, vlm_tokenizer = load_vlm_model(model_name)
    process_vision_info_fn = VLM_CACHE["process_vision_info"]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": (
                        f"Analyze this rendered Manim animation for prompt: '{prompt}'.\n"
                        "Check if vector actually rotates, 3D coordinate frame exists, and labels are legible.\n"
                        "Respond STRICTLY in JSON: {\"valid\": true/false, \"feedback\": \"reasoning\"}"
                    ),
                },
            ],
        }
    ]

    text = vlm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info_fn(messages)
    inputs = vlm_tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(next(vlm_model.parameters()).device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    generated_ids = vlm_model.generate(**inputs, max_new_tokens=300)
    output_text = vlm_tokenizer.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    return parse_vlm_output(output_text)


def extract_code(raw_text: str) -> str:
    match = CODE_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()




def sanitize_and_prepare_code(code: str) -> tuple[Optional[str], Optional[str]]:
    class_match = CLASS_RE.search(code)
    if not class_match:
        return None, "No valid Python class definition (e.g. 'class MyAnimation(ThreeDScene):') was found in the output."

    actual_class_name = class_match.group(1) # Extracts 'HadamardAnimation', 'QuantumGate', etc.

    cleaned_lines = [line for line in code.splitlines() if not MAGIC_LINE_RE.match(line)]
    clean_code = "\n".join(cleaned_lines).strip()

    # Prepend basic imports
    required_imports = []
    if "from manim import *" not in clean_code:
        required_imports.append("from manim import *")
    if "import numpy" not in clean_code and "np." in clean_code:
        required_imports.append("import numpy as np")
    if "import math" not in clean_code and "math." in clean_code:
        required_imports.append("import math")

    if required_imports:
        clean_code = "\n".join(required_imports) + "\n\n" + clean_code

    return clean_code, actual_class_name

def check_syntax(code: str) -> Optional[str]:
    """Fast local check for Python syntax errors before calling external processes."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"SyntaxError on line {exc.lineno}: {exc.msg}"


def render_manim_scene(code: str, class_name: str, output_root: Path) -> dict:
    """Executes Manim rendering via CLI and captures actual Python tracebacks upon failure."""
    output_root.mkdir(parents=True, exist_ok=True)
    result = {"ok": False, "error": None, "media_file": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        scene_path = Path(tmpdir) / "scene.py"
        scene_path.write_text(code, encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "manim",
            "-ql",
            "--disable_caching",
            "--media_dir",
            str(output_root),
            str(scene_path),
            class_name,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if proc.returncode == 0:
            result["ok"] = True
            mp4_files = sorted(output_root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4_files:
                result["media_file"] = str(mp4_files[0].resolve())
            else:
                result["error"] = "Manim process finished with exit code 0, but no MP4 output file was generated."
        else:
            # Extract standard Python Traceback or stderr output directly
            raw_err = proc.stderr.strip() or proc.stdout.strip()
            result["error"] = raw_err[-2500:]  # Send recent error trace back to reflection loop

    return result


def build_reflection_prompt(prompt: str, failed_code: str, error_msg: str) -> str:
    """Combines original intent, failed code, and error trace into a targeted repair request."""
    return (
        f"Original Target Animation: {prompt}\n\n"
        f"The previous attempt failed with the following error:\n{error_msg}\n\n"
        f"Failed Code:\n```python\n{failed_code}\n```\n\n"
        "Please fix the error and output ONLY the revised Python code block inside ```python ... ```."
    )


def process_prompt(prompt: str, max_retries: int) -> bool:
    current_prompt = prompt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        console.print(f"\n[bold cyan]Attempt {attempt}/{max_retries}[/bold cyan] Generating code...")

        try:
            raw_output = generate_manim_code(current_prompt, attempt=attempt)
        except Exception as exc:
            console.print(f"[red]Generation call failed:[/red] {exc}")
            return False

        # Step 1: Code Extraction
        extracted_code = extract_code(raw_output)
        if not extracted_code:
            console.print("[red]Validation failed:[/red] Missing ```python code block.")
            current_prompt = build_reflection_prompt(prompt, raw_output, "Your response did not include a valid ```python ... ``` code block.")
            continue

        # Step 2: Sanitize Code & Class Name
        cleaned_code, class_name_or_err = sanitize_and_prepare_code(extracted_code)
        if cleaned_code is None:
            console.print(f"[red]Validation failed:[/red] {class_name_or_err}")
            current_prompt = build_reflection_prompt(prompt, extracted_code, class_name_or_err)
            continue

        # Step 3: Fast Syntax Verification (Local AST)
        syntax_error = check_syntax(cleaned_code)
        if syntax_error:
            console.print(f"[red]Syntax Check Failed:[/red] {syntax_error}")
            current_prompt = build_reflection_prompt(prompt, cleaned_code, syntax_error)
            continue

        console.print("[green]✓ Syntax & Imports Validated.[/green] Compiling & Rendering scene...")

        # Step 4: Render Video via Manim
        render_result = render_manim_scene(cleaned_code, class_name_or_err, OUTPUT_DIR)

        if not render_result["ok"]:
            console.print(f"[red]Rendering Execution Error:[/red]\n{render_result['error']}")
            current_prompt = build_reflection_prompt(prompt, cleaned_code, render_result['error'])
            continue

        console.print("[green]✓ Video rendered successfully![/green] Passing MP4 to VLM visual evaluator...")

        # Step 5: VLM Review of Rendered Video
        try:
            vlm_result = run_vlm(render_result["media_file"], prompt)
        except Exception as exc:
            vlm_result = {"status": "ERROR", "valid": False, "feedback": str(exc)}

        if vlm_result["status"] == "SUCCESS" and vlm_result["valid"]:
            console.print("[bold green]✓ Visual evaluation passed successfully![/bold green]")
            console.print(f"[green]Saved Output Video:[/green] {render_result['media_file']}")
            save_to_dataset(prompt=prompt, code=cleaned_code)
            return True

        feedback = vlm_result.get("feedback") or vlm_result.get("error", "Unknown visual check failure.")
        console.print(f"[yellow]⚠️ VLM Visual Evaluation Failed:[/yellow] {feedback}")
        current_prompt = build_reflection_prompt(prompt, cleaned_code, f"Visual Review Feedback: {feedback}")

    console.print("[red]Reflection loop reached maximum retries without generating a passing scene.[/red]")
    return False


def run_interactive(max_retries: int) -> None:
    console.print("[bold green]Interactive Mode Active.[/bold green] Type 'exit' to quit.\n")
    while True:
        prompt_text = Prompt.ask("[cyan]Quantum animation prompt[/cyan]")
        if prompt_text.strip().lower() in {"exit", "quit"}:
            break
        if prompt_text.strip():
            process_prompt(prompt_text, max_retries=max_retries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum-Manim-AI Pipeline")
    parser.add_argument("-p", "--prompt", help="Quantum prompt to generate.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive terminal mode.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-correction loops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_banner()

    if not args.prompt and not args.interactive:
        console.print("[yellow]Provide --prompt or --interactive to begin.[/yellow]")
        sys.exit(0)

    if args.prompt:
        success = process_prompt(args.prompt, args.max_retries)
        sys.exit(0 if success else 1)

    run_interactive(args.max_retries)


if __name__ == "__main__":
    main()