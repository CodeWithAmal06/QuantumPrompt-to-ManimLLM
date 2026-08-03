#!/usr/bin/env python3
"""Quantum-Manim-AI CLI

Core command-line entry point for generating Manim animations from quantum prompts,
validating the generated Python scene, compiling it through Manim, and retrying with
self-correction when rendering fails.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

try:
    from IPython.display import Video, display
except ImportError:
    Video = None
    display = None

console = Console()

CODE_BLOCK_RE = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)
MAGIC_LINE_RE = re.compile(r"^%%manim\s+(?:-\S+\s+)*(\S+)\s*$", re.MULTILINE)
CLASS_RE = re.compile(r"class\s+(\w+)\s*\(")
DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

# Path for saving curated successful generations to Google Drive / local storage
DATASET_PATH = Path("/content/drive/MyDrive/input.jsonl")

PROMPT_INSTRUCTIONS = (
    "Generate a runnable Manim scene in Python using Manim v0.20 syntax.\n"
    "CRITICAL REQUIREMENTS FOR QUANTUM ANIMATIONS:\n"
    "1. Define exactly ONE subclass of `ThreeDScene` (NOT standard `Scene`).\n"
    "2. Setup 3D camera and axes in `construct()`:\n"
    "   axes = ThreeDAxes()\n"
    "   self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)\n"
    "3. Define a state vector (e.g., Arrow3D or Line with arrow head) starting along the Z-axis for |0>.\n"
    "4. Avoid deprecated calls like `ShowCreation`, `ShowCreationThenFadeOut`, or module-level `.render()` calls.\n"
    "5. DYNAMIC ROTATION IS MANDATORY: You MUST visually animate the vector moving. Use `self.play(Rotate(vector, angle=..., axis=...))` or `self.play(Transform(...))` to show the gate action.\n"
    "6. Do NOT just morph static text labels. The 3D vector MUST rotate in space.\n\n"
    "MINIMAL WORKING PATTERN EXAMPLE:\n"
    "```python\n"
    "from manim import *\n\n"
    "class QuantumScene(ThreeDScene):\n"
    "    def construct(self):\n"
    "        axes = ThreeDAxes()\n"
    "        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)\n"
    "        sphere = Sphere(radius=2, fill_opacity=0.1)\n"
    "        vector = Arrow3D(start=ORIGIN, end=OUT*2, color=BLUE)\n"
    "        self.add(axes, sphere, vector)\n"
    "        self.wait(0.5)\n"
    "        # Rotate vector to show gate transformation\n"
    "        self.play(Rotate(vector, angle=PI/2, axis=RIGHT, run_time=2))\n"
    "        self.wait(1)\n"
    "```\n"
    "Return ONLY a single ```python ... ``` code block containing the complete script."
)

OUTPUT_DIR = Path("output")
MODEL_CACHE = {"model": None, "tokenizer": None}
VLM_CACHE = {"model": None, "tokenizer": None, "process_vision_info": None}
DEFAULT_VLM_MODEL = DEFAULT_MODEL_NAME


def show_banner() -> None:
    banner = Panel(
        "[bold cyan]Quantum-Manim-AI[/bold cyan]\n"
        "[white]Translate quantum computing prompts into Manim animations using a fine-tuned Qwen-2.5-7B pipeline.[/white]",
        title="[bold green]Quantum-Manim-AI[/bold green]",
        border_style="bright_blue",
    )
    console.print(banner)


def save_to_dataset(prompt: str, code: str, filepath: Path = DATASET_PATH) -> None:
    """Appends successful prompt-code pairs to JSONL file with duplicate prevention."""
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


def load_qwen_model(model_name: str = DEFAULT_MODEL_NAME) -> tuple[object, object]:
    if MODEL_CACHE["model"] is not None and MODEL_CACHE["tokenizer"] is not None:
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install unsloth to generate Manim code from Qwen2.5."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    MODEL_CACHE["model"] = model
    MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def serialize_token_inputs(tokens: dict) -> dict:
    result = {}
    for key, value in tokens.items():
        if hasattr(value, "to"):
            result[key] = value.to(next(MODEL_CACHE["model"].parameters()).device)
        else:
            result[key] = value
    return result


def generate_manim_code(prompt: str, model_name: str = DEFAULT_MODEL_NAME, attempt: int = 1) -> str:
    model, tokenizer = load_qwen_model(model_name)
    messages = [{"role": "user", "content": f"{PROMPT_INSTRUCTIONS}\n\nPrompt:\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True)
    inputs = serialize_token_inputs(inputs)
    
    # Scale temperature higher on retries (e.g. attempt 1 = 0.2, attempt 2 = 0.6, attempt 3 = 0.8)
    # Higher temperature introduces variability so Qwen doesn't repeat the same static 2D code.
    temperature = 0.2 if attempt == 1 else min(0.2 + (attempt - 1) * 0.3, 0.8)
    do_sample = temperature > 0.0

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        temperature=temperature,
        do_sample=do_sample
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
        data = json.loads(cleaned_text)
    except json.JSONDecodeError:
        valid_match = re.search(r'"valid"\s*:\s*(true|false)', cleaned_text, re.IGNORECASE)
        feedback_match = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned_text)
        if valid_match and feedback_match:
            try:
                feedback_value = json.loads(f'"{feedback_match.group(1)}"')
            except Exception:
                feedback_value = feedback_match.group(1)
            return {
                "status": "SUCCESS",
                "valid": valid_match.group(1).lower() == "true",
                "feedback": str(feedback_value),
            }
        return {
            "status": "ERROR",
            "valid": False,
            "feedback": "",
            "error": f"Failed to parse VLM output as JSON: {cleaned_text}",
        }

    if isinstance(data, dict) and "valid" in data:
        return {
            "status": "SUCCESS",
            "valid": bool(data["valid"]),
            "feedback": str(data.get("feedback", "")),
        }

    return {"status": "ERROR", "valid": False, "feedback": "", "error": "VLM JSON output missing 'valid' key."}


def run_vlm(video_path: str, prompt: str, model_name: str = DEFAULT_VLM_MODEL) -> dict:
    if not os.path.exists(video_path):
        return {"status": "ERROR", "valid": False, "feedback": "", "error": f"Video file not found: {video_path}"}

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
                        f"Analyze this rendered Manim animation for the prompt: '{prompt}'.\n"
                        "Check whether the animation is a valid, pedagogical quantum visual.\n"
                        "STRICT EVALUATION RULES:\n"
                        "1. STAGNANT VECTOR TEST: Does the state vector/arrow actually ROTATE or MOVE across space? "
                        "If the arrow remains stationary while only text labels change, set 'valid': false.\n"
                        "2. DIMENSIONALITY TEST: For gate operations like Hadamard, Pauli-X/Y/Z, or phase shifts, "
                        "is there a 3D Bloch sphere or coordinate system shown? If it is a flat 2D line with no dynamic rotation, set 'valid': false.\n"
                        "3. VISIBILITY TEST: Is text overlapping and obscuring the central state vector? If text blocks the view, set 'valid': false.\n"
                        "4. ACCURACY: Does the vector movement accurately represent the requested quantum gate transition?\n\n"
                        "Respond STRICTLY in JSON format: {\"valid\": true/false, \"feedback\": \"detailed explanation of failure or success\"}"
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

    parsed_output = parse_vlm_output(output_text)
    if parsed_output["status"] == "SUCCESS":
        return parsed_output

    return {
        "status": "ERROR",
        "valid": False,
        "feedback": parsed_output.get("feedback", ""),
        "error": parsed_output.get("error", f"Failed to parse VLM output: {output_text}"),
    }


def extract_code(raw_text: str) -> str:
    match = CODE_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def check_magic_line_classname(code: str) -> tuple[Optional[str], Optional[str]]:
    class_match = CLASS_RE.search(code)
    if not class_match:
        return None, "No 'class Foo(Scene):' or 'class Foo(ThreeDScene):' definition found in the generated code."

    actual_class_name = class_match.group(1)
    magic_match = MAGIC_LINE_RE.search(code)
    if not magic_match:
        return code, actual_class_name

    magic_class_name = magic_match.group(1)
    if magic_class_name != actual_class_name:
        return None, (
            f"%%manim metadata claims class '{magic_class_name}', "
            f"but the code defines '{actual_class_name}'."
        )

    cleaned_lines = [line for line in code.splitlines() if not MAGIC_LINE_RE.match(line)]
    return "\n".join(cleaned_lines).strip(), actual_class_name


def ensure_manim_imports(code: str) -> str:
    if "from manim import *" in code:
        return code
    return "from manim import *\n\n" + code


def check_syntax(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"


def tail_file(path: Path, lines: int = 100) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            all_lines = handle.readlines()
        return "".join(all_lines[-lines:]).strip()
    except Exception as exc:
        return f"Could not read log file {path}: {exc}"


def render_manim_scene(code: str, class_name: str, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    result = {"ok": False, "error": None, "media_file": None, "log_file": None}

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
                result["error"] = "Manim completed without producing an MP4 artifact."
        else:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            error_text = stderr
            log_match = re.search(r"log file:\s*(.+\.log)", stderr)
            if log_match:
                log_path = Path(log_match.group(1).strip())
                if not log_path.is_absolute():
                    log_path = output_root / log_path
                if log_path.exists():
                    result["log_file"] = str(log_path.resolve())
                    error_text += "\n\n--- LaTeX log excerpt ---\n"
                    error_text += tail_file(log_path, lines=80)
            result["error"] = error_text[-4000:]

    return result


def build_reflection_prompt(prompt: str, code: str, error: str) -> str:
    # Auto-translate abstract VLM complaints into concrete Manim instructions
    hints = []
    if "Stagnant Vector Test" in error or "stationary" in error or "no rotation" in error:
        hints.append(
            "FIX REQUIRED: You failed to animate vector movement. "
            "Use `self.play(Rotate(vec, angle=PI/2, axis=...))` or `self.play(Transform(vec1, vec2))` "
            "so the vector physically rotates in space."
        )
    if "Dimensionality Test" in error or "Bloch sphere" in error or "3D" in error:
        hints.append(
            "FIX REQUIRED: Use `ThreeDScene` instead of `Scene`. "
            "Add `ThreeDAxes()`, `Sphere(...)`, and set camera orientation using `self.set_camera_orientation(phi=75*DEGREES, theta=-45*DEGREES)`."
        )

    hint_str = "\n".join(hints) if hints else "Fix all compilation or visual errors."

    return (
        "The previous code failed visual or code validation.\n"
        f"CRITICAL REMEDIES TO APPLY:\n{hint_str}\n\n"
        f"Original Prompt: {prompt}\n\n"
        f"Failed Code:\n```python\n{code}\n```\n\n"
        f"Validation Feedback:\n{error}\n\n"
        "Provide a completely rewritten script fixing these exact issues inside ```python ... ```."
    )


def process_prompt(prompt: str, max_retries: int, model_name: str) -> bool:
    current_prompt = prompt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        console.print(f"[bold cyan][1/4][/bold cyan] Generating Manim code from Qwen2.5-7B... (attempt {attempt})")
        try:
            raw_output = generate_manim_code(current_prompt, model_name=model_name)
        except Exception as exc:
            console.print(f"[red]Model generation failed:[/red] {exc}")
            return False

        console.print("[bold cyan][2/4][/bold cyan] Validating code syntax...")
        code = extract_code(raw_output)
        if not code:
            console.print("[red]No Python code block was found in the model output.[/red]")
            current_prompt = build_reflection_prompt(prompt, raw_output, "Missing ```python ... ``` block.")
            continue

        cleaned_code, class_name_or_error = check_magic_line_classname(code)
        if cleaned_code is None:
            console.print(f"[red]Validation failed:[/red] {class_name_or_error}")
            current_prompt = build_reflection_prompt(prompt, code, class_name_or_error)
            continue

        cleaned_code = ensure_manim_imports(cleaned_code)

        syntax_error = check_syntax(cleaned_code)
        if syntax_error:
            console.print(f"[red]Syntax error detected:[/red] {syntax_error}")
            current_prompt = build_reflection_prompt(prompt, cleaned_code, syntax_error)
            continue

        console.print("[bold cyan][3/4][/bold cyan] Compiling Manim animation via system process...")
        render_result = render_manim_scene(cleaned_code, class_name_or_error, OUTPUT_DIR)

        if render_result["ok"] and render_result["media_file"]:
            if render_result["media_file"] and os.path.exists(render_result["media_file"]):
                try:
                    vlm_result = run_vlm(render_result["media_file"], prompt)
                except Exception as exc:
                    vlm_result = {"status": "ERROR", "valid": False, "feedback": "", "error": str(exc)}
            else:
                vlm_result = {"status": "ERROR", "valid": False, "feedback": "", "error": "Rendered file missing."}

            if vlm_result["status"] == "ERROR":
                feedback = vlm_result["feedback"] or vlm_result["error"]
                console.print(f"[yellow]⚠️ VLM Review did not return a usable result:[/yellow] {feedback}")
                current_prompt = build_reflection_prompt(prompt, cleaned_code, feedback)
                continue

            if vlm_result["valid"] is False:
                feedback = vlm_result["feedback"]
                console.print(f"[red]❌ Visual Review Failed:[/red] {feedback}")
                current_prompt = build_reflection_prompt(prompt, cleaned_code, feedback)
                continue

            console.print("[bold green][✓][/bold green] Passed visual evaluation and animation rendered successfully!")
            console.print(f"[green]File:[/green] {render_result['media_file']}")

            # Save valid pair to drive dataset
            save_to_dataset(prompt=prompt, code=cleaned_code)
            return True

        console.print("[bold yellow][4/4][/bold yellow] Reflection loop triggered -> Attempting self-correction...")
        error_message = render_result["error"] or "Unknown rendering failure."
        console.print(f"[red]Render failure:[/red] {error_message}")

        if "NameError: name 'ShowCreation' is not defined" in error_message:
            error_message += "\nHint: Use `Create(...)` or `self.play(Create(...))` instead of deprecated `ShowCreation`."
        elif "NameError: name 'ShowCreationThenFadeOut' is not defined" in error_message:
            error_message += "\nHint: Use `self.play(Create(...))` followed by `self.play(FadeOut(...))`."

        current_prompt = build_reflection_prompt(prompt, cleaned_code, error_message)

    console.print("[red]Reflection loop reached max retries without rendering a usable animation.[/red]")
    return False


def run_interactive(max_retries: int, model_name: str) -> None:
    console.print("[bold green]Entering interactive prompt mode.[/bold green]")
    console.print("Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n")

    while True:
        prompt_text = Prompt.ask("[cyan]Quantum animation prompt[/cyan]")
        if prompt_text.strip().lower() in {"exit", "quit"}:
            console.print("[bold]Goodbye.[/bold]")
            break

        if prompt_text.strip():
            process_prompt(prompt_text, max_retries=max_retries, model_name=model_name)
            console.print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum-Manim-AI CLI: generate, validate, and render Manim scenes from quantum prompts."
    )
    parser.add_argument("-p", "--prompt", help="Quantum animation prompt to generate a Manim scene for.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open an interactive prompt loop for multiple generation requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum self-correction attempts when rendering fails.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Fine-tuned Qwen model name or path to use for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_banner()

    if not args.prompt and not args.interactive:
        console.print("[yellow]Use --prompt or --interactive to start the pipeline.[/yellow]\n")
        console.print("Run with -h for usage examples.")
        sys.exit(0)

    if args.prompt and args.interactive:
        console.print("[red]Error:[/red] Cannot use --prompt and --interactive at the same time.")
        sys.exit(1)

    if args.prompt:
        success = process_prompt(args.prompt, args.max_retries, args.model_name)
        sys.exit(0 if success else 1)

    run_interactive(args.max_retries, args.model_name)


if __name__ == "__main__":
    main()