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
PROMPT_INSTRUCTIONS = (
    "Generate a runnable Manim scene in Python using Manim v0.20 syntax. "
    "Start with `from manim import *` or explicit Manim imports, define exactly one Scene subclass, "
    "and do not call `.render()` at the module level. Use `self.play(Create(...))`, `FadeOut(...)`, "
    "and avoid deprecated calls like `ShowCreation` and `ShowCreationThenFadeOut`. "
    "If you need a fade-out effect, use `self.play(FadeOut(mobject))` after `Create(...)`. "
    "Return only a single ```python ... ``` code block containing the entire script."
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


def generate_manim_code(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    model, tokenizer = load_qwen_model(model_name)
    messages = [{"role": "user", "content": f"{PROMPT_INSTRUCTIONS}\n\nPrompt:\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True)
    inputs = serialize_token_inputs(inputs)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
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
                        "Check for mathematical correctness:\n"
                        "1. Does the object move/rotate in the correct direction?\n"
                        "2. Is the final state accurate?\n\n"
                        "Respond STRICTLY as JSON: {\"valid\": bool, \"feedback\": \"...\"}"
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

    try:
        data = json.loads(output_text)
        if "valid" in data:
            return {
                "status": "SUCCESS",
                "valid": bool(data["valid"]),
                "feedback": str(data.get("feedback", "")),
            }
        return {"status": "ERROR", "valid": False, "feedback": "", "error": "VLM JSON output missing 'valid' key."}
    except Exception as exc:
        return {"status": "ERROR", "valid": False, "feedback": "", "error": f"Failed to parse VLM output as JSON: {exc}: {output_text}"}


def extract_code(raw_text: str) -> str:
    match = CODE_BLOCK_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def check_magic_line_classname(code: str) -> tuple[Optional[str], Optional[str]]:
    class_match = CLASS_RE.search(code)
    if not class_match:
        return None, "No 'class Foo(Scene):' definition found in the generated code."

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


IMPORT_CHECK_RE = re.compile(r"^(?:from\s+manim\s+import\s+.*|import\s+manim.*)$", re.MULTILINE)


def ensure_manim_imports(code: str) -> str:
    # Always add a broad Manim wildcard import if no `from manim import *` is present.
    # This prevents NameError for color constants and utility names like CYAN, LEFT, RIGHT, etc.
    if "from manim import *" in code:
        return code
    return "from manim import *\n\n" + code


def check_syntax(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"


def render_manim_scene(code: str, class_name: str, output_root: Path) -> dict:
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
                result["error"] = "Manim completed without producing an MP4 artifact."
        else:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            result["error"] = stderr[-4000:]

    return result


def build_reflection_prompt(prompt: str, code: str, error: str) -> str:
    return (
        "The previously generated Manim code failed to compile or render. "
        "Please fix the code and return the full corrected script inside ```python ... ```." 
        f"\n\nOriginal prompt:\n{prompt}\n\nGenerated code:\n```python\n{code}\n```\n\nError:\n{error}"
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
                console.print(f"[yellow]⚠️ VLM Review Skipped:[/yellow] {vlm_result['error']}")
                console.print("[bold green][✓][/bold green] Animation rendered successfully! Output saved to ./output/.")
                console.print(f"[green]File:[/green] {render_result['media_file']}")
                return True

            if vlm_result["valid"] is False:
                feedback = vlm_result["feedback"]
                console.print(f"[red]❌ Visual Review Failed:[/red] {feedback}")
                current_prompt = (
                    f"The following Manim code was generated for the prompt: '{prompt}'\n\n"
                    f"```python\n{cleaned_code}\n```\n\n"
                    f"When rendering with Manim, it compiled successfully, but a visual review detected this error:\n{feedback}\n\n"
                    f"Please analyze the error, fix the code, and output the complete corrected script inside ```python ... ```.")
                continue

            console.print("[bold green][✓][/bold green] Passed visual evaluation and animation rendered successfully! Output saved to ./output/.")
            console.print(f"[green]File:[/green] {render_result['media_file']}")
            return True

        console.print("[bold yellow][4/4][/bold yellow] Reflection loop triggered (if compilation fails) -> Attempting self-correction...")
        error_message = render_result["error"] or "Unknown rendering failure."
        console.print(f"[red]Render failure:[/red] {error_message}")

        if "NameError: name 'ShowCreation' is not defined" in error_message:
            error_message += (
                "\nHint: Use `Create(qubit_state)` or `self.play(Create(qubit_state))` "
                "instead of deprecated `ShowCreation`."
            )
        elif "NameError: name 'ShowCreationThenFadeOut' is not defined" in error_message:
            error_message += (
                "\nHint: Use `self.play(Create(qubit_state))` followed by `self.play(FadeOut(qubit_state))`."
            )

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
