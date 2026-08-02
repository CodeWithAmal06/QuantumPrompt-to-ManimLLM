#!/usr/bin/env python3
"""Quantum-Manim-AI CLI

Core command-line entry point for generating Manim animations from quantum prompts,
validating the generated Python scene, compiling it through Manim, and retrying with
self-correction when rendering fails.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

CODE_BLOCK_RE = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL | re.IGNORECASE)
MAGIC_LINE_RE = re.compile(r"^%%manim\s+(?:-\S+\s+)*(\S+)\s*$", re.MULTILINE)
CLASS_RE = re.compile(r"class\s+(\w+)\s*\(")
DEFAULT_MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
OUTPUT_DIR = Path("output")
MODEL_CACHE = {"model": None, "tokenizer": None}


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
    instructions = (
        "Generate a runnable Manim scene in Python from the following quantum animation prompt. "
        "Return only a single ```python ... ``` code block with the full scene definition."
    )
    messages = [{"role": "user", "content": f"{instructions}\n\nPrompt:\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True)
    inputs = serialize_token_inputs(inputs)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return decoded.strip()


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

        syntax_error = check_syntax(cleaned_code)
        if syntax_error:
            console.print(f"[red]Syntax error detected:[/red] {syntax_error}")
            current_prompt = build_reflection_prompt(prompt, cleaned_code, syntax_error)
            continue

        console.print("[bold cyan][3/4][/bold cyan] Compiling Manim animation via system process...")
        render_result = render_manim_scene(cleaned_code, class_name_or_error, OUTPUT_DIR)

        if render_result["ok"] and render_result["media_file"]:
            console.print("[bold green][✓][/bold green] Animation rendered successfully! Output saved to ./output/.")
            console.print(f"[green]File:[/green] {render_result['media_file']}")
            return True

        console.print("[bold yellow][4/4][/bold yellow] Reflection loop triggered (if compilation fails) -> Attempting self-correction...")
        error_message = render_result["error"] or "Unknown rendering failure."
        console.print(f"[red]Render failure:[/red] {error_message}")
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
