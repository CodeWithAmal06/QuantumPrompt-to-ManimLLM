#!/usr/bin/env python3
"""
validate_manim_dataset.py (merged version)

Renders every Manim script in a JSONL instruction-tuning dataset headlessly
to confirm it actually executes before it's kept in the training data.

This version keeps two things from a hand-written first attempt:
  - extract_code(): pulls the ```python ... ``` block out of the assistant
    message (used as-is, it was already correct).
  - The idea of cross-checking that the %%manim magic line's class name
    actually matches the class defined in the code -- a real failure mode
    worth catching that the original version didn't check for.

Usage:
    python validate_manim_dataset.py
    python validate_manim_dataset.py --timeout 90 --workers 4
    python validate_manim_dataset.py --input-jsonl /path/to/input.jsonl

If Google Drive is not mounted, the script will fall back to a local input.jsonl file in the workspace.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


CODE_BLOCK_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
MAGIC_LINE_RE = re.compile(r"^%%manim\s+(?:-\S+\s+)*(\S+)\s*$", re.MULTILINE)
CLASS_RE = re.compile(r"class\s+(\w+)\s*\(")
DEFAULT_VLM_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
VLM_STATE = {"model": None, "tokenizer": None, "process_vision_info": None, "torch": None, "error": None}



def extract_code(assistant_content: str) -> str | None:
    """Pull the code out of a ```python ... ``` block. (unchanged from the
    original attempt -- this was already correct.)"""
    match = CODE_BLOCK_RE.search(assistant_content)
    return match.group(1) if match else None


def check_magic_line_classname(idx: int, code: str) -> tuple[str, str] | None:
    """Cross-check the %%manim magic line's class name against the actual
    'class Foo(Scene):' definition in the code. Returns (cleaned_code,
    class_name) on success, or None (after printing why) on failure.

    This also handles the case where there's no magic line at all --
    that's fine, we just fall back to whatever CLASS_RE finds.
    """
    class_match = CLASS_RE.search(code)
    if not class_match:
        print(f"Row {idx}: no 'class Foo(Scene):' definition found")
        return None
    actual_class_name = class_match.group(1)

    magic_match = MAGIC_LINE_RE.search(code)
    if not magic_match:
        # No magic line present -- not an error, just nothing to cross-check.
        # The code is used as-is.
        return code, actual_class_name

    magic_class_name = magic_match.group(1)
    if magic_class_name != actual_class_name:
        print(f"Row {idx}: magic line says '{magic_class_name}' but the "
              f"class is actually named '{actual_class_name}' -- mismatch")
        return None

    # Magic line matches. Strip it out (plus a following blank line, if any)
    # since "%%manim ..." isn't valid standalone Python syntax.
    lines = code.splitlines()
    rest = lines[1:]
    if rest and rest[0].strip() == "":
        rest = rest[1:]
    cleaned_code = "\n".join(rest)
    return cleaned_code, actual_class_name


def check_syntax(code: str) -> str | None:
    """Fast pre-check with ast -- catches broken syntax without paying for
    a full manim process spin-up."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"


def has_required_keys(py_dict: dict) -> str | None:
    """Validate structure on the *parsed dict*, not the raw text line.
    Returns an error string if something's missing, else None."""
    if "messages" not in py_dict:
        return "missing 'messages' key"
    messages = py_dict["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return "'messages' must be a list with at least 2 entries"
    for i, m in enumerate(messages[:2]):
        if "role" not in m or "content" not in m:
            return f"messages[{i}] missing 'role' or 'content'"
    return None


def load_vlm_if_available(model_name: str = DEFAULT_VLM_MODEL):
    """Load the Unsloth vision-language model lazily when the optional deps exist."""
    if VLM_STATE["model"] is not None and VLM_STATE["tokenizer"] is not None:
        return VLM_STATE["model"], VLM_STATE["tokenizer"]

    if VLM_STATE["error"] is not None:
        return None, None

    try:
        from unsloth import FastLanguageModel
        from qwen_vl_utils import process_vision_info
        import torch
    except Exception as exc:
        VLM_STATE["error"] = str(exc)
        return None, None

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        VLM_STATE["model"] = model
        VLM_STATE["tokenizer"] = tokenizer
        VLM_STATE["process_vision_info"] = process_vision_info
        VLM_STATE["torch"] = torch
        return model, tokenizer
    except Exception as exc:
        VLM_STATE["error"] = str(exc)
        return None, None


def parse_vlm_response(raw_text: str) -> dict:
    """Parse the JSON returned by the VLM reviewer and normalize it to a dict."""
    if not raw_text:
        return {"status": "ERROR", "valid": False, "feedback": "", "error": "Empty VLM response."}

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return {
            "status": "OK",
            "valid": bool(payload.get("valid", False)),
            "feedback": str(payload.get("feedback", "")),
            "error": None,
        }

    return {"status": "ERROR", "valid": False, "feedback": "", "error": f"VLM response was not valid JSON: {raw_text}"}


def run_vlm(video_path: str, prompt: str, model_name: str = DEFAULT_VLM_MODEL) -> dict:
    """Ask a vision-language model to review a rendered Manim video against the prompt."""
    if not os.path.exists(video_path):
        return {"status": "ERROR", "valid": False, "feedback": "", "error": f"Video file not found: {video_path}"}

    model, tokenizer = load_vlm_if_available(model_name)
    if model is None or tokenizer is None:
        return {
            "status": "ERROR",
            "valid": False,
            "feedback": "",
            "error": f"VLM unavailable: {VLM_STATE['error'] or 'Install unsloth and qwen_vl_utils to enable visual review.'}",
        }

    try:
        process_vision_info = VLM_STATE["process_vision_info"]
        torch = VLM_STATE["torch"]
        messages = [{
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
        }]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(next(model.parameters()).device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_vlm_response(decoded_output)
    except Exception as exc:
        return {"status": "ERROR", "valid": False, "feedback": "", "error": f"VLM execution failed: {exc}"}


def render_one(index: int, code: str, class_name: str, timeout: int, workdir: Path) -> dict:
    """Write the script to a temp file and render only the final frame at
    low quality -- enough to execute every line of construct() without
    paying for full video encoding."""
    result = {"index": index, "class_name": class_name, "ok": False, "error": None, "media_file": None}

    # NOTE: encoding="utf-8" is required here. Without it, write_text() falls
    # back to the OS default encoding -- which is cp1252 on Windows, not
    # UTF-8 -- and any script containing non-Latin-1 characters (emoji,
    # special math symbols, etc.) raises UnicodeEncodeError.
    scene_file = workdir / f"scene_{index}.py"
    media_dir = workdir / f"media_{index}"

    try:
        scene_file.write_text(code, encoding="utf-8")
    except Exception as e:
        # Catch-all: a failure writing the file (bad encoding or otherwise)
        # should be recorded as a normal failed row, not crash the whole
        # parallel run by propagating out of this worker.
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    # Run manim as "python -m manim" through the SAME interpreter running
    # this script, rather than relying on a bare "manim" command resolved
    # via PATH. On uv-managed venvs, "manim" on PATH is a tiny trampoline
    # .exe that has to resolve its own file location to find the venv's
    # python.exe -- and that resolution can fail (e.g. "uv trampoline failed
    # to canonicalize script path"), especially under multiprocessing or
    # inside synced folders like OneDrive. Going through sys.executable
    # skips that layer entirely and talks to the real interpreter directly.
    cmd = [
        sys.executable, "-m", "manim", "-ql", "-s", "--disable_caching",
        "--media_dir", str(media_dir),
        str(scene_file), class_name,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            result["ok"] = True
            mp4_files = list(media_dir.glob("**/*.mp4"))
            if mp4_files:
                output_video_path = workdir / f"scene_{index}.mp4"
                shutil.copy(mp4_files[0], output_video_path)
                result["media_file"] = str(output_video_path)
        else:
            tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
            result["error"] = tail or proc.stdout.strip()[-2000:]
    except subprocess.TimeoutExpired:
        result["error"] = f"Timed out after {timeout}s"
    except FileNotFoundError:
        result["error"] = "manim executable not found on PATH -- is it installed?"
    finally:
        shutil.rmtree(media_dir, ignore_errors=True)

    return result


def validate_row(index: int, py_dict: dict, timeout: int, workdir: Path, enable_vlm: bool = False, vlm_model_name: str = DEFAULT_VLM_MODEL) -> dict:
    # Top-level safety net: NO exception from anything below should ever
    # propagate out of this function. A worker process crashing on one bad
    # row would otherwise take down the entire ProcessPoolExecutor run via
    # future.result() in main() -- we always want a per-row result dict back.
    try:
        key_error = has_required_keys(py_dict)
        if key_error:
            return {"index": index, "ok": False, "error": key_error, "class_name": None}

        assistant_content = py_dict["messages"][1]["content"]
        code = extract_code(assistant_content)
        if code is None:
            return {"index": index, "ok": False, "error": "No ```python code block found", "class_name": None}

        checked = check_magic_line_classname(index, code)
        if checked is None:
            return {"index": index, "ok": False, "error": "Magic-line/class-name check failed (see log above)", "class_name": None}
        cleaned_code, class_name = checked

        syntax_error = check_syntax(cleaned_code)
        if syntax_error:
            return {"index": index, "ok": False, "error": syntax_error, "class_name": class_name}

        render_result = render_one(index, cleaned_code, class_name, timeout, workdir)
        if not render_result["ok"]:
            return {"index": index, "ok": False, "error": render_result["error"], "class_name": class_name}

        if enable_vlm and render_result.get("media_file"):
            prompt = py_dict.get("messages", [{}])[0].get("content", "") if py_dict.get("messages") else ""
            vlm_result = run_vlm(render_result["media_file"], prompt, vlm_model_name)
            if vlm_result.get("status") == "ERROR":
                return {"index": index, "ok": False, "error": f"VLM validation failed: {vlm_result.get('error')}", "class_name": class_name}
            if vlm_result.get("valid") is False:
                return {"index": index, "ok": False, "error": f"VLM visual review failed: {vlm_result.get('feedback') or 'No feedback returned.'}", "class_name": class_name}

        return render_result
    except Exception as e:
        return {"index": index, "ok": False, "error": f"Unexpected {type(e).__name__}: {e}", "class_name": None}


def resolve_dataset_path(explicit_path: str | None) -> Path:
    """Resolve the dataset path from an explicit argument or a list of common fallback locations."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    drive_candidates = [
        Path("/content/drive/MyDrive/input.jsonl"),
        Path("/mnt/drive/MyDrive/input.jsonl"),
        Path("C:/content/drive/MyDrive/input.jsonl"),
        Path("C:/mnt/drive/MyDrive/input.jsonl"),
    ]
    candidates.extend(drive_candidates)

    script_dir = Path(__file__).resolve().parent
    workspace_candidates = [
        script_dir / "input.jsonl",
        Path.cwd() / "input.jsonl",
        Path("input.jsonl"),
    ]
    candidates.extend(workspace_candidates)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if explicit_path:
        return Path(explicit_path).expanduser()
    return (script_dir / "input.jsonl").resolve()


def main():
    parser = argparse.ArgumentParser(description="Validate a Manim JSONL dataset by actually rendering each row.")
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Path to the JSONL dataset file. If omitted, the script tries the Google Drive path first and then falls back to a local input.jsonl file.",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--enable-vlm", action="store_true", help="Run an optional Unsloth vision-language review on each rendered video.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="Vision-language model to use for visual review.")
    args = parser.parse_args()

    input_path = resolve_dataset_path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {input_path}. "
            "Make sure the file exists in your Google Drive mount or pass --input-jsonl with a different path."
        )

    print(f"Using dataset file: {input_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_path, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            try:
                py_dict = json.loads(line)  # loads(), not load() -- line is a string
                rows.append(py_dict)
            except json.JSONDecodeError:
                print(f"Corrupted JSON on line {idx + 1}, skipping")

    print(f"Loaded {len(rows)} rows. Validating with {args.workers} workers "
          f"(timeout={args.timeout}s each)...")

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(validate_row, i, row, args.timeout, workdir, args.enable_vlm, args.vlm_model): i
                for i, row in enumerate(rows)
            }
            for done_count, future in enumerate(as_completed(futures), 1):
                res = future.result()
                results[res["index"]] = res
                status = "OK " if res["ok"] else "FAIL"
                print(f"[{done_count}/{len(rows)}] row {res['index']:>3} "
                      f"({res.get('class_name') or '?'}): {status}")

    validated_rows, failed_rows = [], []
    for i, row in enumerate(rows):
        res = results[i]
        if res["ok"]:
            validated_rows.append(row)
        else:
            failed_row = dict(row)
            failed_row["_validation_error"] = res["error"]
            failed_row["_class_name"] = res.get("class_name")
            failed_rows.append(failed_row)

    with open(outdir / "validated.jsonl", "w", encoding="utf-8") as f:
        for row in validated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(outdir / "failed.jsonl", "w", encoding="utf-8") as f:
        for row in failed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(outdir / "validation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Validation report for {input_path}\n")
        f.write(f"Total rows: {len(rows)}\nPassed: {len(validated_rows)}\nFailed: {len(failed_rows)}\n\n")
        f.write("FAILURES:\n\n")
        for i, row in enumerate(rows):
            res = results[i]
            if not res["ok"]:
                f.write(f"--- Row {i} ({res.get('class_name') or '?'}) ---\n{res['error']}\n\n")

    print(f"\nDone. {len(validated_rows)}/{len(rows)} rows passed.")
    print(f"Wrote: {outdir/'validated.jsonl'}, {outdir/'failed.jsonl'}, {outdir/'validation_report.txt'}")


if __name__ == "__main__":
    main()