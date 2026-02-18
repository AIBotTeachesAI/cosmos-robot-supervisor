#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers

LEVEL2_SYSTEM = """
You are a robot monitoring supervisor.

You will receive ONE Level-1 classification result for an entire video.

Input format:
{
  "label": "SUCCESS_LIFT | OBSTRUCTION | MISS_GRASP",
  "confidence": 0-1,
  "evidence": [string, ...]
}

Your task:
1. Choose the action based ONLY on the label:
   - SUCCESS_LIFT → PROCEED
   - OBSTRUCTION → ABORT
   - MISS_GRASP → REPLAN

2. In "why", concisely restate the evidence provided.
   - Do NOT add new interpretation.
   - Do NOT invent causes.
   - Do NOT explain the mapping logic.

Output ONLY valid JSON:
{"action":"PROCEED|ABORT|REPLAN","why":"...","confidence":0-1}
""".strip()


def read_single_json_or_jsonl(path: Path) -> Dict[str, Any]:
    """
    Accepts:
      - a .json file containing one object
      - a .jsonl file (we take the last non-empty line as the latest record)
    Returns one dict.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty file: {path}")

    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: JSONL
    last_obj: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last_obj = json.loads(line)
    if not isinstance(last_obj, dict):
        raise ValueError(f"Could not parse a JSON object from: {path}")
    return last_obj


def parse_json_strict(raw: str) -> Dict[str, Any]:
    """
    Parses model output that might be wrapped in ```json ... ``` fences.
    """
    s = (raw or "").strip()
    s = s.replace("```json", "").replace("```", "").strip()
    return json.loads(s)


def run_level2(
    model,
    processor,
    level1_obj: Dict[str, Any],
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> Dict[str, Any]:
    # Keep only the fields we claim to provide
    inp = {
        "label": level1_obj.get("label"),
        "confidence": level1_obj.get("confidence"),
        "evidence": level1_obj.get("evidence", []),
    }

    user_text = "Input:\n" + json.dumps(inp, ensure_ascii=False, indent=2)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": LEVEL2_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        # If you ever set do_sample=True, you can optionally add temperature/top_p here.
        generated_ids = model.generate(**gen_kwargs)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    try:
        out = parse_json_strict(raw)
    except Exception:
        out = {
            "action": "REPLAN",
            "why": "Model output was not valid JSON.",
            "confidence": 0.0,
            "raw": raw,
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
        "outputs/approach1/level1_fullvideo_success.jsonl",
        "outputs/approach1/level1_fullvideo_obstruction.jsonl",
        "outputs/approach1/level1_fullvideo_push_down.jsonl",
        ],
        help="Level-1 output files (.json or .jsonl). You can pass 1+ files.",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/approach1",
        help="Directory to write Level-2 outputs.",
    )
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument(
        "--do_sample",
        action="store_true",
        help="If set, enables sampling (non-deterministic). Default is deterministic.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(0)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)

    summary: List[Dict[str, Any]] = []

    for in_path_str in args.inputs:
        in_path = Path(in_path_str).expanduser().resolve()
        level1_obj = read_single_json_or_jsonl(in_path)

        # Helpful name for output files        
        name = in_path.name.replace("level1_", "").replace(".jsonl", "").replace(".json", "")
        out_path = out_dir / f"level2_{name}.level2.json"

        out = run_level2(
            model=model,
            processor=processor,
            level1_obj=level1_obj,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
        )

        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({
         "input": str(in_path.relative_to(Path.cwd())),
         "output": str(out_path.relative_to(Path.cwd())),
         "result": out
       })


        print(f"[OK] {in_path.name} -> {out_path.name}")
        print(json.dumps(out, indent=2))
        print("-" * 60)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary: {(out_dir / 'summary.json').relative_to(Path.cwd())}")



if __name__ == "__main__":
    main()


