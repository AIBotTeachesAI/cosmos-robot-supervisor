#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import transformers


LEVEL1_SYSTEM = """
You are a strict video classifier for a robot demo.
You must output ONLY valid JSON.
No extra text.
""".strip()

LEVEL1_USER = """
Classify this entire video into EXACTLY ONE label:

Labels (choose one):
1) "OBSTRUCTION"  = a stuffed toy / plush / soft toy is visible at any time in the video.
2) "SUCCESS_LIFT" = the orange block is clearly lifted off the surface and held in the gripper/in air at any time with gripper closed.
3) "MISS_GRASP"   = the orange block is not visible on the table/surface (released/falls/down)


Output JSON only, with this schema:
{
  "label": "OBSTRUCTION|SUCCESS_LIFT|MISS_GRASP",
  "evidence": ["short bullet-like observations from the video"],
  "confidence": 0.0
}

Rules:
- If toy is visible even once -> OBSTRUCTION (ignore everything else).
""".strip()


def robust_json_parse(raw: str) -> dict:
    raw2 = (raw or "").strip()
    raw2 = raw2.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw2)
        if not isinstance(obj, dict):
            raise ValueError("Not a JSON object")
        return obj
    except Exception:
        return {"label": "MISS_GRASP", "evidence": ["json_parse_failed"], "confidence": 0.0}


def classify_video(model, processor, video_path: Path, fps: int, max_new_tokens: int) -> tuple[dict, str]:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": LEVEL1_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": LEVEL1_USER},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated, strict=False)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    parsed = robust_json_parse(raw)

    # normalize a bit
    label = str(parsed.get("label", "MISS_GRASP")).strip().upper()
    if label not in {"OBSTRUCTION", "SUCCESS_LIFT", "MISS_GRASP"}:
        label = "MISS_GRASP"
    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    parsed = {"label": label, "evidence": evidence[:6], "confidence": conf}
    return parsed, raw


def iter_videos(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".mp4":
        return [input_path]
    if input_path.is_dir():
        vids = sorted(input_path.glob("*.mp4"))
        return vids
    raise SystemExit(f"Input path not found or not supported: {input_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a .mp4 or a folder containing .mp4")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transformers.set_seed(args.seed)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)

    videos = iter_videos(in_path)
    if not videos:
        raise SystemExit(f"No .mp4 files found in: {in_path}")

    print(f"Found {len(videos)} video(s). Writing to: {out_path.relative_to(Path.cwd())}")

    with out_path.open("w", encoding="utf-8") as f:
        for i, vp in enumerate(videos, 1):
            parsed, raw = classify_video(model, processor, vp, fps=args.fps, max_new_tokens=args.max_new_tokens)
            rec = {
                "video": str(vp.relative_to(Path.cwd())),
                "fps": args.fps,
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "evidence": parsed["evidence"],
                "raw": raw,  # keep for debugging
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(videos)}] {vp.name} -> {parsed['label']} (conf={parsed['confidence']:.2f})")

    print("Done.")


if __name__ == "__main__":
    main()


