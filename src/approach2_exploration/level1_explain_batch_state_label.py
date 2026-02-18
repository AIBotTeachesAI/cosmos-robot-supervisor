#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import transformers

PIXELS_PER_TOKEN = 32**2


SYSTEM = """
You are a visual observer.

Describe only what is directly visible in the video.
Do not judge success or failure.
Do not infer intent.
Do not assume grasping unless clearly visible.
If something is unclear, say so.

"""

USER = """
This is a wrist-camera video clip. Orange block can be dropped down by the gripper
or held by the gripper.
The orange block can also disappear. Pay close attention and report correctly. 

Please describe, in plain language:
- What the gripper does
- What the orange block does
- Whether the block moves (up, down, sideways, or not at all)
- Whether any other object appears near or in front of the block
- Whether the orange block disappears
- If something else has picked up the orange block and block is no longer visible
- Whether the orange block falls down or dropped

Only describe what you can clearly see in the video.
Do not conclude whether the task succeeds or fails.

"""


def _repo_root_from_this_file() -> Path:
    # file: <repo>/src/approach2_exploration/explain_batch_state_label.py
    # parents: explain_batch_state_label.py -> approach2_exploration -> src -> <repo>
    return Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--model", default="nvidia/Cosmos-Reason2-2B")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument(
        "--repo_root",
        default=None,
        help="Optional path to cosmos-supervisor repo root. If omitted, inferred from this script location.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root_from_this_file()

    clips_root = repo_root / "videos" / "approach2_exploration" / "derived_inputs" / "clips_slow_sliding"
    out_dir = repo_root / "outputs" / "approach2_exploration"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scenario -> output filename mapping (exact filenames you requested)
    scenario_to_out = {
        "push_down": "push_down_1_explain_state_label.jsonl",
        "miss_grasp": "miss_grasp_explain_state_label.jsonl",
        "obstruction": "obstruction_explain_state_label.jsonl",
        "success": "success_explain_state_label.jsonl",
    }

    transformers.set_seed(0)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(args.model)

    # vision token limits (defaults from sample) — unchanged
    min_vision_tokens = 256
    max_vision_tokens = 8192
    processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }
    processor.video_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    # Print once (same spirit as your old script)
    print("GEN SETTINGS:", {"do_sample": False, "temperature": 0.1, "top_p": 1.0})

    for scenario, out_name in scenario_to_out.items():
        scenario_dir = clips_root / scenario
        out_path = out_dir / out_name

        if not scenario_dir.exists():
            print(f"[SKIP] Missing clips folder: {scenario_dir}")
            continue

        clips = sorted(scenario_dir.glob("*.mp4"))
        if not clips:
            print(f"[SKIP] No .mp4 found in: {scenario_dir}")
            continue

        print("=" * 80)
        print(f"SCENARIO : {scenario}")
        print(f"CLIPS    : {scenario_dir.relative_to(repo_root)}  (found {len(clips)})")
        print(f"OUTPUT   : {out_path.relative_to(repo_root)}")
        print("=" * 80)

        with out_path.open("w", encoding="utf-8") as f:
            for i, clip in enumerate(clips, 1):
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": str(clip)},
                            {"type": "text", "text": USER},
                        ],
                    },
                ]

                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=args.fps,
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,     # IMPORTANT (unchanged)
                        temperature=0.1,
                        top_p=1.0,
                    )

                trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
                ]
                text = processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                #rec = {"clip": str(clip), "fps": args.fps, "text": text}
                rel_clip = clip.relative_to(repo_root)
                rec = {"clip": str(rel_clip), "fps": args.fps, "text": text}

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Print relative clip name for portability
                rel_clip = clip.relative_to(repo_root) if clip.is_absolute() else clip
                print(f"[{i}/{len(clips)}] {rel_clip}")

        print(f"Saved: {out_path.relative_to(repo_root)}")

    print("\n✅ Done. Outputs are in:", out_dir.relative_to(repo_root))


if __name__ == "__main__":
    main()

