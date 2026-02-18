#!/usr/bin/env python3
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import torch
import transformers


# -----------------------------
# Inputs (defaults)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../cosmos-supervisor
OUT_DIR   = REPO_ROOT / "outputs" / "approach2_exploration"

DEFAULT_FILES = {
    "success":     OUT_DIR / "success_explain_state_label.jsonl",
    "obstruction": OUT_DIR / "obstruction_explain_state_label.jsonl",
    "push_down":   OUT_DIR / "push_down_1_explain_state_label.jsonl",
}


# -----------------------------
# Heuristic state extraction (FIXED)
# -----------------------------
@dataclass
class ClipState:
    t_sec: int
    clip: str
    toy_visible: bool
    lift: bool
    drop: bool
    block_motion: str  # "up" | "down" | "sideways" | "none"
    confidence: float
    evidence: List[str]


_TIME_RE = re.compile(r"_(\d+)s\.mp4$", re.IGNORECASE)

def parse_t_sec(clip_path: str) -> int:
    m = _TIME_RE.search(clip_path)
    return int(m.group(1)) if m else -1

def norm_text(s: str) -> str:
    return (s or "").strip().lower()

def has_any(txt: str, kws: List[str]) -> bool:
    return any(k in txt for k in kws)

def extract_state_from_text(text: str, clip_path: str) -> ClipState:
    """
    Deterministic keyword logic tuned for your semantics:
    - success      = block stays in gripper / in air   (lift=True, drop=False, motion=up)
    - miss_grasp   = block falls/drops to table        (drop=True, motion=down)
    - obstruction  = toy visible                       (toy_visible=True)
    """
    t = parse_t_sec(clip_path)
    txt = norm_text(text)
    evidence: List[str] = []

    def has_any(s: str, kws: List[str]) -> bool:
        return any(k in s for k in kws)

    # -------------------------
    # 1) Toy / obstruction
    # -------------------------
    toy_kws = [
        "stuffed", "plush", "plushie", "toy", "teddy", "bear", "monkey", "mouse",
        "music", "foot", "doll"
    ]
    toy_visible = has_any(txt, toy_kws)
    if toy_visible:
        evidence.append("toy_visible: toy keywords matched")

    # -------------------------
    # 2) Controlled set-down (NOT a failure drop)
    # -------------------------
    controlled_setdown_kws = [
        "placed on", "placed onto", "set down", "sets down",
        "lowered", "lowering", "gently", "carefully",
        "released onto", "released on",
        "allowing the block to rest",
        "comes into contact with the table",
        "settles on the table", "settles onto the table",
        "rests on the table",
        "returned to its original position",
        "placed back on the surface", "placed back on the table"
    ]
    controlled_setdown = has_any(txt, controlled_setdown_kws)
    if controlled_setdown:
        evidence.append("controlled_setdown: placement/rest keywords matched")

    # -------------------------
    # 3) Lift detection (strong vs weak)
    # -------------------------
    lift_strong_kws = [
        "in the air", "suspended",
        "off the table", "off the surface",
        "lifted off", "picked up", "raised off",
        "held above", "above the table"
    ]
    lift_weak_kws = ["lift", "lifted", "lifting", "raised", "pick up", "picked up"]

    # Lift = strong evidence OR weak evidence but NOT obviously a set-down/resting clip
    lift = has_any(txt, lift_strong_kws) or (has_any(txt, lift_weak_kws) and not controlled_setdown)
    if lift:
        evidence.append("lift: lift cues matched (strong or weak w/o setdown)")

    # -------------------------
    # 4) Drop detection (UNCONTROLLED only)
    # -------------------------
    # If you see only "released" without fall/slip/lost grip wording -> NOT a drop.
    drop_uncontrolled_kws = [
        "falls", "fell", "falling",
        "dropped", "drops", "dropping",
        "slips", "slipped",
        "drops out", "falls out",
        "loses grip", "lost grip",
        "accident", "accidentally", "uncontrolled"
    ]
    mentions_release = ("released" in txt) or ("let go" in txt)

    drop = has_any(txt, drop_uncontrolled_kws)

    # If it mentions release AND also mentions fall/slip/drop -> drop=True
    if mentions_release and has_any(txt, ["falls", "fell", "falling", "slip", "slipped", "dropped", "drops"]):
        drop = True
        evidence.append("drop: release + fall/slip/drop wording")

    # If it looks like controlled setdown and we don't have strong uncontrolled cues -> drop=False
    if controlled_setdown and not has_any(txt, ["slip", "slipped", "lost grip", "uncontrolled", "accident", "accidentally"]):
        if drop:
            evidence.append("drop overridden: controlled setdown detected")
        drop = False

    if drop:
        evidence.append("drop: uncontrolled fall/drop cues matched")

    # -------------------------
    # 5) Motion classification (BLOCK motion, not gripper motion)
    # -------------------------
    # IMPORTANT: do NOT use generic "downward"/"descends" (often describes gripper).
    up_kws = [
        "moves upward", "moved upward", "upward",
        "lifted off", "off the table", "off the surface",
        "in the air", "picked up", "raised off", "held above"
    ]
    down_kws = [
        # only block-down events (fall/drop/press)
        "falls to the table", "falls onto the table",
        "fell onto the table", "dropped onto the", "dropped on the",
        "falls", "fell", "falling", "dropped", "drops",
        "pushed down", "pressed down", "into the surface"
    ]
    side_kws = [
        "sideways", "slides", "sliding",
        "to the left", "to the right",
        "shift", "shifted",
        "moved slightly to the", "moves to the"
    ]

    block_motion = "none"

    # Drop overrides to down (coarse and reliable)
    if drop:
        block_motion = "down"
        evidence.append("block_motion=down: drop=True")
    elif has_any(txt, down_kws):
        block_motion = "down"
        evidence.append("block_motion=down: explicit block-down cues")
    elif lift or has_any(txt, up_kws):
        block_motion = "up"
        evidence.append("block_motion=up: lift/up cues")
    elif has_any(txt, side_kws):
        block_motion = "sideways"
        evidence.append("block_motion=sideways: lateral cues")
    else:
        evidence.append("block_motion=none: no motion cues")

    # If lift=True and drop=False, force motion up (prevents “downward” gripper wording from poisoning)
    if lift and not drop:
        if block_motion != "up":
            evidence.append("block_motion forced to up (lift=True, drop=False)")
        block_motion = "up"

    # -------------------------
    # 6) Stationary phrasing (ONLY if no lift/drop/motion)
    # -------------------------
    stationary_kws = [
        "block remains stationary",
        "orange block remains stationary",
        "block stays in place",
        "block does not move",
        "no movement of the orange block"
    ]
    if has_any(txt, stationary_kws) and (not lift) and (not drop) and (block_motion == "none"):
        evidence.append("stationary confirms none")

    # -------------------------
    # 7) Confidence heuristic
    # -------------------------
    conf = 0.55
    if toy_visible:
        conf += 0.15
    if lift:
        conf += 0.15
    if drop:
        conf += 0.15
    if block_motion != "none":
        conf += 0.08

    contradictions = 0
    if lift and controlled_setdown:
        contradictions += 1
        evidence.append("contradiction: lift + controlled_setdown")
    if has_any(txt, stationary_kws) and block_motion != "none":
        contradictions += 1
        evidence.append("mixed: stationary + motion cues")

    conf -= 0.10 * contradictions
    conf = max(0.2, min(0.95, conf))

    return ClipState(
        t_sec=t,
        clip=clip_path,
        toy_visible=toy_visible,
        lift=lift,
        drop=drop,
        block_motion=block_motion,
        confidence=round(conf, 2),
        evidence=evidence,
    )

def pick_one_state(states: List[ClipState], scenario: str) -> ClipState:
    # assumes states already sorted by time
    if not states:
        raise ValueError("empty timeline")

    if scenario == "success":
        return states[0]          # first
    if scenario in ("obstruction", "miss_grasp"):
        return states[-1]         # last
    return states[-1]


def load_label_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_timeline(rows: List[Dict[str, Any]]) -> List[ClipState]:
    states = []
    for r in rows:
        clip = r.get("clip", "")
        txt = r.get("text", "")
        states.append(extract_state_from_text(txt, clip))
    states.sort(key=lambda x: (x.t_sec if x.t_sec >= 0 else 10**9, x.clip))
    return states


def timeline_to_text(states: List[ClipState]) -> str:
    lines = []
    for s in states:
        t_str = f"{s.t_sec}s" if s.t_sec >= 0 else "unknown_t"
        lines.append(
            f"t={t_str}: "
            f"toy_visible={str(s.toy_visible).lower()}, "
            f"lift={str(s.lift).lower()}, "
            f"drop={str(s.drop).lower()}, "
            f"block_motion={s.block_motion}, "
            f"conf={s.confidence}"
        )
    return "\n".join(lines)


# -----------------------------
# Level-2 with Cosmos 
# -----------------------------


LEVEL2_SYSTEM = """
You are a robot monitoring supervisor.

You will receive a time-ordered timeline like:
t=0s: toy_visible=..., lift=..., drop=..., block_motion=..., conf=...

Output ONLY valid JSON:
{"action":"PROCEED|REPLAN|ABORT","why":"...", "confidence":0-1}

CRITICAL FORMAT:
- "why" MUST cite evidence using timestamps exactly like:
  "toy_visible=true at t=12"
  "lift=true at t=0,2,4"
  "drop=true at t=4,6,10"
- Do NOT restate rules.
- Do NOT explain decision logic.
- Do NOT mention precedence.

DECISION RULES (follow exactly):

1) ABORT
  If toy_visible=true at any timestamp, you MUST output {"action":"ABORT", ...}.
In that case, outputting REPLAN or PROCEED is invalid.

Example:
Input:
t=1s: toy_visible=true, lift=false, drop=false, block_motion=none, conf=0.6
Output:
{"action":"ABORT","why":"toy_visible=true at t=1","confidence":0.6}


2) PROCEED
IF lift=true at 2 or more timestamps
AND toy_visible is never true:
  action=PROCEED
  (Ignore drop signals in this case; camera depth may be unreliable.)

3) REPLAN
IF drop=true at ANY timestamp
AND lift=false at the FINAL timestamp:
  action=REPLAN

4) Otherwise:
action=REPLAN
"""

LEVEL2_USER_TEMPLATE = """

Timeline:
{timeline}

Return JSON only. In "why", mention the exact timestamps that triggered your choice.

"""

def run_level2_cosmos(model, processor, timeline_text: str, max_new_tokens: int = 256) -> Tuple[Dict[str, Any], str]:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": LEVEL2_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": LEVEL2_USER_TEMPLATE.format(timeline=timeline_text)}]},
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    try:
        raw2 = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw2)
    except Exception:
        parsed = {"action": "REPLAN", "why": "Could not parse model JSON output", "confidence": 0.0}

    return parsed, raw


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = "nvidia/Cosmos-Reason2-2B"
    #model_name = "nvidia/Cosmos-Reason2-8B"
    transformers.set_seed(0)

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

    results_summary = {}

    for scenario, infile in DEFAULT_FILES.items():
        if not infile.exists():
            print(f"[SKIP] Missing file: {infile}")
            continue

        print(f"\n=== Scenario: {scenario} ===")
        rows = load_label_jsonl(infile)
        states = build_timeline(rows)
        one = pick_one_state(states, scenario)
        one_text = timeline_to_text([one])

        print("\nSelected one row for Level-2:")
        print(one_text)

        level2_json, raw = run_level2_cosmos(model, processor, one_text)
        #timeline_text = timeline_to_text(states)

        #timeline_out = OUT_DIR / f"{scenario}_timeline.json"
        #timeline_out.write_text(json.dumps([asdict(s) for s in states], ensure_ascii=False, indent=2), encoding="utf-8")
        #print(f"Saved timeline: {timeline_out}")

        #level2_json, raw = run_level2_cosmos(model, processor, timeline_text)

        out_json = OUT_DIR / f"{scenario}_level2.json"
        out_raw = OUT_DIR / f"{scenario}_level2_raw.txt"
        out_json.write_text(json.dumps(level2_json, ensure_ascii=False, indent=2), encoding="utf-8")
        out_raw.write_text(raw, encoding="utf-8")

        print("Level-2 timeline input:")
        print(one_text)
        print("\nLevel-2 parsed output:")
        print(json.dumps(level2_json, indent=2))

        results_summary[scenario] = level2_json

    summary_out = OUT_DIR / "all_scenarios_level2_summary.json"
    summary_out.write_text(json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_out}")


if __name__ == "__main__":
    main()

