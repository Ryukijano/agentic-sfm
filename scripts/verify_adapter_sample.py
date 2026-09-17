#!/usr/bin/env python
"""Offline adapter sanity check: does a checkpoint generate clean tool JSON?

Settles "bad weights vs bad serving" (Thinker D verification plan step 1/2).
Loads the adapter with HF+PEFT, feeds a real MegaDepth pair + the exact
trainer prompt, samples a few times at rollout settings (temp 1.0, top_p 0.95),
and classifies each completion as clean / degenerate / corrupt.

Usage:
  python scripts/verify_adapter_sample.py <adapter_dir> [n_samples] [seed]
"""

import json
import re
import sys

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, "src")

from agentic_sfm.agent.policy import SYSTEM_PROMPT


def classify(text: str) -> str:
    if not text.strip():
        return "empty"
    dup = len(re.findall(r'"bbox"', text)) > 1
    bare = bool(re.search(r',\s*"?\d+\.\d+', text)) and not re.search(r'"bbox":', text.split('}}')[0] + '}}')
    m = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', text)
    tiny = False
    if m:
        try:
            vals = [float(v) for v in m.group(1).split(",")]
            if len(vals) == 4:
                tiny = min(abs(vals[2] - vals[0]), abs(vals[3] - vals[1])) < 0.25
        except ValueError:
            pass
    if dup or bare:
        return "corrupt"
    if tiny:
        return "tiny_box"
    if '"tool"' in text:
        return "clean"
    return "other"


def main():
    adapter = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    torch.set_grad_enabled(False)
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3.5-2B", torch_dtype=torch.float32, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    img_a = Image.open("/scratch/kcwp264/data/megadepth/megadepth_test_1500/Undistorted_SfM/0015/images/1017541041_394e9f81d4_o.jpg").convert("RGB")
    img_b = Image.open("/scratch/kcwp264/data/megadepth/megadepth_test_1500/Undistorted_SfM/0015/images/310129573_eb686a9dc1_o.jpg").convert("RGB")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": img_a},
            {"type": "image", "image": img_b},
            {"type": "text", "text": 'Match these two images. Call tools to achieve the best matching result, then output {"tool": "done"}.'},
        ]},
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = proc(text=text, images=[img_a, img_b], return_tensors="pt")

    counts = {}
    for i in range(n):
        torch.manual_seed(seed * 100 + i)
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=proc.tokenizer.pad_token_id,
        )
        resp = proc.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cls = classify(resp)
        counts[cls] = counts.get(cls, 0) + 1
        print(f"--- sample {i}: [{cls}]")
        print(resp[:220].replace("\n", "\\n"))

    print("\n=== SUMMARY:", adapter, "->", counts)


if __name__ == "__main__":
    main()
