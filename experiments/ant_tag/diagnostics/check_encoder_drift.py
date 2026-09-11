"""Did the ST encoder inside a saved PPO agent move away from its pretrained weights?

Reads the policy tensors straight out of SB3 zips (agent, best_model or a
100k-step checkpoint) -- no env, no GPU, no unpickling of the extractor
class -- and compares ``features_extractor.encoder.*`` with the pretraining
checkpoint's ``set_transformer.*``. Answers, per agent:

  * FROZEN   every encoder tensor is bit-identical to the checkpoint
  * MOVED    max |delta|, relative Frobenius drift ||delta|| / ||ref||, and the
             number of tensors that changed

so the --st_frozen arm can be shown to have stayed frozen through real PPO
updates and the --st_encoder_lr_scale arm to have actually trained. With
--reference_agent the same comparison runs agent-vs-agent (encoder AND policy
head separately), which is the only reference an end-to-end run has: its
first checkpoint against its last.

    python3 diagnostics/check_encoder_drift.py \
        --pretrained_ckpt experiments/st_pretrain_<v>_plain/<stamp>/checkpoints/checkpoint_best.pt \
        runs/ant_tag_st_<v>/<run>/models/st_agent.zip runs/.../checkpoints/ant_tag_st_100000_steps.zip
    python3 diagnostics/check_encoder_drift.py --reference_agent <first ckpt zip> <last ckpt zip>

Exit status 1 if --expect frozen|moved is given and any agent disagrees.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import torch

# diagnostics/ is one level below the ant_tag scripts: parents[3] is the
# set_transformer repo root (CLAUDE.md). Needed only so the pretraining
# checkpoint's pickled TrainingConfig resolves; nothing else is imported.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ENCODER_PREFIX = "features_extractor.encoder."
HEAD_PREFIXES = ("mlp_extractor.", "action_net.", "value_net.")


def load_policy_tensors(agent_zip: str) -> dict[str, torch.Tensor]:
    with zipfile.ZipFile(agent_zip) as zf:
        raw = zf.read("policy.pth")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def load_pretrained_encoder(ckpt: str) -> dict[str, torch.Tensor]:
    # 3_train_st.py checkpoints pickle their TrainingConfig alongside the
    # weights, so the weights-only loader refuses them; these are local files.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    enc = {k[len("set_transformer."):]: v for k, v in state.items()
           if k.startswith("set_transformer.")}
    if not enc:
        sys.exit(f"{ckpt}: no set_transformer.* tensors (keys: {list(state)[:5]}...)")
    return enc


def subset(policy: dict[str, torch.Tensor], prefixes) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in policy.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):] if len(prefixes) == 1 else k] = v
    return out


def compare(ref: dict[str, torch.Tensor], live: dict[str, torch.Tensor]) -> dict:
    common = sorted(set(ref) & set(live))
    if not common:
        raise SystemExit(f"no common tensors (ref {list(ref)[:3]}..., live {list(live)[:3]}...)")
    max_abs, changed, num, den = 0.0, 0, 0.0, 0.0
    for k in common:
        a, b = ref[k].float(), live[k].float()
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch on {k}: {tuple(a.shape)} vs {tuple(b.shape)} "
                             "(different ST geometry?)")
        d = (a - b).abs()
        m = float(d.max()) if d.numel() else 0.0
        max_abs = max(max_abs, m)
        changed += int(m > 0)
        num += float((a - b).pow(2).sum())
        den += float(a.pow(2).sum())
    return dict(n_tensors=len(common), n_changed=changed, max_abs=max_abs,
                rel_frob=(num ** 0.5) / max(den ** 0.5, 1e-12),
                missing_ref=sorted(set(live) - set(ref)), missing_live=sorted(set(ref) - set(live)))


def fmt(r: dict) -> str:
    verdict = "FROZEN" if r["max_abs"] == 0.0 else "MOVED "
    return (f"{verdict} max|d|={r['max_abs']:.3e} rel_frob={100 * r['rel_frob']:.3f}% "
            f"changed {r['n_changed']}/{r['n_tensors']} tensors")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("agents", nargs="+", help="SB3 zips: st_agent.zip, best_model.zip, ant_tag_st_*_steps.zip")
    ap.add_argument("--pretrained_ckpt", help="3_train_st.py checkpoint_best.pt the encoder was loaded from")
    ap.add_argument("--reference_agent", help="another SB3 zip; compare encoder AND head against it")
    ap.add_argument("--expect", choices=["frozen", "moved"],
                    help="exit 1 if any agent's encoder disagrees with this")
    args = ap.parse_args()
    if not args.pretrained_ckpt and not args.reference_agent:
        ap.error("give --pretrained_ckpt and/or --reference_agent")

    ok = True
    ref_enc = load_pretrained_encoder(args.pretrained_ckpt) if args.pretrained_ckpt else None
    ref_pol = load_policy_tensors(args.reference_agent) if args.reference_agent else None
    for agent in args.agents:
        pol = load_policy_tensors(agent)
        enc = subset(pol, (ENCODER_PREFIX,))
        if not enc:
            print(f"{agent}: no {ENCODER_PREFIX}* tensors -- not an ST agent?")
            ok = False
            continue
        if ref_enc is not None:
            r = compare(ref_enc, enc)
            print(f"{agent}\n  encoder vs pretrained ckpt: {fmt(r)}")
            if r["missing_ref"] or r["missing_live"]:
                print(f"  (unmatched: ckpt-only {r['missing_live']}, agent-only {r['missing_ref']})")
            if args.expect == "frozen" and r["max_abs"] != 0.0:
                ok = False
            if args.expect == "moved" and r["max_abs"] == 0.0:
                ok = False
        if ref_pol is not None:
            r_enc = compare(subset(ref_pol, (ENCODER_PREFIX,)), enc)
            r_head = compare(subset(ref_pol, HEAD_PREFIXES), subset(pol, HEAD_PREFIXES))
            print(f"{agent}\n  encoder vs {args.reference_agent}: {fmt(r_enc)}"
                  f"\n  policy head vs same:             {fmt(r_head)}")
            if args.expect == "frozen" and r_enc["max_abs"] != 0.0:
                ok = False
            if args.expect == "moved" and r_enc["max_abs"] == 0.0:
                ok = False
    if args.expect and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
