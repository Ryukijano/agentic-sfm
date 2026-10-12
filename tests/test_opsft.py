"""Tests for OPSFT-style constrained SFT helpers (agentic_sfm.opsft).

Pure CPU; no HF models required. Mirrors the spec in
src/agentic_sfm/opsft.py: canonical-name mask/sign extraction and the
two-stage per-step constraint (stage-1 gradient masking, stage-2
parameter projection back to theta0 on the wrong side of D).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.opsft import (
    canonical_lora_name,
    extract_direction,
    fingerprint_state_dict,
    mask_gradients_,
    project_parameters_,
    save_direction,
    load_direction,
)


# ---------------------------------------------------------------------------
# canonical_lora_name
# ---------------------------------------------------------------------------

class TestCanonicalLoraName:
    def test_strips_default_adapter(self):
        assert (
            canonical_lora_name("x.lora_A.default.weight")
            == "x.lora_A.weight"
        )

    def test_strips_custom_adapter(self):
        assert (
            canonical_lora_name("x.lora_B.my_adapter.weight", adapter_name="my_adapter")
            == "x.lora_B.weight"
        )

    def test_lora_b_and_embedding(self):
        assert canonical_lora_name("m.lora_B.default.weight") == "m.lora_B.weight"
        assert (
            canonical_lora_name("emb.lora_embedding_A.default.weight")
            == "emb.lora_embedding_A.weight"
        )
        assert (
            canonical_lora_name("m.lora_magnitude_vector.default.weight")
            == "m.lora_magnitude_vector.default"
        )

    def test_non_lora_unchanged(self):
        name = "model.layers.0.mlp.weight"
        assert canonical_lora_name(name) == name
        assert canonical_lora_name("a.default.b") == "a.default.b"


# ---------------------------------------------------------------------------
# extract_direction
# ---------------------------------------------------------------------------

class TestExtractDirection:
    def test_identical_dicts_all_false_mask(self):
        sd = {"a.weight": torch.ones(4), "b.weight": torch.zeros(3)}
        out = extract_direction(sd, sd)
        for k, m in out["mask"].items():
            assert m.dtype == torch.bool
            assert not m.any()
        for k, s in out["sign"].items():
            assert s.dtype == torch.int8
            assert (s == 0).all()

    def test_one_changed_element(self):
        base = {"w": torch.zeros(4)}
        trained = {"w": torch.tensor([0.0, 0.0, 2.0, 0.0])}
        out = extract_direction(base, trained)
        assert out["mask"]["w"].tolist() == [False, False, True, False]
        assert out["sign"]["w"].tolist() == [0, 0, 1, 0]

    def test_sign_values(self):
        base = {"w": torch.zeros(4)}
        trained = {"w": torch.tensor([1.0, -3.0, 0.0, 5.0])}
        out = extract_direction(base, trained)
        assert set(out["sign"]["w"].unique().tolist()) <= {-1, 0, 1}
        assert out["sign"]["w"].tolist() == [1, -1, 0, 1]

    def test_key_mismatch_raises(self):
        with pytest.raises(ValueError):
            extract_direction({"a": torch.zeros(2)}, {"b": torch.zeros(2)})

    def test_meta_counts(self):
        base = {"w": torch.zeros(4)}
        trained = {"w": torch.tensor([1.0, -1.0, 0.0, 0.0])}
        meta = extract_direction(base, trained)["meta"]
        assert isinstance(meta, dict)
        # Sane totals: support size should count the changed elements.
        total = sum(int(v) for v in meta.values() if isinstance(v, (int, float)))
        assert total > 0

    def test_topk_keeps_largest_deltas(self):
        base = {"w": torch.zeros(8)}
        trained = {"w": torch.tensor([1.0, -0.1, 0.05, 4.0, -2.0, 0.01, 0.5, -0.02])}
        out = extract_direction(base, trained, selection="topk", topk_frac=0.25)
        # Global top-25% |delta| over 8 elements -> the 2 largest: 4.0 and -2.0.
        assert out["mask"]["w"].tolist() == [False, False, False, True, True, False, False, False]
        # Sign is computed everywhere, not only on the support.
        assert out["sign"]["w"].tolist() == [1, -1, 1, 1, -1, 1, 1, -1]
        assert out["meta"]["selection"] == "topk"
        assert out["meta"]["n_selected_elems"] == 2

    def test_topk_mask_subset_of_changed(self):
        base = {"a": torch.randn(64, generator=torch.Generator().manual_seed(0)),
                "b": torch.randn(32, generator=torch.Generator().manual_seed(1))}
        trained = {k: v + 0.01 * torch.randn(v.shape, generator=torch.Generator().manual_seed(2))
                   for k, v in base.items()}
        exact = extract_direction(base, trained)
        topk = extract_direction(base, trained, selection="topk", topk_frac=0.2)
        for name in exact["mask"]:
            sel = topk["mask"][name]
            # topk support must be a subset of the changed elements.
            assert bool(((~sel) | exact["mask"][name]).all())
            frac = topk["meta"]["frac_selected"]
            assert 0.15 < frac < 0.25  # ~0.2 up to kthvalue granularity

    def test_selection_validation(self):
        sd = {"w": torch.zeros(2)}
        with pytest.raises(ValueError):
            extract_direction(sd, sd, selection="bogus")
        with pytest.raises(ValueError):
            extract_direction(sd, sd, selection="topk", topk_frac=0.0)
        with pytest.raises(ValueError):
            extract_direction(sd, sd, selection="topk", topk_frac=1.5)

    def test_topk_zero_threshold_stays_constrained(self):
        # Many exact-zero deltas can put the kthvalue threshold at 0;
        # mask must then stay a subset of changed elements so no
        # mask=True/sign=0 (unconstrained) entries slip in.
        base = {"w": torch.zeros(10)}
        trained = {"w": torch.tensor([0.0] * 6 + [0.1, 0.2, 0.3, 0.4])}
        out = extract_direction(base, trained, selection="topk", topk_frac=0.8)
        m, s = out["mask"]["w"], out["sign"]["w"]
        assert bool((m == (s != 0)).all())
        assert int(m.sum()) == 4

    def test_topk_tiny_n_no_crash(self):
        base = {"w": torch.zeros(5)}
        trained = {"w": torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0])}
        out = extract_direction(base, trained, selection="topk", topk_frac=0.01)
        assert int(out["mask"]["w"].sum()) >= 1

    def test_base_fingerprint_recorded(self):
        base = {"w": torch.zeros(4)}
        trained = {"w": torch.ones(4)}
        out = extract_direction(base, trained)
        fp = out["meta"]["base_fingerprint"]
        assert isinstance(fp, str) and len(fp) == 64
        # Same content -> same fingerprint; different -> different.
        assert fingerprint_state_dict(base) == fp
        assert fingerprint_state_dict(trained) != fp


class TestCanonicalLoraNameDoRA:
    def test_magnitude_vector_keeps_adapter_strips_weight(self):
        # peft renamed_dora_weights: live 'x.lora_magnitude_vector.default.weight'
        # is saved as 'x.lora_magnitude_vector.default'.
        assert (
            canonical_lora_name("x.lora_magnitude_vector.default.weight")
            == "x.lora_magnitude_vector.default"
        )

    def test_magnitude_vector_custom_adapter(self):
        assert (
            canonical_lora_name(
                "x.lora_magnitude_vector.my_adapter.weight", adapter_name="my_adapter"
            )
            == "x.lora_magnitude_vector.my_adapter"
        )


# ---------------------------------------------------------------------------
# mask_gradients_  (stage 1)
# ---------------------------------------------------------------------------

def _params():
    """named params with PEFT-style names; canonical key is 'a.lora_A.weight' etc."""
    return {
        "a.lora_A.default.weight": nn.Parameter(torch.zeros(4)),
        "b.lora_B.default.weight": nn.Parameter(torch.zeros(4)),
        "c.weight": nn.Parameter(torch.zeros(4)),
    }


def _direction():
    mask = {
        "a.lora_A.weight": torch.tensor([True, True, True, True]),
        "b.lora_B.weight": torch.tensor([True, True, False, False]),
    }
    sign = {
        "a.lora_A.weight": torch.tensor([1, -1, 1, -1], dtype=torch.int8),
        "b.lora_B.weight": torch.tensor([1, 1, 0, 0], dtype=torch.int8),
    }
    return mask, sign


class TestMaskGradients:
    def test_opposing_and_offsupport_zeroed(self):
        params = _params()
        mask, sign = _direction()
        # a: grad*D>0 at idx 0 (1*1), idx 2 (1*1) -> zeroed; idx 1 (-1*-1>0)
        # also opposing? (-1)*(-1)=1>0 opposing. Craft aligned at idx1: D=-1,
        # want grad*D<0 -> grad>0. Let's set explicit grads.
        params["a.lora_A.default.weight"].grad = torch.tensor([1.0, 1.0, -1.0, -1.0])
        # a idx: D=[1,-1,1,-1]; grad*D = [1, -1, -1, 1] -> idx0,3 zeroed; idx1,2 kept
        params["b.lora_B.default.weight"].grad = torch.tensor([1.0, -1.0, 9.0, 9.0])
        # b: mask=[T,T,F,F], D=[1,1,0,0]; idx0 grad*D=1>0 -> zeroed;
        # idx1 grad*D=-1 aligned -> kept; idx2,3 off-support -> zeroed
        params["c.weight"].grad = torch.ones(4)  # not in mask: untouched

        stats = mask_gradients_(params.items(), mask, sign)

        ga = params["a.lora_A.default.weight"].grad
        assert ga[0].item() == 0.0 and ga[3].item() == 0.0
        assert ga[1].item() == 1.0 and ga[2].item() == -1.0

        gb = params["b.lora_B.default.weight"].grad
        assert gb[0].item() == 0.0
        assert gb[1].item() == -1.0
        assert gb[2].item() == 0.0 and gb[3].item() == 0.0

        assert params["c.weight"].grad.eq(torch.ones(4)).all()

        assert isinstance(stats, dict)
        total = sum(int(v) for v in stats.values() if isinstance(v, (int, float)))
        assert total > 0

    def test_zero_grad_not_counted(self):
        params = _params()
        mask, sign = _direction()
        params["a.lora_A.default.weight"].grad = torch.zeros(4)
        mask_gradients_(params.items(), mask, sign)
        assert params["a.lora_A.default.weight"].grad.eq(torch.zeros(4)).all()


# ---------------------------------------------------------------------------
# project_parameters_  (stage 2)
# ---------------------------------------------------------------------------

class TestProjectParameters:
    def test_off_support_restored(self):
        params = _params()
        mask, sign = _direction()
        theta0 = {n: p.detach().clone() for n, p in params.items()}
        theta0_canon = {canonical_lora_name(n): v for n, v in theta0.items()}

        with torch.no_grad():
            params["b.lora_B.default.weight"][2:] += 5.0  # off-support drift
            project_parameters_(params.items(), theta0_canon, mask, sign)
        assert params["b.lora_B.default.weight"].eq(torch.zeros(4)).all()

    def test_wrong_side_restored_right_side_kept(self):
        params = _params()
        mask, sign = _direction()
        theta0 = {canonical_lora_name(n): p.detach().clone() for n, p in params.items()}

        with torch.no_grad():
            # D = [1,-1,1,-1]; displacement *D<0 -> wrong side.
            params["a.lora_A.default.weight"] += torch.tensor([-1.0, 1.0, -0.5, 0.5])
            project_parameters_(params.items(), theta0, mask, sign)
        assert params["a.lora_A.default.weight"].eq(torch.zeros(4)).all()

        # displacement along D is kept
        with torch.no_grad():
            params["a.lora_A.default.weight"] += torch.tensor([0.3, -0.3, 0.2, -0.2])
            project_parameters_(params.items(), theta0, mask, sign)
        got = params["a.lora_A.default.weight"]
        assert torch.allclose(got, torch.tensor([0.3, -0.3, 0.2, -0.2]))

    def test_param_missing_from_mask_untouched(self):
        params = _params()
        mask, sign = _direction()
        theta0 = {canonical_lora_name(n): p.detach().clone() for n, p in params.items()}
        with torch.no_grad():
            params["c.weight"] += 7.0
            project_parameters_(params.items(), theta0, mask, sign)
        assert params["c.weight"].eq(7.0 * torch.ones(4)).all()


# ---------------------------------------------------------------------------
# End-to-end: AdamW + per-step mask + project keeps constraint invariant
# ---------------------------------------------------------------------------

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(4))
        self.lora_B = nn.Parameter(torch.zeros(4))


class TinyNamed(nn.Module):
    """Wrap params under PEFT-style names: 'lora_A.default.weight'."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(4))
        self.b = nn.Parameter(torch.zeros(4))

    def peft_named_parameters(self):
        return {
            "a.lora_A.default.weight": self.a,
            "b.lora_B.default.weight": self.b,
        }


def test_end_to_end_adamw_respects_direction():
    torch.manual_seed(0)
    model = TinyNamed()
    named = model.peft_named_parameters()

    # Direction: a increases, b decreases; full support on both.
    mask = {
        "a.lora_A.weight": torch.tensor([True, True, False, False]),
        "b.lora_B.weight": torch.tensor([True, False, False, False]),
    }
    sign = {
        "a.lora_A.weight": torch.tensor([1, -1, 0, 0], dtype=torch.int8),
        "b.lora_B.weight": torch.tensor([-1, 0, 0, 0], dtype=torch.int8),
    }
    theta0 = {n: p.detach().clone() for n, p in named.items()}
    theta0 = {canonical_lora_name(k): v for k, v in theta0.items()}

    target = torch.tensor([1.0, 1.0, 1.0, 1.0])
    opt = torch.optim.AdamW(named.values(), lr=0.05, weight_decay=0.01)

    for _ in range(30):
        opt.zero_grad()
        loss = ((model.a - target) ** 2).sum() + ((model.b + target) ** 2).sum()
        loss.backward()
        mask_gradients_(named.items(), mask, sign)
        opt.step()
        with torch.no_grad():
            project_parameters_(named.items(), theta0, mask, sign)

        # Off-support elements are exactly theta0 (weight decay would pull
        # them to 0... theta0 IS 0 here, so use a non-zero theta0 below).
        disp_a = model.a.detach() - theta0["a.lora_A.weight"]
        da = sign["a.lora_A.weight"].float()
        supported = mask["a.lora_A.weight"] & (da != 0)
        assert ((disp_a * da)[supported] >= -1e-7).all()
        assert (disp_a[~mask["a.lora_A.weight"]] == 0).all()

        disp_b = model.b.detach() - theta0["b.lora_B.weight"]
        db = sign["b.lora_B.weight"].float()
        supported_b = mask["b.lora_B.weight"] & (db != 0)
        assert ((disp_b * db)[supported_b] >= -1e-7).all()
        assert (disp_b[~mask["b.lora_B.weight"]] == 0).all()


def test_end_to_end_weight_decay_cannot_move_offsupport():
    """Non-zero theta0: weight decay would drag off-support params toward 0;
    stage-2 projection must pin them to theta0 every step."""
    model = TinyNamed()
    named = model.peft_named_parameters()
    with torch.no_grad():
        model.a.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        model.b.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))

    mask = {"a.lora_A.weight": torch.tensor([True, False, False, False])}
    sign = {"a.lora_A.weight": torch.tensor([1, 0, 0, 0], dtype=torch.int8)}
    theta0 = {canonical_lora_name(n): p.detach().clone() for n, p in named.items()}

    opt = torch.optim.AdamW(named.values(), lr=0.05, weight_decay=0.01)
    for _ in range(10):
        opt.zero_grad()
        ((model.a - torch.tensor([1.0, 0.0, 0.0, 0.0])) ** 2).sum().backward()
        mask_gradients_(named.items(), mask, sign)
        opt.step()
        with torch.no_grad():
            project_parameters_(named.items(), theta0, mask, sign)

        # Off-support a[1:] and all of b must be exactly theta0 (0.5).
        assert model.a[1:].eq(torch.full((3,), 0.5)).all()
        assert model.b.eq(torch.full((4,), 0.5)).all()


# ---------------------------------------------------------------------------
# save/load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    base = {"w": torch.zeros(3)}
    trained = {"w": torch.tensor([1.0, 0.0, -2.0])}
    artifact = extract_direction(base, trained)
    path = str(tmp_path / "direction.pt")
    save_direction(artifact, path)
    loaded = load_direction(path)
    for k in artifact["mask"]:
        assert loaded["mask"][k].eq(artifact["mask"][k]).all()
        assert loaded["sign"][k].eq(artifact["sign"][k]).all()
    assert loaded["meta"] == artifact["meta"]
