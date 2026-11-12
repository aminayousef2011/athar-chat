"""Protocol v3 freezes (gates G2+G3) on synthetic data: S3 cross-fitting, the size-matched control,
write-once files, tamper detection and the command line. Runs offline, without the private texts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from athar.schema import GoldItem
from athar.v3 import __main__ as cli
from athar.v3 import controls as C
from athar.v3 import data as D
from athar.v3 import systems as S

REF = datetime.fromisoformat("2026-11-01T08:00:00+02:00")
TEXTS = {"create_reminder": "فكرني بكرة الساعة 9 الصبح اراجع التقرير رقم {k}",
         "create_alarm": "صحيني بكرة الساعة 7 الصبح يوم {k}",
         "unsupported": "امسح التذكير رقم {k} من فضلك"}


def _gold(prefix: str, families: list[str], n: int) -> list[GoldItem]:
    out = []
    for k in range(n):
        intent = list(TEXTS)[k % 3]
        extra: dict[str, Any] = ({"route": "ABSTAIN", "unsupported_reason": "deletion"} if intent == "unsupported"
                                 else {"route": "CLARIFY", "clarify_targets": ["date"]})
        out.append(GoldItem.model_validate({"id": f"{prefix}{k:03d}", "family_id": families[k % len(families)],
                                            "text": TEXTS[intent].format(k=k), "ref_datetime": REF,
                                            "intent": intent, "source": "contributor", **extra}))
    return out


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Two v3 blocks, a drafted pool and a natural set whose cards overlap block1 only."""
    v3 = [D.V3Item(id=f"c00{k}-a-x{k}", text=TEXTS[list(TEXTS)[k % 3]].format(k=k), ref_datetime=REF,
                   timezone="Africa/Cairo", family_id=f"fam-c00{k}", block="block1" if k < 3 else "block2",
                   writer=f"X{k}") for k in range(6)]
    cards = {"block1": {"fam-c000", "fam-c001", "fam-c002"}, "block2": {"fam-c003", "fam-c004", "fam-c005"}}
    natural = _gold("n", ["fam-c000", "fam-c009", "fam-c010"], 12)
    drafted = _gold("d", ["fam-t1", "fam-t2"], 9)
    params = S.s3_params()   # the frozen S3-gcv settings, read before REPO is redirected
    monkeypatch.setattr(S, "s3_params", lambda: params)
    monkeypatch.setattr(D, "load", lambda *a, **k: v3)
    monkeypatch.setattr(D, "block_cards", lambda *a, **k: cards)
    monkeypatch.setattr(S, "natural_items", lambda: natural)
    monkeypatch.setattr(S, "drafted_pool", lambda: drafted)
    monkeypatch.setattr(S, "REPO", tmp_path)
    monkeypatch.setattr(S, "PRED_DIR", tmp_path / "results" / "v3" / "predictions")
    for rel in C.SOURCES:   # verify looks for each frozen source under REPO (hashes come from the real tree)
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((D.REPO / rel).read_bytes())
    (tmp_path / "models_v3").mkdir()
    return {"out": tmp_path / "models_v3", "v3": v3, "natural": natural, "drafted": drafted, "root": tmp_path}


def test_freeze_s3_cross_fits_by_block_and_verifies(world: dict[str, Any]) -> None:
    fz = S.freeze_s3(out=world["out"])
    models = fz["models"]
    assert set(models) == {"S3-drafted", "S3-human-block1", "S3-human-block2"}
    # block1's cards include fam-c000, so its natural messages on that card are left out
    on_c000 = sum(it.family_id == "fam-c000" for it in world["natural"])
    assert models["S3-human-block1"]["natural_in_pool"] == len(world["natural"]) - on_c000
    assert models["S3-human-block2"]["natural_in_pool"] == len(world["natural"])
    preds = json.loads((world["root"] / fz["predictions"]["S3-human"]["file"]).read_text(encoding="utf-8"))
    assert set(preds) == {it.id for it in world["v3"]}
    assert S.verify(world["out"]) == []
    with pytest.raises(S.FreezeError):
        S.freeze_s3(out=world["out"])


def test_verify_reports_every_kind_of_change(world: dict[str, Any]) -> None:
    fz = S.freeze_s3(out=world["out"])
    (world["root"] / fz["predictions"]["S3-drafted"]["file"]).write_text("{}", encoding="utf-8")
    (world["out"] / "S3-human-block1.joblib").write_bytes(b"tampered")
    freeze = world["out"] / "freeze_s3.json"
    data = json.loads(freeze.read_text(encoding="utf-8"))
    data["params"]["C"] = 999
    freeze.write_text(json.dumps(data), encoding="utf-8")
    problems = " | ".join(S.verify(world["out"]))
    assert "was edited" in problems
    assert "model S3-human-block1 missing or changed" in problems
    assert "predictions S3-drafted missing or changed" in problems


def test_verify_without_any_freeze(tmp_path: Path) -> None:
    assert S.verify(tmp_path) == [f"no v3 freeze in {tmp_path}"]


def test_contributors_only_control_has_no_drafted_item(world: dict[str, Any]) -> None:
    fz = C.freeze(out=world["out"])
    on_c000 = sum(it.family_id == "fam-c000" for it in world["natural"])
    assert fz["models"]["S3-contrib-only-block1"]["n_train"] == len(world["natural"]) - on_c000
    assert fz["models"]["S3-contrib-only-block2"]["n_train"] == len(world["natural"])
    assert fz["amendment"] == "A4" and "amendments_sha256" in fz
    pool = C.contributors_pool("block2", D.block_cards(), world["natural"])
    assert all(not it.id.startswith("d") for it in pool)
    with pytest.raises(S.FreezeError):
        C.freeze(out=world["out"])


def test_controls_command_line(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert C.main([]) == 2
    monkeypatch.setattr(C, "freeze", lambda: {"freeze_sha256": "abc", "models": {"m": {"n_train": 3}}})
    assert C.main(["freeze"]) == 0
    assert "abc" in capsys.readouterr().out


def test_v3_command_line_dispatch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(D, "prepare", lambda: {"messages": 6})
    monkeypatch.setattr(S, "freeze_s3", lambda: {"freeze_sha256": "s3hash", "v3_set_sha256": "x" * 20})
    monkeypatch.setattr(S, "freeze_s6", lambda device, seeds: {"freeze_sha256": f"s6-{device}-{seeds}"})
    assert cli.main(["prepare"]) == 0
    assert cli.main(["freeze-s3"]) == 0
    assert cli.main(["freeze-s6", "--device", "cuda", "--seeds", "0", "1"]) == 0
    out = capsys.readouterr().out
    assert "'messages': 6" in out and "s3hash" in out and "s6-cuda-[0, 1]" in out
    monkeypatch.setattr(S, "verify", lambda: [])
    assert cli.main(["verify"]) == 0
    monkeypatch.setattr(S, "verify", lambda: ["changed"])
    assert cli.main(["verify"]) == 1


def test_amendments_file_is_append_only(tmp_path: Path) -> None:
    from athar.v3 import amend

    f = tmp_path / "amendments.md"
    f.write_text("| A1 | first |\n", encoding="utf-8")
    recorded = amend.snapshot(f)
    f.write_text("| A1 | first |\n| A2 | appended |\n", encoding="utf-8")
    assert amend.check(recorded, f) is None
    f.write_text("| A1 | FIRST |\n| A2 | appended |\n", encoding="utf-8")
    assert "edited" in (amend.check(recorded, f) or "")
    f.write_text("| A1 |\n", encoding="utf-8")
    assert "shorter" in (amend.check(recorded, f) or "")


def test_amendments_command_on_the_committed_freezes(capsys: pytest.CaptureFixture[str]) -> None:
    from athar.v3 import amend

    assert amend.check_all() == []
    assert amend.main() == 0
    assert "append-only OK" in capsys.readouterr().out
