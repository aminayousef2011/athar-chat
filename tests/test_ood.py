"""Egyptian out-of-distribution check (athar.eval.ood), offline parts."""

import pytest

from athar.eval import ood


def test_mapping_follows_the_massive_rule():
    assert ood.map_intent("alarm_set") == "create_alarm"
    assert ood.map_intent("calendar_set") is None
    assert ood.map_intent("alarm_remove") == ood.map_intent("weather_query") == "unsupported"


def test_score_counts_recall_per_label_and_false_executions():
    s = ood.score(["unsupported", "unsupported", "create_alarm"], ["unsupported", "create_reminder", "create_alarm"])
    assert (s["recall_unsupported"], s["recall_create_alarm"], s["false_execution_intents"]) == (0.5, 1.0, 1)


def test_load_refuses_a_file_that_is_not_the_pinned_one(tmp_path):
    f = tmp_path / "egy_test.jsonl"
    f.write_text('{"id": "x", "utt": "اعمل منبه", "intent": "alarm_set", "source": "human"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        ood.load(f, download=False)
    with pytest.raises(FileNotFoundError):
        ood.load(tmp_path / "missing.jsonl", download=False)


def test_cli_on_a_local_copy(split_dir, tmp_path, monkeypatch, capsys):
    import hashlib
    import json

    from athar.intent import train as T

    rows = [{"id": "e1", "utt": "اعمل منبه يصحيني سبعة الصبح", "intent": "alarm_set", "source": "human"},
            {"id": "e2", "utt": "وطى الصوت من فضلك", "intent": "audio_volume_down", "source": "human"},
            {"id": "e3", "utt": "سجل انه عندي اجتماع بكره", "intent": "calendar_set", "source": "human"},
            {"id": "e4", "utt": "شغل اغنية", "intent": "play_music", "source": "synthetic"}]
    f = tmp_path / "egy_test.jsonl"
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(ood, "SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    assert [r["gold"] for r in ood.load(f, download=False)] == ["create_alarm", "unsupported"]
    T.fit_and_freeze(split_dir / "manifest.json", split_dir / "train.yaml", split_dir / "models", log=lambda _: None)
    out = tmp_path / "ood.json"
    assert ood.main(["--models", str(split_dir / "models"), "--data", str(f), "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert set(result["systems"]) == {"S1", "S2", "S3"} and result["data"]["revision"] == ood.REVISION
    assert "native-written Egyptian items" in capsys.readouterr().out
    assert ood.main(["--models", str(tmp_path / "none"), "--data", str(f), "--out", str(out)]) == 2
