from pathlib import Path

import pytest

from selfdgs import _outputs


def test_install_staged_outputs_restores_previous_outputs_on_failure(
    tmp_path,
    monkeypatch,
):
    outdir = tmp_path / "results"
    staged = tmp_path / "staged"
    outdir.mkdir()
    staged.mkdir()
    (outdir / "first.txt").write_text("old first")
    (outdir / "second.txt").write_text("old second")
    (outdir / "unrelated.txt").write_text("keep me")
    (staged / "first.txt").write_text("new first")
    (staged / "second.txt").write_text("new second")

    real_move = _outputs._move_path

    def fail_on_second_staged_move(source: Path, destination: Path) -> None:
        if source.parent == staged and source.name == "second.txt":
            raise OSError("simulated install failure")
        real_move(source, destination)

    monkeypatch.setattr(_outputs, "_move_path", fail_on_second_staged_move)

    with pytest.raises(OSError, match="simulated install failure"):
        _outputs.install_staged_outputs(
            staged,
            outdir,
            owned_names={"first.txt", "second.txt"},
        )

    assert (outdir / "first.txt").read_text() == "old first"
    assert (outdir / "second.txt").read_text() == "old second"
    assert (outdir / "unrelated.txt").read_text() == "keep me"
