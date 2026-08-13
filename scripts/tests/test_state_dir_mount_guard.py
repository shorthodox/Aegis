"""A writable path is not a persistent path, and only is_mount() knows.

2026-08-13: the published track record was found empty — 0 signals, 0 closed
trades, balance back at the $10,000 initial. Not corrupted, not partially
lost. Reset, and reset on every deploy before that one.

No Railway volume had ever been attached to the service. AEGIS_STATE_DIR was
unset, so config.py resolved STATE_DIR to the in-repo data/ dir, which inside
the container is /app/data on the ephemeral overlay. Five redeploys that week,
five wipes.

Nothing complained, and that is the part worth a test. The resolution code ends
in:

    STATE_DIR.mkdir(parents=True, exist_ok=True)

which CREATES the directory when it is missing. An unmounted path is therefore
indistinguishable from a mounted one by every signal the process had: it
exists, it is writable, writes succeed, reads come back. The Firestore mirror
that was supposed to cover this case had never written a document either — it
targets a database that does not exist — so both safety nets were down at once
and neither said so.

The guard added alongside these tests refuses to boot in a Railway environment
when state would land somewhere ephemeral. These tests pin the four decisions it
makes, because a guard that silently stops guarding is worse than none.
"""
import pytest

from scripts.engine import config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Neither variable set unless a test sets it."""
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("AEGIS_STATE_DIR", raising=False)
    monkeypatch.delenv("AEGIS_ALLOW_EPHEMERAL_STATE", raising=False)


def test_local_dev_is_never_blocked(monkeypatch, tmp_path):
    """Off Railway, the in-repo data/ dir is correct and must not raise.

    tmp_path is deliberately NOT a mount point. The guard has to stay quiet
    anyway, or every local run and every CI job breaks.
    """
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    config._assert_state_dir_is_persistent()


def test_railway_without_state_dir_refuses_to_boot(monkeypatch, tmp_path):
    """The exact production condition on 2026-08-13: on Railway, var unset."""
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    with pytest.raises(RuntimeError) as exc:
        config._assert_state_dir_is_persistent()

    msg = str(exc.value)
    assert "AEGIS_STATE_DIR" in msg
    assert "destroyed on the next redeploy" in msg


def test_railway_with_unmounted_path_refuses_to_boot(monkeypatch, tmp_path):
    """The subtler failure: the var IS set, but points at a plain directory.

    This is what a half-done fix looks like — someone sets AEGIS_STATE_DIR=/data
    and never attaches the volume. The path exists and is writable, so every
    check short of is_mount() passes.
    """
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AEGIS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    assert tmp_path.exists() and not tmp_path.is_mount()  # the trap, stated

    with pytest.raises(RuntimeError) as exc:
        config._assert_state_dir_is_persistent()
    assert "NOT a mount point" in str(exc.value)


def test_railway_with_real_mount_boots(monkeypatch, tmp_path):
    """A genuinely mounted volume passes."""
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AEGIS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    monkeypatch.setattr(type(tmp_path), "is_mount", lambda self: True)

    config._assert_state_dir_is_persistent()


def test_escape_hatch_boots_but_is_loud(monkeypatch, tmp_path, capsys):
    """The incident override works, and cannot be used quietly.

    Booting without a volume must stay possible during an outage, but it must
    never be silent — a warning nobody prints is how this bug survived weeks.
    """
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setenv("AEGIS_ALLOW_EPHEMERAL_STATE", "1")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)

    config._assert_state_dir_is_persistent()

    out = capsys.readouterr().out
    assert "mount guard" in out.lower()
    assert "destroyed" in out.lower()
