from __future__ import annotations

from pathlib import Path
import pytest
from typer.testing import CliRunner

import certs_cli.main as cli


runner = CliRunner()


def test_help_shows_convert_command():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    # Typer help output should list the command
    assert "convert" in result.stdout


def test_convert_single_calls_core(monkeypatch, tmp_path: Path):
    pfx = tmp_path / "a.pfx"
    pfx.write_bytes(b"dummy")
    out = tmp_path / "a.pem"

    called = {}

    def fake_pfx_to_pem(pfx_path, pem_path=None, **kwargs):
        called["pfx_path"] = Path(pfx_path)
        called["pem_path"] = Path(pem_path) if pem_path is not None else None
        called["kwargs"] = kwargs
        return out

    # Patch the names inside the CLI module (important!)
    monkeypatch.setattr(cli, "pfx_to_pem", fake_pfx_to_pem)

    result = runner.invoke(cli.app, ["convert", str(pfx), "--out", str(out)])
    assert result.exit_code == 0
    assert str(out) in result.stdout

    assert called["pfx_path"] == pfx
    assert called["pem_path"] == out


def test_convert_split_calls_core(monkeypatch, tmp_path: Path):
    pfx = tmp_path / "a.pfx"
    pfx.write_bytes(b"dummy")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    called = {}

    def fake_split(pfx_path, out_dir=None, **kwargs):
        called["pfx_path"] = Path(pfx_path)
        called["out_dir"] = Path(out_dir) if out_dir is not None else None
        called["kwargs"] = kwargs
        return {"cert": out_dir / "cert.pem", "key": out_dir / "key.pem"}

    monkeypatch.setattr(cli, "pfx_to_pem_split", fake_split)

    result = runner.invoke(cli.app, ["convert", str(pfx), "--split", "--out-dir", str(out_dir)])
    assert result.exit_code == 0
    assert "cert:" in result.stdout
    assert "key:" in result.stdout

    assert called["pfx_path"] == pfx
    assert called["out_dir"] == out_dir


def test_prompt_password_passes_password(monkeypatch, tmp_path: Path):
    pfx = tmp_path / "a.pfx"
    pfx.write_bytes(b"dummy")

    seen = {}

    def fake_pfx_to_pem(pfx_path, pem_path=None, **kwargs):
        seen.update(kwargs)
        return tmp_path / "out.pem"

    monkeypatch.setattr(cli, "pfx_to_pem", fake_pfx_to_pem)

    # --prompt-password triggers a prompt; provide input via runner.invoke
    result = runner.invoke(
        cli.app,
        ["convert", str(pfx), "--prompt-password"],
        input="supersecret\n",
    )
    assert result.exit_code == 0
    assert seen.get("password") == "supersecret"


def test_prompt_password_rejects_conflicting_password_flags(tmp_path: Path):
    pfx = tmp_path / "a.pfx"
    pfx.write_bytes(b"dummy")

    # prompt-password + password should error
    result = runner.invoke(cli.app, ["convert", str(pfx), "--prompt-password", "--password", "x"])
    assert result.exit_code != 0
    assert "Use only one of" in result.stdout or "Use only one of" in result.stderr


def test_password_env_passes_env_name(monkeypatch, tmp_path: Path):
    pfx = tmp_path / "a.pfx"
    pfx.write_bytes(b"dummy")

    seen = {}

    def fake_pfx_to_pem(pfx_path, pem_path=None, **kwargs):
        seen.update(kwargs)
        return tmp_path / "out.pem"

    monkeypatch.setattr(cli, "pfx_to_pem", fake_pfx_to_pem)

    result = runner.invoke(cli.app, ["convert", str(pfx), "--password-env", "PFX_PASS"])
    assert result.exit_code == 0
    assert seen.get("password_env") == "PFX_PASS"
