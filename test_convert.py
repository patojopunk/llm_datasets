from __future__ import annotations

from pathlib import Path
import types
import pytest

import certs_core.convert as conv


def test_passin_args():
    assert conv._passin_args(None, None) == []
    assert conv._passin_args("x", None) == ["-passin", "pass:x"]
    assert conv._passin_args(None, "VAR") == ["-passin", "env:VAR"]


def test_passin_args_rejects_both():
    with pytest.raises(ValueError):
        conv._passin_args("x", "VAR")


def test_resolve_openssl_uses_explicit_path():
    assert conv._resolve_openssl("/custom/openssl") == "/custom/openssl"


def test_resolve_openssl_not_found(monkeypatch):
    monkeypatch.setattr(conv.shutil, "which", lambda name: None)
    with pytest.raises(conv.OpenSSLNotFoundError):
        conv._resolve_openssl(None)


def test_run_success(monkeypatch):
    def fake_run(cmd, capture_output, text):
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(conv.subprocess, "run", fake_run)
    conv._run(["openssl", "version"])  # should not raise


def test_run_failure_raises(monkeypatch):
    def fake_run(cmd, capture_output, text):
        return types.SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(conv.subprocess, "run", fake_run)

    with pytest.raises(conv.OpenSSLError) as e:
        conv._run(["openssl", "pkcs12", "-in", "x.pfx"])

    msg = str(e.value)
    assert "OpenSSL command failed" in msg
    assert "STDERR" in msg


def test_pfx_to_pem_builds_expected_command(monkeypatch, tmp_path: Path):
    # Make openssl resolution deterministic
    monkeypatch.setattr(conv, "_resolve_openssl", lambda openssl=None: "openssl")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(conv.subprocess, "run", fake_run)

    pfx = tmp_path / "in.pfx"
    pfx.write_bytes(b"dummy")
    out = tmp_path / "out.pem"

    got = conv.pfx_to_pem(
        pfx,
        out,
        password_env="PFX_PASS",
        include_chain=False,
    )
    assert got == out
    assert len(calls) == 1

    cmd = calls[0]
    assert cmd[0] == "openssl"
    assert cmd[1:3] == ["pkcs12", "-in"]
    assert "-nodes" in cmd
    assert "-clcerts" in cmd  # include_chain=False
    assert cmd[-2:] == ["-passin", "env:PFX_PASS"]


def test_pfx_to_pem_refuses_overwrite(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(conv, "_resolve_openssl", lambda openssl=None: "openssl")
    monkeypatch.setattr(conv.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))

    pfx = tmp_path / "in.pfx"
    pfx.write_bytes(b"dummy")
    out = tmp_path / "out.pem"
    out.write_text("exists")

    with pytest.raises(FileExistsError):
        conv.pfx_to_pem(pfx, out, overwrite=False)


def test_pfx_to_pem_split_calls_three_commands_when_chain_succeeds(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(conv, "_resolve_openssl", lambda openssl=None: "openssl")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(conv.subprocess, "run", fake_run)

    pfx = tmp_path / "in.pfx"
    pfx.write_bytes(b"dummy")
    out_dir = tmp_path / "out"

    outputs = conv.pfx_to_pem_split(pfx, out_dir=out_dir, password="pw", include_chain=True)
    assert outputs["cert"].name == "cert.pem"
    assert outputs["key"].name == "key.pem"
    assert outputs["chain"].name == "chain.pem"

    assert len(calls) == 3
    # verify some flags exist across calls
    assert any("-clcerts" in c for c in calls)
    assert any("-nocerts" in c for c in calls)
    assert any("-cacerts" in c for c in calls)
    assert all(["-passin", "pass:pw"] == c[-2:] for c in calls)


def test_pfx_to_pem_split_omits_chain_if_openssl_fails_for_chain(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(conv, "_resolve_openssl", lambda openssl=None: "openssl")

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        # Make chain extraction fail only
        if "-cacerts" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="no ca certs")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(conv.subprocess, "run", fake_run)

    pfx = tmp_path / "in.pfx"
    pfx.write_bytes(b"dummy")

    outputs = conv.pfx_to_pem_split(pfx, out_dir=tmp_path, include_chain=True)
    assert "cert" in outputs and "key" in outputs
    assert "chain" not in outputs  # chain failure is tolerated
    assert len(calls) == 3  # still attempted
