# tests/fusion/test_parity_check.py
"""Smoke tests for parity check."""

from __future__ import annotations

from backend.fusion.parity_check import ParityResult, save_parity_report


def test_parity_result_report_formatting():
    r = ParityResult(
        v2_sharpe=1.2,
        v3_sharpe=1.5,
        v2_hit_rate=0.54,
        v3_hit_rate=0.57,
        v2_loss=0.005,
        v3_loss=0.004,
        passed=True,
    )
    report = r.report()
    assert "PASSED" in report
    assert "1.200" in report or "1.20" in report


def test_save_parity_report(tmp_path):
    r = ParityResult(
        v2_sharpe=1.0,
        v3_sharpe=0.9,
        v2_hit_rate=0.5,
        v3_hit_rate=0.48,
        v2_loss=0.006,
        v3_loss=0.007,
        passed=False,
    )
    out = tmp_path / "report.md"
    save_parity_report(r, out)
    text = out.read_text()
    assert "FAILED" in text
    assert "Sharpe Ratio" in text
