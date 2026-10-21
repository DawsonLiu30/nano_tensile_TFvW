"""Pure postprocessing tests: no live DFT calculator and no production writes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_bulk_vacancy_dftpy_20260924.py"
SPEC = importlib.util.spec_from_file_location("bulk_vacancy_analysis", PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_real_zero_vacancy_unit_difference_and_component_reconciliation():
    terms = {"TOTAL": -6797.929042609489, "KEDF-TF": 2313.1160338708833,
             "KEDF-VW": 143.96670124003197, "XC": -2463.0133018516967,
             "HARTREE": 39.679980783770304, "PSEUDO": 1281.927909777924,
             "II": -8113.606366430402, "KEDF": 2457.082735110915}
    raw = -6797.928987244351
    energy_Ha = raw / analysis.RAW_ASE_HA_TO_EV
    assert analysis.normalize_energy(raw, terms, energy_Ha) == terms["TOTAL"]
    with pytest.raises(ValueError, match="conversion"):
        analysis.normalize_energy(terms["TOTAL"], terms, energy_Ha)
    corrupt = dict(terms, KEDF=2 * terms["KEDF"])
    with pytest.raises(ValueError, match="kinetic"):
        analysis.normalize_energy(raw, corrupt, energy_Ha)


def test_periodic_displacement_retains_raw_and_removes_translation():
    initial = np.array([[9.9, 0., 0.], [2., 0., 0.]])
    final = np.array([[0.2, 0.1, 0.], [2.1, 0.1, 0.]])
    raw, detrended, mean = analysis.displacements(initial, final, np.eye(3) * 10)
    np.testing.assert_allclose(raw, [[0.3, 0.1, 0.], [0.1, 0.1, 0.]], atol=1e-14)
    np.testing.assert_allclose(mean, [0.2, 0.1, 0.], atol=1e-14)
    np.testing.assert_allclose(detrended, [[0.1, 0., 0.], [-0.1, 0., 0.]], atol=1e-14)


def test_formation_and_binding_reservoir_and_sign():
    rows = []
    for case, q, energy, ts in (("0V", 0, -1080., 108.), ("1V", 1, -1069., 107.3),
                                ("2V_1NN_D110", 2, -1058.4, 106.5)):
        rows.append({"case": case, "q": q, "energy_eV_common": energy,
                     "Ts_eV": ts, "weighted_TF_eV": ts * 0.8, "weighted_vW_eV": ts * 0.2})
    analysis.add_energy_differences(rows)
    assert rows[1]["E_formation_eV"] == pytest.approx(1.)
    assert rows[2]["E_formation_eV"] == pytest.approx(1.6)
    assert rows[2]["E_binding_eV"] == pytest.approx(0.4)
    assert rows[2]["Ts_formation_eV"] == pytest.approx(0.5)
    assert rows[2]["Ts_binding_eV"] == pytest.approx(0.1)
    assert rows[1]["E_binding_eV"] is None
    assert rows[2]["weighted_TF_binding_eV"] + rows[2]["weighted_vW_binding_eV"] == pytest.approx(0.1)


def test_incomplete_campaign_cannot_create_final_output(tmp_path):
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(tmp_path / "runs", "attempt_001", output)
    assert not output.exists()


def test_artifact_digest_and_path_escape_rejected(tmp_path, monkeypatch):
    file = tmp_path / "one.txt"
    file.write_text("original")
    monkeypatch.setattr(analysis, "REQUIRED_ARTIFACTS", {"one.txt"})
    result = {"artifact_sha256": {"one.txt": analysis.sha256(file)}}
    analysis.verify_artifacts(tmp_path, result)
    file.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        analysis.verify_artifacts(tmp_path, result)
    with pytest.raises(ValueError, match="escapes"):
        analysis.safe_child(tmp_path, "../outside")
