#!/usr/bin/env python3
"""Hash-verified, fresh fixed-cell QE relaxations for the 20260924 bulk campaign.

Only ``run --execute --pw '...'`` starts a calculation. All attempts and scratch
must live outside Git; each attempt is new and a per-case lock prevents overlap.
This runner deliberately does not reuse the historical fixed-geometry SCF runner.
Python standard library only; execution snapshots also work on Linux compute nodes.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shlex
import signal
import socket
import subprocess
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = REPO_ROOT / "campaigns/al_bulk_vacancy_relax_20260924"
CAMPAIGN_ID = "al_bulk_vacancy_relax_20260924"
RY_TO_EV = 13.605693122994
BOHR_TO_A = 0.529177210903
FORCE_TO_EV_A = RY_TO_EV / BOHR_TO_A
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
ASSIGNMENT = re.compile(r"\b([a-zA-Z_]\w*(?:\(\d+\))?)\s*=\s*('(?:[^']|'')*'|\"[^\"]*\"|[^,\s]+)")
NAMELIST = re.compile(r"&([A-Za-z]+)\b((?:'[^']*'|\"[^\"]*\"|[^/'\"]|/(?!\s*(?:\n|$)))*)/", re.S)
CONTROL_OPERATIONAL = {"pseudo_dir", "outdir", "wfcdir", "prefix", "max_seconds", "restart_mode",
                       "disk_io", "verbosity", "wf_collect", "lkpoint_dir"}
CONTROL_SCIENCE = {"calculation", "nstep", "tstress", "tprnfor", "etot_conv_thr", "forc_conv_thr", "iprint"}
ELECTRON_OPERATIONAL = {"startingpot", "startingwfc"}
CASE_IDS = {"0V", "1V", "2V_1NN_D110", "2V_2NN_D100", "2V_3NN_D211", "2V_4NN_D110", "2V_5NN_D310", "2V_6NN_D111"}


class CampaignError(ValueError):
    """A provenance, input-format, or execution-safety requirement failed."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def contained(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def campaign_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise CampaignError(f"Campaign paths must be relative POSIX paths: {name!r}")
    if ".." in Path(name).parts:
        raise CampaignError(f"Parent traversal in campaign path: {name}")
    path = (root / name).resolve()
    if not contained(path, root) or not path.is_file():
        raise CampaignError(f"Missing or escaping campaign file: {name}")
    return path


def number(text):
    try:
        value = float(str(text).replace("D", "e").replace("d", "e"))
    except (ValueError, TypeError):
        raise CampaignError(f"Invalid numeric value: {text!r}") from None
    if not math.isfinite(value):
        raise CampaignError("Non-finite numeric value")
    return value


def unquote(text):
    return str(text).strip().strip("'\"")


def strip_comments(text):
    return re.sub(r"('[^']*'|\"[^\"]*\")|![^\n]*", lambda m: m.group(1) or "", text)


def parse_input(text):
    """Parse a deliberately narrow, explicit Al fixed-cell relaxation input."""
    clean = strip_comments(text)
    blocks, spans = {}, []
    for match in NAMELIST.finditer(clean):
        name = match.group(1).lower()
        if name in blocks:
            raise CampaignError(f"Duplicate namelist: {name}")
        values = {}
        for item in ASSIGNMENT.finditer(match.group(2)):
            key = item.group(1).lower()
            if key in values:
                raise CampaignError(f"Duplicate {name}.{key}")
            values[key] = item.group(2).strip()
        if ASSIGNMENT.sub("", match.group(2)).replace(",", "").strip():
            raise CampaignError(f"Unsupported namelist syntax: {name}")
        blocks[name] = values
        spans.append(match.span())
    if set(blocks) != {"control", "system", "electrons", "ions"}:
        raise CampaignError("Exactly CONTROL, SYSTEM, ELECTRONS and IONS are required; CELL is forbidden")
    control, system, electrons, ions = [blocks[x] for x in ("control", "system", "electrons", "ions")]
    if set(control) - CONTROL_SCIENCE - CONTROL_OPERATIONAL:
        raise CampaignError("Unsupported CONTROL settings: " + ", ".join(sorted(set(control) - CONTROL_SCIENCE - CONTROL_OPERATIONAL)))
    if unquote(control.get("calculation", "")).lower() != "relax":
        raise CampaignError("calculation='relax' is required; SCF and vc-relax are not this campaign")
    if number(system.get("ibrav", "nan")) != 0 or number(system.get("ntyp", "nan")) != 1:
        raise CampaignError("Explicit ibrav=0 and ntyp=1 are required")
    nat_value = number(system.get("nat", "nan"))
    nat = int(nat_value)
    if nat != nat_value or nat not in {106, 107, 108}:
        raise CampaignError("This campaign requires nat 108, 107 or 106")
    if unquote(ions.get("ion_dynamics", "")).lower() != "bfgs":
        raise CampaignError("ion_dynamics='bfgs' is required")
    if set(ions) - {"ion_dynamics", "upscale", "bfgs_ndim", "trust_radius_max", "trust_radius_min", "trust_radius_ini", "w_1", "w_2"}:
        raise CampaignError("Unsupported IONS settings (no restart or constrained dynamics)")
    for key in ("tstress", "tprnfor"):
        if control.get(key, "").lower() != ".true.":
            raise CampaignError(f"{key}=.true. is required")
    for name, value in (("conv_thr", electrons.get("conv_thr")), ("forc_conv_thr", control.get("forc_conv_thr")),
                        ("etot_conv_thr", control.get("etot_conv_thr"))):
        if number(value) <= 0:
            raise CampaignError(f"Positive {name} required")
    if number(control.get("nstep", "nan")) != 200:
        raise CampaignError("nstep=200 required")
    for key, expected in {"ecutwfc": 60, "ecutrho": 240, "nbnd": 200, "degauss": 0.02}.items():
        if not math.isclose(number(system.get(key, "nan")), expected, abs_tol=1e-12):
            raise CampaignError(f"Unexpected SYSTEM.{key}")
    if unquote(system.get("occupations", "")).lower() != "smearing" or unquote(system.get("smearing", "")).lower() not in {"mv", "cold", "marzari-vanderbilt"}:
        raise CampaignError("Marzari-Vanderbilt smearing required")
    if "input_dft" in system and unquote(system["input_dft"]).lower() not in {"lda", "pz"}:
        raise CampaignError("Only LDA/PZ is allowed with this campaign pseudopotential")
    if not math.isclose(number(electrons["conv_thr"]), 1e-9, rel_tol=1e-10) or not math.isclose(number(control["etot_conv_thr"]), 1e-6, rel_tol=1e-10):
        raise CampaignError("Expected conv_thr=1e-9 Ry and etot_conv_thr=1e-6 Ry")
    for start, end in reversed(spans):
        clean = clean[:start] + clean[end:]
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    cards, index = {}, 0
    while index < len(lines):
        header = lines[index]
        name = header.split()[0].upper()
        units = header[len(name):].strip().strip("{}()").strip().lower()
        if name in cards or name not in {"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS"}:
            raise CampaignError(f"Unsupported or duplicate card: {header}")
        count = {"ATOMIC_SPECIES": 1, "ATOMIC_POSITIONS": nat, "CELL_PARAMETERS": 3, "K_POINTS": 1}[name]
        rows = [line.split() for line in lines[index + 1:index + count + 1]]
        if len(rows) != count:
            raise CampaignError(f"Missing rows in {name}")
        cards[name] = {"units": units, "rows": rows}
        index += count + 1
    if len(cards) != 4:
        raise CampaignError("All species, position, cell, k-point cards are required")
    species = cards["ATOMIC_SPECIES"]["rows"][0]
    if len(species) != 3 or species[0] != "Al" or species[2] != "Al.pz-vbc.UPF" or number(species[1]) <= 0:
        raise CampaignError("Expected the Al.pz-vbc.UPF Al species")
    cell_card = cards["CELL_PARAMETERS"]
    if cell_card["units"] not in {"angstrom", "bohr"} or any(len(row) != 3 for row in cell_card["rows"]):
        raise CampaignError("CELL_PARAMETERS requires three explicit vectors in angstrom or bohr")
    factor = BOHR_TO_A if cell_card["units"] == "bohr" else 1.0
    cell = [[number(x) * factor for x in row] for row in cell_card["rows"]]
    edge = 11.863741218
    if any(abs(cell[i][j] - (edge if i == j else 0.0)) > 1e-8 for i in range(3) for j in range(3)):
        raise CampaignError("Cell must be the fixed 11.863741218 A cubic bulk cell")
    if cards["K_POINTS"]["units"] != "automatic" or cards["K_POINTS"]["rows"] != [["3", "3", "3", "0", "0", "0"]]:
        raise CampaignError("Expected K_POINTS automatic: 3 3 3 0 0 0")
    pos = cards["ATOMIC_POSITIONS"]
    if pos["units"] not in {"angstrom", "bohr", "crystal"}:
        raise CampaignError("Explicit angstrom, bohr or crystal positions required")
    points = []
    for row in pos["rows"]:
        if len(row) != 4 or row[0] != "Al":
            raise CampaignError("Expected unconstrained Al positions")
        points.append([number(x) for x in row[1:]])
    if len({tuple(p) for p in points}) != nat:
        raise CampaignError("Duplicate atomic positions")
    positions = convert_positions(points, pos["units"], cell)
    science = {"namelists": {name: {key: value for key, value in values.items()
                if key not in (CONTROL_OPERATIONAL if name == "control" else ELECTRON_OPERATIONAL if name == "electrons" else set())}
                for name, values in blocks.items()}, "cards": cards}
    return {"nat": nat, "blocks": blocks, "cards": cards, "scientific": science,
            "conv_thr_Ry": number(electrons["conv_thr"]), "force_threshold_Ry_bohr": number(control["forc_conv_thr"]),
            "cell_A": cell, "positions_A": positions, "pseudo_name": species[2]}


def convert_positions(points, units, cell, alat_A=None):
    if units == "angstrom":
        return points
    if units == "bohr":
        return [[v * BOHR_TO_A for v in row] for row in points]
    if units == "crystal":
        return [[sum(row[i] * cell[i][j] for i in range(3)) for j in range(3)] for row in points]
    if units.startswith("alat") and alat_A is not None:
        return [[v * alat_A for v in row] for row in points]
    raise CampaignError(f"Unsupported position units: {units}")


def checked_bytes(path, expected, label):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CampaignError(f"Invalid SHA-256 for {label}")
    data = path.read_bytes()
    if sha_bytes(data) != expected:
        raise CampaignError(f"SHA-256 mismatch: {label}")
    return data


def check_poscar(data, parsed):
    lines = [line.strip() for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    if len(lines) != 8 + parsed["nat"] or lines[5] != "Al" or lines[6] != str(parsed["nat"]):
        raise CampaignError("POSCAR must contain only the corresponding unconstrained Al structure")
    scale = number(lines[1])
    if scale <= 0:
        raise CampaignError("Positive POSCAR scale required")
    cell = [[number(v) * scale for v in line.split()] for line in lines[2:5]]
    if any(len(row) != 3 for row in cell) or any(abs(cell[i][j] - parsed["cell_A"][i][j]) > 1e-8 for i in range(3) for j in range(3)):
        raise CampaignError("POSCAR cell differs from QE cell")
    if lines[7].lower() not in {"direct", "cartesian"}:
        raise CampaignError("Unsupported POSCAR coordinate mode")
    rows = [[number(v) for v in line.split()] for line in lines[8:]]
    if any(len(row) != 3 for row in rows):
        raise CampaignError("Invalid POSCAR coordinates")
    positions = convert_positions(rows, "crystal", cell) if lines[7].lower() == "direct" else [[v * scale for v in row] for row in rows]
    if any(abs(a - b) > 1e-8 for left, right in zip(positions, parsed["positions_A"]) for a, b in zip(left, right)):
        raise CampaignError("POSCAR and QE initial coordinates/order differ")


def verify_campaign(campaign):
    root = Path(campaign).resolve()
    manifest_path = campaign_file(root, "qe_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    if manifest.get("schema_version") != 1 or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("Expected this campaign's schema_version=1 qe_manifest.json")
    if not isinstance(manifest.get("cases"), dict) or not manifest["cases"]:
        raise CampaignError("Manifest cases must be a nonempty dictionary")
    if not isinstance(manifest.get("protocol"), dict):
        raise CampaignError("Manifest protocol must be a dictionary")
    pseudo = campaign_file(root, manifest["pseudo"]["file"])
    if pseudo.name != "Al.pz-vbc.UPF":
        raise CampaignError("Unexpected pseudopotential filename")
    pseudo_bytes = checked_bytes(pseudo, manifest["pseudo"]["sha256"], "pseudopotential")
    geometry_bytes = None
    if "geometry_manifest" in manifest:
        item = manifest["geometry_manifest"]
        geometry_bytes = checked_bytes(campaign_file(root, item["file"]), item["sha256"], "geometry manifest")
    cases, common = {}, None
    for case_id, case in manifest["cases"].items():
        if case_id not in CASE_IDS:
            raise CampaignError(f"Unknown case ID: {case_id}")
        source = campaign_file(root, case["input"])
        data = checked_bytes(source, case["sha256"], case_id + " input")
        parsed = parse_input(data.decode("utf-8-sig"))
        expected_q = 0 if case_id == "0V" else 1 if case_id == "1V" else 2
        if case.get("q") != expected_q or case.get("nat") != 108 - expected_q or parsed["nat"] != case["nat"]:
            raise CampaignError(f"Atom/vacancy count mismatch: {case_id}")
        structure = campaign_file(root, case["structure"])
        structure_bytes = checked_bytes(structure, case["structure_sha256"], case_id + " structure")
        check_poscar(structure_bytes, parsed)
        optional = {}
        if "extxyz" in case:
            path = campaign_file(root, case["extxyz"])
            optional = {"path": path, "bytes": checked_bytes(path, case["extxyz_sha256"], case_id + " extxyz")}
        shared = json.loads(json.dumps(parsed["scientific"]))
        del shared["cards"]["ATOMIC_POSITIONS"]
        del shared["namelists"]["system"]["nat"]
        if common is not None and common != shared:
            raise CampaignError(f"Scientific settings differ between cases: {case_id}")
        common = shared
        cases[case_id] = {"metadata": case, "source": source, "input_bytes": data, "parsed": parsed,
                          "structure": structure, "structure_bytes": structure_bytes, "extxyz": optional,
                          "scientific_sha256": sha_json({"input": parsed["scientific"], "pseudo_sha256": manifest["pseudo"]["sha256"]})}
        protocol = manifest["protocol"]
        for protocol_key, block, input_key in (("forc_conv_thr_Ry_bohr", "control", "forc_conv_thr"),
                                                ("etot_conv_thr_Ry", "control", "etot_conv_thr"),
                                                ("conv_thr_Ry", "electrons", "conv_thr"),
                                                ("electron_maxstep", "electrons", "electron_maxstep")):
            if protocol_key in protocol and not math.isclose(number(protocol[protocol_key]), number(parsed["blocks"][block].get(input_key, "nan")), rel_tol=1e-10):
                raise CampaignError(f"Manifest {protocol_key} disagrees with {case_id} input")
    edge = manifest["protocol"].get("cell_edge_A")
    if edge is not None and abs(number(edge) - 11.863741218) > 1e-8:
        raise CampaignError("Manifest cell_edge_A disagrees with input")
    return {"root": root, "manifest": manifest, "manifest_bytes": manifest_bytes,
            "manifest_sha256": sha_bytes(manifest_bytes), "geometry_bytes": geometry_bytes, "pseudo": pseudo, "pseudo_bytes": pseudo_bytes,
            "pseudo_sha256": manifest["pseudo"]["sha256"], "cases": cases,
            "protocol_sha256": sha_json({"protocol": manifest["protocol"], "common": common, "pseudo_sha256": manifest["pseudo"]["sha256"]})}


def runtime_input(parsed, prefix, max_seconds):
    blocks = json.loads(json.dumps(parsed["blocks"]))
    blocks["control"] = {k: v for k, v in blocks["control"].items() if k not in CONTROL_OPERATIONAL}
    blocks["control"].update(pseudo_dir="'./pseudo'", outdir="'./scratch'", prefix=f"'{prefix}'",
                             restart_mode="'from_scratch'", max_seconds=format(max_seconds, ".12g"),
                             disk_io="'low'", verbosity="'high'")
    for key in ELECTRON_OPERATIONAL:
        blocks["electrons"].pop(key, None)
    blocks["electrons"].update(startingpot="'atomic'", startingwfc="'atomic+random'")
    output = []
    for name in ("control", "system", "electrons", "ions"):
        output.extend(["&" + name.upper(), *[f"  {key} = {value}," for key, value in blocks[name].items()], "/"])
    for name in ("ATOMIC_SPECIES", "CELL_PARAMETERS", "ATOMIC_POSITIONS", "K_POINTS"):
        card = parsed["cards"][name]
        output.append(name + (" " + card["units"] if card["units"] else ""))
        output.extend(" ".join(row) for row in card["rows"])
    return "\n".join(output) + "\n"


def run_root_checked(run_root, campaign_root):
    root = Path(run_root).expanduser().resolve()
    if root == Path(root.anchor) or contained(root, REPO_ROOT) or contained(root, campaign_root):
        raise CampaignError("Run root must be outside the repository/campaign and not a filesystem root")
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise CampaignError("Run root cannot be inside any Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextlib.contextmanager
def case_lock(root, case_id):
    if case_id not in CASE_IDS:
        raise CampaignError("Invalid lock case")
    lock_dir = Path(root) / ".locks" / CAMPAIGN_ID
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / (case_id + ".lock")
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise CampaignError(f"Case already locked: {path}; inspect host/PID/job before manual recovery") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "job_id": os.environ.get("SLURM_JOB_ID"),
                       "token": token, "started_utc": utc_now()}, handle)
        yield
    finally:
        # Do not remove a lock another process replaced during manual intervention.
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("token") == token:
                path.unlink()
        except (OSError, ValueError):
            pass


def command_argv(pw):
    if not pw:
        raise CampaignError("--pw must explicitly name the verified QE/MPI argv")
    try:
        command = shlex.split(pw)
    except ValueError as exc:
        raise CampaignError(str(exc)) from exc
    if not command or any(x in {"|", "||", "&&", ";", ">", ">>", "<", "&"} for x in command):
        raise CampaignError("--pw is an argv, not shell syntax")
    if any(x.lower() in {"-in", "-inp", "-input", "--input", "-i"} or x.startswith(("-in=", "--input=")) for x in command):
        raise CampaignError("The runner exclusively supplies the QE input argument")
    return command + ["-in", "pw.in"]


def snapshot_file(path, data):
    with Path(path).open("xb") as handle:
        handle.write(data)
    Path(path).chmod(0o444)


def create_attempt(verified, root, case_id, command, threads=1, max_seconds=39600):
    if not isinstance(threads, int) or threads < 1 or not 0 < number(max_seconds) <= 39600:
        raise CampaignError("Positive threads and 0 < max_seconds <= 39600 required")
    case = verified["cases"][case_id]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    directory = root / case_id / stamp
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("source", "pseudo", "scratch"):
        (directory / name).mkdir()
    snapshots = {"source/qe_manifest.json": verified["manifest_bytes"], "source/pw.in": case["input_bytes"],
                 "source/POSCAR": case["structure_bytes"], "pseudo/Al.pz-vbc.UPF": verified["pseudo_bytes"],
                 "source/runner.py": Path(__file__).read_bytes()}
    if verified["geometry_bytes"] is not None:
        snapshots["source/geometry.json"] = verified["geometry_bytes"]
    if case["extxyz"]:
        snapshots["source/structure.extxyz"] = case["extxyz"]["bytes"]
    for filename, data in snapshots.items():
        snapshot_file(directory / filename, data)
    text = runtime_input(case["parsed"], "al_" + case_id + "_" + uuid.uuid4().hex[:12], max_seconds)
    runtime_parsed = parse_input(text)
    science_sha = sha_json({"input": runtime_parsed["scientific"], "pseudo_sha256": verified["pseudo_sha256"]})
    if science_sha != case["scientific_sha256"]:
        raise CampaignError("Runtime sanitization changed scientific input")
    snapshot_file(directory / "pw.in", text.encode("utf-8"))
    file_hashes = {filename: sha_bytes(data) for filename, data in snapshots.items()}
    file_hashes["pw.in"] = sha_file(directory / "pw.in")
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {"bulk_vacancy_qe_attempt": 1, "campaign_id": CAMPAIGN_ID, "case_id": case_id,
              "status": "prepared", "prepared_utc": utc_now(), "source_repo": str(REPO_ROOT), "git_commit": commit,
              "host": socket.gethostname(), "pid": os.getpid(), "python": sys.version, "command": command,
              "manifest_sha256": verified["manifest_sha256"], "protocol_sha256": verified["protocol_sha256"],
              "scientific_sha256": science_sha, "source_input_sha256": case["metadata"]["sha256"],
              "runtime_input_sha256": file_hashes["pw.in"], "pseudo_sha256": verified["pseudo_sha256"],
              "immutable_file_sha256": file_hashes, "protocol": verified["manifest"]["protocol"],
              "nat": case["parsed"]["nat"], "q": case["metadata"]["q"], "max_seconds": max_seconds,
              "resources": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST")},
              "thread_environment": {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
              "execution_mode": "fresh_fixed_cell_relaxation_no_restart"}
    save_json(directory / "attempt.json", record)
    return directory, record


def immutable_errors(directory, record):
    errors = []
    for name, expected in record["immutable_file_sha256"].items():
        try:
            path = campaign_file(directory, name)
            if sha_file(path) != expected:
                errors.append("Immutable file modified: " + name)
        except (CampaignError, OSError):
            errors.append("Immutable file missing/escaping: " + name)
    return errors


def geometry_frames(output, parsed):
    frames, errors = [], []
    alat = re.findall(r"lattice parameter\s*\(alat\)\s*=\s*(" + NUMBER + r")\s*a\.u\.", output, re.I)
    alat_A = number(alat[-1]) * BOHR_TO_A if alat else None
    lines = output.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"\s*ATOMIC_POSITIONS\s*[({]?\s*([^)}\n]*)[)}]?\s*$", line, re.I)
        if not match:
            continue
        units = match.group(1).strip().lower()
        explicit_alat = re.search(r"alat\s*=\s*(" + NUMBER + r")", units)
        this_alat = number(explicit_alat.group(1)) * BOHR_TO_A if explicit_alat else alat_A
        points = []
        for row in lines[i + 1:i + 1 + parsed["nat"]]:
            fields = row.split()
            if len(fields) < 4 or fields[0] != "Al":
                break
            try:
                points.append([number(v) for v in fields[1:4]])
            except CampaignError:
                break
        if len(points) != parsed["nat"]:
            errors.append(f"Incomplete position block at output line {i + 1}")
            continue
        try:
            positions = convert_positions(points, units, parsed["cell_A"], this_alat)
        except CampaignError as exc:
            errors.append(str(exc))
            continue
        frames.append({"source_line": i + 1, "positions_A": positions, "cell_A": parsed["cell_A"],
                       "symbols": ["Al"] * parsed["nat"], "source_units": units})
    return frames, errors


def parse_output(output, parsed, returncode=0, stderr="", force_target_eV_A=0.001):
    """Distinguish electronic convergence, ionic convergence, and normal program exit."""
    energies = re.findall(r"!\s*total\s+energy\s*=\s*(" + NUMBER + r")\s*Ry", output, re.I)
    final_energy = re.findall(r"Final\s+energy\s*=\s*(" + NUMBER + r")\s*Ry", output, re.I)
    accuracies = re.findall(r"estimated\s+scf\s+accuracy\s*[<=>]+\s*(" + NUMBER + r")\s*Ry", output, re.I)
    # QE's BFGS failure line also contains "convergence not achieved". Only
    # standalone SCF status lines describe electronic convergence.
    messages = list(re.finditer(r"^[ \t]*convergence[ \t]+(?:has[ \t]+been[ \t]+achieved|NOT[ \t]+achieved)\b[^\r\n]*\r?$",
                                output, re.I | re.M))
    electronic = bool(messages and "not" not in messages[-1].group(0).lower())
    accuracy = number(accuracies[-1]) if accuracies else None
    electronic = electronic and accuracy is not None and 0 <= accuracy <= parsed["conv_thr_Ry"]
    ionic = bool(re.search(r"\bbfgs\s+converged\s+in\b|\bionic\s+convergence\s+(?:has\s+been\s+achieved|achieved)\b", output, re.I))
    combined = output + "\n" + stderr
    ionic_optimizer_failed = bool(re.search(r"\bbfgs\s+failed\s+after\b", combined, re.I))
    ionic_stop_reason = None
    if ionic_optimizer_failed:
        ionic = False
        reset_limit = re.search(r"history\s+already\s+reset\s+at\s+previous\s+step:\s*(?:exiting|stopping)", combined, re.I)
        ionic_stop_reason = "bfgs_trust_radius_exhausted" if reset_limit else "bfgs_failed"
    time_limit = bool(re.search(r"maximum\s+(?:cpu\s+|wall\s+)?time\s+(?:exceeded|reached)|max_seconds|time\s+limit\s+(?:exceeded|reached)|DUE\s+TO\s+TIME\s+LIMIT", combined, re.I))
    step_limit = bool(re.search(r"(?:maximum|max)\s+(?:number\s+of\s+)?(?:ionic\s+|bfgs\s+)?steps\s+(?:has\s+been\s+|is\s+)?(?:reached|exceeded)|bfgs\s+not\s+converged", combined, re.I))
    fatal = bool(re.search(r"Error\s+in\s+routine|%{3,}|MPI_ABORT|segmentation\s+fault", combined, re.I))
    done = bool(re.search(r"\bJOB\s+DONE\.", output, re.I))
    frames, geometry_errors = geometry_frames(output, parsed)
    final_region = re.search(r"Begin\s+final\s+coordinates(.*?)End\s+final\s+coordinates", output, re.I | re.S)
    final_frames = geometry_frames(final_region.group(1), parsed)[0] if final_region else []
    # Final coordinates must be explicitly emitted, not just the latest trial step.
    final_geometry = final_frames[-1] if final_frames else None
    force_blocks = []
    pattern = re.compile(r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*(" + NUMBER + r")\s+(" + NUMBER + r")\s+(" + NUMBER + r")", re.I)
    for part in re.split(r"Forces acting on atoms", output, flags=re.I)[1:]:
        rows = pattern.findall(part.split("total   stress", 1)[0])
        if len(rows) >= parsed["nat"]:
            force_blocks.append([[number(v) for v in row] for row in rows[:parsed["nat"]]])
    forces = force_blocks[-1] if force_blocks else None
    forces_ev = [[v * FORCE_TO_EV_A for v in row] for row in forces] if forces else None
    max_force = max(math.sqrt(sum(v * v for v in row)) for row in forces_ev) if forces_ev else None
    stress = None
    stress_pattern = re.compile(r"total\s+stress\s*\([^\n]*\)\s*[^\n]*\n((?:[^\n]*\n){3})", re.I)
    for match in stress_pattern.finditer(output):
        rows = [re.findall(NUMBER, line) for line in match.group(1).splitlines()]
        if len(rows) == 3 and all(len(row) >= 6 for row in rows):
            stress = {"Ry_bohr3": [[number(v) for v in row[:3]] for row in rows],
                      "kbar": [[number(v) for v in row[3:6]] for row in rows]}
    energy = number(energies[-1]) if energies else None
    reasons = []
    if returncode != 0:
        reasons.append(f"Process return code: {returncode!r}")
    if not electronic:
        reasons.append("Final electronic convergence/accuracy criterion not established")
    if not ionic:
        reasons.append("Explicit BFGS ionic convergence not reported")
    if ionic_optimizer_failed:
        reasons.append("BFGS trust-radius/history reset limit reached before ionic convergence"
                       if ionic_stop_reason == "bfgs_trust_radius_exhausted"
                       else "BFGS optimizer reported failure before ionic convergence")
    if not done:
        reasons.append("JOB DONE missing")
    if time_limit:
        reasons.append("QE/scheduler time limit reached")
    if step_limit:
        reasons.append("Ionic step limit reached")
    if fatal:
        reasons.append("QE or MPI error reported")
    if energy is None:
        reasons.append("No final SCF total energy")
    if final_energy and energy is not None and abs(number(final_energy[-1]) - energy) > 1e-5:
        reasons.append("Final and last SCF energies disagree")
    converged = not reasons
    data_reasons = []
    if final_geometry is None:
        data_reasons.append("Final coordinates unavailable")
    if forces is None:
        data_reasons.append("Final per-atom forces unavailable")
    if stress is None:
        data_reasons.append("Final stress unavailable")
    force_ok = max_force is not None and max_force <= force_target_eV_A + 1e-8
    if converged and not force_ok:
        data_reasons.append("QE ionic criteria passed, but atom force norm target needs tighter relaxation/review")
    status = "converged" if converged else "time_limit" if time_limit else "step_limit" if step_limit else "process_failed" if returncode != 0 or fatal else "electronic_not_converged" if not electronic else "ionic_optimizer_failed" if ionic_optimizer_failed else "ionic_not_converged"
    if converged and data_reasons:
        status = "converged_needs_review"
    versions = re.findall(r"Program\s+PWSCF\s+v\.([\w.\-]+)", output)
    return {"status": status, "complete": converged, "usable": converged and not data_reasons,
            "electronic_converged": electronic, "ionic_converged": ionic, "job_done": done,
            "ionic_optimizer_failed": ionic_optimizer_failed, "ionic_stop_reason": ionic_stop_reason,
            "time_limit": time_limit, "step_limit": step_limit, "reasons": reasons, "data_quality_reasons": data_reasons,
            "qe_version": versions[-1] if versions else None, "energy_Ry": energy,
            "energy_eV": energy * RY_TO_EV if energy is not None else None,
            "last_scf_accuracy_Ry": accuracy, "conv_thr_Ry": parsed["conv_thr_Ry"],
            "forces_Ry_bohr": forces, "forces_eV_A": forces_ev, "max_force_norm_eV_A": max_force,
            "force_norm_target_eV_A": force_target_eV_A, "force_norm_target_pass": force_ok,
            "input_force_threshold_Ry_bohr": parsed["force_threshold_Ry_bohr"], "stress": stress,
            "final_geometry": final_geometry, "trajectory": frames, "geometry_parse_notes": geometry_errors}


def write_derived(directory, result):
    save_json(directory / "trajectory.json", result["trajectory"])
    if result["final_geometry"] is not None:
        save_json(directory / "final_geometry.json", result["final_geometry"])
        frame = result["final_geometry"]
        lattice = " ".join(format(v, ".14g") for row in frame["cell_A"] for v in row)
        lines = [str(len(frame["positions_A"])), f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"']
        lines.extend("Al " + " ".join(format(v, ".14g") for v in row) for row in frame["positions_A"])
        (directory / "final.extxyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(directory / "result.json", result)


def run(campaign, run_root, case_id, *, execute=False, pw=None, threads=1, max_seconds=39600):
    if not execute:
        raise CampaignError("Production run requires --execute; use verify/dry-check otherwise")
    command = command_argv(pw)
    verified = verify_campaign(campaign)
    if case_id not in verified["cases"]:
        raise CampaignError(f"Case absent from manifest: {case_id}")
    root = run_root_checked(run_root, verified["root"])
    with case_lock(root, case_id):
        directory, record = create_attempt(verified, root, case_id, command, threads, max_seconds)
        errors = immutable_errors(directory, record)
        if errors:
            raise CampaignError("; ".join(errors))
        record.update(status="running", started_utc=utc_now())
        save_json(directory / "attempt.json", record)
        environment = os.environ.copy()
        environment.update(record["thread_environment"])
        code, exception, process = None, None, None
        with (directory / "pw.out").open("xb") as output, (directory / "pw.err").open("xb") as error:
            try:
                process = subprocess.Popen(command, cwd=directory, env=environment, stdout=output, stderr=error,
                                           start_new_session=(os.name != "nt"))
                code = process.wait()
            except KeyboardInterrupt:
                exception = "Interrupted by operator"
                if process is not None:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        code = process.wait()
            except OSError as exc:
                exception = str(exc)
        parsed = verified["cases"][case_id]["parsed"]
        result = parse_output((directory / "pw.out").read_text(encoding="utf-8", errors="replace"), parsed, code,
                              (directory / "pw.err").read_text(encoding="utf-8", errors="replace"))
        errors = immutable_errors(directory, record)
        if exception:
            errors.append(exception)
        if errors:
            result.update(complete=False, usable=False, status="provenance_or_execution_failed")
            result["reasons"].extend(errors)
        write_derived(directory, result)
        record.update(status=result["status"], finished_utc=utc_now(), returncode=code,
                      output_sha256=sha_file(directory / "pw.out"), stderr_sha256=sha_file(directory / "pw.err"),
                      result_sha256=sha_file(directory / "result.json"), complete=result["complete"], usable=result["usable"])
        save_json(directory / "attempt.json", record)
    return {"attempt": str(directory), "status": result["status"], "complete": result["complete"], "usable": result["usable"],
            "energy_Ry": result["energy_Ry"], "max_force_norm_eV_A": result["max_force_norm_eV_A"],
            "reasons": result["reasons"], "data_quality_reasons": result["data_quality_reasons"]}


def inspect_attempt(attempt, campaign):
    directory = Path(attempt).resolve()
    record = json.loads((directory / "attempt.json").read_text(encoding="utf-8"))
    if record.get("bulk_vacancy_qe_attempt") != 1 or record.get("campaign_id") != CAMPAIGN_ID:
        raise CampaignError("Not a bulk vacancy relaxation attempt")
    verified = verify_campaign(campaign)
    case = verified["cases"].get(record.get("case_id"))
    if case is None:
        raise CampaignError("Attempt case absent from current manifest")
    errors = immutable_errors(directory, record)
    for key, expected in {"manifest_sha256": verified["manifest_sha256"], "protocol_sha256": verified["protocol_sha256"],
                          "scientific_sha256": case["scientific_sha256"], "pseudo_sha256": verified["pseudo_sha256"]}.items():
        if record.get(key) != expected:
            errors.append("Current campaign mismatch: " + key)
    for filename, key in (("pw.out", "output_sha256"), ("pw.err", "stderr_sha256"), ("result.json", "result_sha256")):
        if not (directory / filename).is_file() or sha_file(directory / filename) != record.get(key):
            errors.append("Missing/modified output: " + filename)
    if errors:
        return {"attempt": str(directory), "complete": False, "usable": False, "status": "provenance_failed", "reasons": errors}
    parsed = parse_input((directory / "pw.in").read_text(encoding="utf-8"))
    actual_science = sha_json({"input": parsed["scientific"], "pseudo_sha256": verified["pseudo_sha256"]})
    if actual_science != case["scientific_sha256"]:
        raise CampaignError("Runtime scientific hash mismatch")
    result = parse_output((directory / "pw.out").read_text(encoding="utf-8", errors="replace"), parsed, record.get("returncode"),
                          (directory / "pw.err").read_text(encoding="utf-8", errors="replace"))
    if not record.get("finished_utc") or record.get("status") == "running":
        result.update(complete=False, usable=False, status="attempt_not_finalized")
        result["reasons"].append("Attempt did not finalize")
    return {"attempt": str(directory), "case_id": record["case_id"], **result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("verify", "dry-check", "run", "inspect"):
        child = sub.add_parser(action)
        child.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
        if action in {"dry-check", "run"}:
            child.add_argument("--case", required=True)
        if action == "run":
            child.add_argument("--run-root", "--work-root", dest="run_root", type=Path, required=True)
            child.add_argument("--execute", action="store_true")
            child.add_argument("--pw", required=True)
            child.add_argument("--threads", type=int, default=1)
            child.add_argument("--max-seconds", type=float, default=39600)
        if action == "inspect":
            child.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action in {"verify", "dry-check"}:
            verified = verify_campaign(args.campaign)
            if args.action == "dry-check" and args.case not in verified["cases"]:
                raise CampaignError("Case absent from manifest")
            result = {"verified": True, "launched": False, "campaign": str(verified["root"]),
                      "protocol_sha256": verified["protocol_sha256"], "manifest_sha256": verified["manifest_sha256"],
                      "cases": {key: {"nat": value["parsed"]["nat"], "scientific_sha256": value["scientific_sha256"]}
                                for key, value in verified["cases"].items() if args.action == "verify" or key == args.case}}
        elif args.action == "run":
            result = run(args.campaign, args.run_root, args.case, execute=args.execute, pw=args.pw,
                         threads=args.threads, max_seconds=args.max_seconds)
        else:
            result = inspect_attempt(args.attempt, args.campaign)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 1 if args.action in {"run", "inspect"} and not result["usable"] else 0
    except (CampaignError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"bulk_vacancy_qe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
