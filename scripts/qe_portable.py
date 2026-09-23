#!/usr/bin/env python3
"""Verified, fresh-SCF QE campaign execution; Python standard library only.

Inputs and pseudopotentials live in Git; scratch and outputs live outside it.
``prepare`` never launches QE. ``run`` requires --execute and always creates a
new attempt: QE wavefunctions/checkpoints are deliberately not migrated here.
The runner accepts the simple, explicit SCF input format used by this campaign.
It fails closed on unsupported geometry rather than silently interpreting it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = REPO_ROOT / "campaigns" / "qe_divacancy_20260923"
RY_TO_EV = 13.605693122994
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
ASSIGNMENT = re.compile(r"\b([a-zA-Z_]\w*(?:\(\d+\))?)\s*=\s*('(?:[^']|'')*'|\"[^\"]*\"|[^,\s]+)")
RUNTIME_KEYS = {"control": {"pseudo_dir", "outdir", "prefix", "max_seconds", "restart_mode"},
                "electrons": {"startingpot", "startingwfc"}}


class CampaignError(ValueError):
    """Input, provenance, or safety condition was not satisfied."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def save_json(path, value):
    # Atomic replacement only within this runner's newly created attempt.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def contained(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def campaign_file(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not contained(path, root) or not path.is_file():
        raise CampaignError(f"Missing or escaping campaign file: {name}")
    return path


def number(text):
    value = float(str(text).replace("D", "e").replace("d", "e"))
    if not math.isfinite(value):
        raise CampaignError("Non-finite numerical setting")
    return value


def strip_comments(text):
    # Preserve exclamation marks inside quoted file names.
    return re.sub(r"('[^']*'|\"[^\"]*\")|![^\n]*", lambda m: m.group(1) or "", text)


def unquote(text):
    return text.strip().strip("'\"")


def input_blocks(text):
    text = strip_comments(text)
    blocks = {}
    spans = {}
    # A closing slash is a standalone token, never a slash inside a path.
    expression = re.compile(r"&([A-Za-z]+)\b((?:'[^']*'|\"[^\"]*\"|[^/'\"]|/(?!\s*(?:\n|$)))*)/", re.S)
    for match in expression.finditer(text):
        name = match.group(1).lower()
        if name in blocks:
            raise CampaignError(f"Duplicate namelist: {name}")
        values = {}
        for item in ASSIGNMENT.finditer(match.group(2)):
            key = item.group(1).lower()
            if key in values:
                raise CampaignError(f"Duplicate {name}.{key}")
            values[key] = item.group(2).strip().lower()
        remainder = ASSIGNMENT.sub("", match.group(2)).replace(",", "").strip()
        if remainder:
            raise CampaignError(f"Unsupported syntax in &{name}: {remainder[:80]}")
        blocks[name] = values
        spans[name] = match.span()
    if not {"control", "system", "electrons"}.issubset(blocks):
        raise CampaignError("CONTROL, SYSTEM, and ELECTRONS namelists are required")
    if set(blocks) - {"control", "system", "electrons"}:
        raise CampaignError("This runner accepts fixed-geometry SCF only (no IONS/CELL namelists)")
    card_text = text
    for start, end in sorted(spans.values(), reverse=True):
        card_text = card_text[:start] + card_text[end:]
    return blocks, [line.strip() for line in card_text.splitlines() if line.strip()]


def parse_input(text, allow_restart=False):
    blocks, lines = input_blocks(text)
    control, system, electrons = (blocks[name] for name in ("control", "system", "electrons"))
    if unquote(control.get("calculation", "")) != "scf":
        raise CampaignError("Only calculation='scf' is supported")
    if unquote(control.get("restart_mode", "")) not in ({"from_scratch", "restart"} if allow_restart else {"from_scratch"}):
        raise CampaignError("A fresh-SCF campaign must explicitly set restart_mode='from_scratch'")
    if (unquote(electrons.get("startingpot", "")) not in ({"atomic", "file"} if allow_restart else {"atomic"})
            or unquote(electrons.get("startingwfc", "")) not in ({"atomic+random", "file"} if allow_restart else {"atomic+random"})):
        raise CampaignError("Set startingpot='atomic' and startingwfc='atomic+random' explicitly")
    for key in ("pseudo_dir", "outdir", "prefix"):
        if key not in control:
            raise CampaignError(f"Missing CONTROL.{key}")
    if number(system.get("ibrav", "nan")) != 0:
        raise CampaignError("Explicit CELL_PARAMETERS and ibrav=0 are required")
    nat_value, ntyp_value = number(system.get("nat", "nan")), number(system.get("ntyp", "nan"))
    nat, ntyp = int(nat_value), int(ntyp_value)
    if nat <= 0 or ntyp != 1 or nat != nat_value or ntyp != ntyp_value:
        raise CampaignError("Positive integer nat and ntyp=1 are required for this Al campaign")
    threshold = number(electrons.get("conv_thr", "nan"))
    if threshold <= 0:
        raise CampaignError("conv_thr must be positive")
    cards = {}
    index = 0
    while index < len(lines):
        header = lines[index]
        name = header.split()[0].upper()
        units = header[len(name):].strip().strip("{}()").strip().lower()
        if name in cards or name not in {"ATOMIC_SPECIES", "ATOMIC_POSITIONS", "CELL_PARAMETERS", "K_POINTS"}:
            raise CampaignError(f"Duplicate or unsupported card: {header}")
        if name == "ATOMIC_SPECIES":
            count = ntyp
        elif name == "ATOMIC_POSITIONS":
            count = nat
            if units not in {"angstrom", "crystal", "bohr"}:
                raise CampaignError("ATOMIC_POSITIONS needs explicit angstrom, bohr, or crystal units")
        elif name == "CELL_PARAMETERS":
            count = 3
            if units not in {"angstrom", "bohr"}:
                raise CampaignError("CELL_PARAMETERS needs explicit angstrom or bohr units")
        else:
            if units != "automatic":
                raise CampaignError("This campaign requires explicit K_POINTS automatic")
            count = 1
        rows = [line.split() for line in lines[index + 1:index + 1 + count]]
        if len(rows) != count:
            raise CampaignError(f"Wrong row count in {name}")
        cards[name] = {"units": units, "rows": rows}
        index += count + 1
    if len(cards) != 4:
        raise CampaignError("All four geometry/species/k-point cards are required")
    cell = cards["CELL_PARAMETERS"]["rows"]
    if any(len(row) != 3 for row in cell):
        raise CampaignError("CELL_PARAMETERS must contain three 3D vectors")
    c = [[number(value) for value in row] for row in cell]
    determinant = (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
                   - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
                   + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]))
    if abs(determinant) < 1e-12:
        raise CampaignError("Singular simulation cell")
    species = cards["ATOMIC_SPECIES"]["rows"]
    if len(species[0]) != 3 or species[0][0] != "Al" or number(species[0][1]) <= 0:
        raise CampaignError("Expected one explicit Al species and pseudopotential")
    seen = set()
    for row in cards["ATOMIC_POSITIONS"]["rows"]:
        if len(row) not in {4, 7} or row[0] != "Al":
            raise CampaignError("Wrong atom count or unsupported ATOMIC_POSITIONS row")
        point = tuple(number(value) for value in row[1:4])
        if point in seen:
            raise CampaignError("Duplicate atomic positions")
        seen.add(point)
        if len(row) == 7 and any(value not in {"0", "1"} for value in row[4:]):
            raise CampaignError("Invalid position constraint flags")
    kpoints = cards["K_POINTS"]["rows"][0]
    if len(kpoints) != 6 or any(not re.fullmatch(r"\d+", x) for x in kpoints):
        raise CampaignError("K_POINTS automatic requires six integers")
    if any(int(x) <= 0 for x in kpoints[:3]) or any(x not in {"0", "1"} for x in kpoints[3:]):
        raise CampaignError("Invalid automatic k-point mesh/offset")
    scientific = {
        "namelists": {name: {key: value for key, value in settings.items()
                             if key not in RUNTIME_KEYS.get(name, set())}
                      for name, settings in blocks.items()},
        "cards": cards,
    }
    common = json.loads(json.dumps(scientific))
    del common["cards"]["ATOMIC_POSITIONS"]
    # Historical generators printed equivalent cell vectors at different precision.
    # This 1e-12 A/bohr canonicalization is only for the shared-settings check;
    # original input bytes and per-case scientific hashes remain unchanged.
    common["cards"]["CELL_PARAMETERS"]["rows"] = [[round(v, 12) for v in row] for row in c]
    return {"scientific": scientific, "common": common, "conv_thr_Ry": threshold,
            "nat": nat, "pseudo_name": species[0][2], "blocks": blocks}


def verify_campaign(campaign):
    root = Path(campaign).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or not manifest.get("cases"):
        raise CampaignError("Expected manifest schema_version=1 with nonempty cases")
    sums = root / "SHA256SUMS.json"
    if sums.is_file():
        for name, expected in json.loads(sums.read_text(encoding="utf-8")).items():
            if digest_file(campaign_file(root, name)) != expected:
                raise CampaignError(f"Snapshot SHA-256 mismatch: {name}")
    pseudo = campaign_file(root, manifest["pseudo"]["file"])
    pseudo_sha = digest_file(pseudo)
    if pseudo_sha != manifest["pseudo"]["sha256"]:
        raise CampaignError("Pseudopotential SHA-256 mismatch")
    cases, common = {}, None
    for case in manifest["cases"]:
        case_id = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) or ".." in case_id or case_id in cases:
            raise CampaignError(f"Invalid or duplicate case ID: {case_id}")
        if case.get("status") not in {"pending", "completed"}:
            raise CampaignError(f"Unrecognized case status: {case_id}")
        source = campaign_file(root, case["input"])
        if digest_file(source) != case["sha256"]:
            raise CampaignError(f"Input SHA-256 mismatch: {case_id}")
        text = source.read_text(encoding="utf-8-sig")
        parsed = parse_input(text)
        if parsed["pseudo_name"] != pseudo.name:
            raise CampaignError(f"Pseudopotential filename mismatch: {case_id}")
        if common is not None and common != parsed["common"]:
            raise CampaignError(f"Scientific settings or cell differ between cases: {case_id}")
        common = parsed["common"]
        cases[case_id] = {**case, "source": source, "text": text, "parsed": parsed,
                          "scientific_sha256": digest_json({"input": parsed["scientific"], "pseudo_sha256": pseudo_sha})}
    pairs = manifest.get("pairs", [])
    pair_ids = set()
    for pair in pairs:
        if pair["id"] in pair_ids or pair["left"] not in cases or pair["right"] not in cases or pair["left"] == pair["right"]:
            raise CampaignError("Invalid manifest pair definition")
        pair_ids.add(pair["id"])
    protocol_sha = digest_json({"protocol": manifest.get("protocol", {}), "common": common, "pseudo_sha256": pseudo_sha})
    return {"root": root, "manifest": manifest, "cases": cases, "pseudo": pseudo,
            "pseudo_sha256": pseudo_sha, "protocol_sha256": protocol_sha}


def work_root_checked(work_root, campaign):
    root = Path(work_root).expanduser().resolve()
    if contained(root, REPO_ROOT) or contained(root, campaign["root"]):
        raise CampaignError("--work-root must be outside the Git repository and campaign")
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextlib.contextmanager
def case_lock(root, case_id):
    locks = root / ".locks"
    locks.mkdir(exist_ok=True)
    path = locks / (case_id + ".lock")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise CampaignError(f"Case locked: {path}. Inspect its host/PID; no automatic stale-lock deletion.") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "started_utc": utc_now()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def replace_setting(text, block_name, key, value):
    expression = re.compile(r"(&" + re.escape(block_name) + r"\b)((?:'[^']*'|\"[^\"]*\"|[^/'\"]|/(?!\s*(?:\n|$)))*)/", re.I | re.S)
    match = expression.search(text)
    if not match:
        raise CampaignError(f"Missing namelist {block_name}")
    body = match.group(2)
    target = re.compile(r"(\b" + re.escape(key) + r"\s*=\s*)('(?:[^']|'')*'|\"[^\"]*\"|[^,\s]+)", re.I)
    body, count = target.subn(lambda m: m.group(1) + value, body)
    if count == 0:
        body += f"  {key} = {value},\n"
    if count > 1:
        raise CampaignError(f"Duplicate setting {key}")
    return text[:match.start()] + match.group(1) + body + "/" + text[match.end():]


def positive(value, kind=float):
    result = kind(value)
    if not math.isfinite(result) or result <= 0:
        raise CampaignError("Runtime limits/thread counts must be positive")
    return result


def create_attempt(campaign, root, case_id, max_seconds=None, threads=1):
    if case_id not in campaign["cases"]:
        raise CampaignError(f"Unknown case: {case_id}")
    threads = positive(threads, int)
    if max_seconds is not None:
        max_seconds = positive(max_seconds)
    case = campaign["cases"][case_id]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    directory = root / case_id / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "scratch").mkdir()
    (directory / "pseudo").mkdir()
    pseudo = directory / "pseudo" / campaign["pseudo"].name
    pseudo.write_bytes(campaign["pseudo"].read_bytes())
    if digest_file(pseudo) != campaign["pseudo_sha256"]:
        raise CampaignError("Pseudopotential changed while snapshotting")
    text = case["text"]
    for key, value in {"pseudo_dir": directory / "pseudo", "outdir": directory / "scratch"}.items():
        path = str(value.resolve()).replace("\\", "/")
        if "'" in path or "\n" in path:
            raise CampaignError("QE runtime paths cannot contain single quotes or newlines")
        text = replace_setting(text, "control", key, "'" + path + "'")
    if max_seconds is not None:
        text = replace_setting(text, "control", "max_seconds", format(max_seconds, ".12g"))
    parsed = parse_input(text)
    scientific_sha = digest_json({"input": parsed["scientific"], "pseudo_sha256": campaign["pseudo_sha256"]})
    if scientific_sha != case["scientific_sha256"]:
        raise CampaignError("Runtime editing unexpectedly changed scientific settings")
    (directory / "pw.in").write_text(text, encoding="utf-8", newline="\n")
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {"qe_portable_attempt": 1, "case_id": case_id, "status": "prepared", "git_commit": commit,
              "prepared_utc": utc_now(), "host": socket.gethostname(), "python": sys.version,
              "campaign_manifest_sha256": digest_file(campaign["root"] / "manifest.json"),
              "template_sha256": case["sha256"], "scientific_sha256": scientific_sha,
              "protocol_sha256": campaign["protocol_sha256"], "pseudo_sha256": campaign["pseudo_sha256"],
              "pseudo_filename": pseudo.name, "runtime_input_sha256": digest_file(directory / "pw.in"),
              "conv_thr_Ry": parsed["conv_thr_Ry"], "nat": parsed["nat"],
              "max_seconds_override": max_seconds,
              "threads": {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
              "migration_mode": "fresh_scf_no_wavefunction_restart"}
    save_json(directory / "attempt.json", record)
    return directory, record


def parse_output(output, threshold, returncode):
    energies = re.findall(r"!\s*total\s+energy\s*=\s*(" + NUMBER + r")\s*Ry", output, re.I)
    accuracies = re.findall(r"estimated\s+scf\s+accuracy\s*[<=>]+\s*(" + NUMBER + r")\s*Ry", output, re.I)
    reasons = []
    if returncode != 0:
        reasons.append(f"return code is {returncode!r}, not 0")
    if not re.search(r"convergence\s+has\s+been\s+achieved", output, re.I):
        reasons.append("SCF convergence message missing")
    if not re.search(r"\bJOB\s+DONE\.", output, re.I):
        reasons.append("JOB DONE missing")
    if re.search(r"convergence\s+NOT\s+achieved|Error\s+in\s+routine|%{3,}|maximum\s+(?:number\s+of\s+)?(?:steps|iterations)|maximum\s+(?:cpu|wall)\s+time|time\s+limit\s+(?:reached|exceeded)", output, re.I):
        reasons.append("QE reports an error, nonconvergence, or runtime/iteration limit")
    energy = number(energies[-1]) if energies else None
    accuracy = number(accuracies[-1]) if accuracies else None
    if energy is None:
        reasons.append("final ! total energy missing")
    if accuracy is None or accuracy > threshold or accuracy < 0:
        reasons.append("last estimated SCF accuracy missing or above conv_thr")
    versions = re.findall(r"Program\s+PWSCF\s+v\.([\w.\-]+)", output)
    return {"complete": not reasons, "reasons": reasons, "qe_version": versions[-1] if versions else None, "energy_Ry": energy,
            "energy_eV": energy * RY_TO_EV if energy is not None else None,
            "last_scf_accuracy_Ry": accuracy, "conv_thr_Ry": threshold}


def prepare(campaign, work_root, case_id, max_seconds=None, threads=1):
    verified = verify_campaign(campaign)
    root = work_root_checked(work_root, verified)
    if case_id not in verified["cases"]:
        raise CampaignError(f"Unknown case: {case_id}")
    with case_lock(root, case_id):
        directory, record = create_attempt(verified, root, case_id, max_seconds, threads)
    return {"attempt": str(directory), "status": record["status"], "launched": False}


def run(campaign, work_root, case_id, *, execute=False, pw=None, max_seconds=None, threads=1):
    if not execute:
        raise CampaignError("run requires --execute; use prepare for a nonexecuting check")
    verified = verify_campaign(campaign)
    root = work_root_checked(work_root, verified)
    if case_id not in verified["cases"]:
        raise CampaignError(f"Unknown case: {case_id}")
    command = shlex.split(pw or os.environ.get("PW_COMMAND", "pw.x"))
    if not command or any(token in {"|", "||", "&&", ";", ">", "<"} for token in command):
        raise CampaignError("--pw/PW_COMMAND must be an executable argv, not shell syntax")
    with case_lock(root, case_id):
        directory, record = create_attempt(verified, root, case_id, max_seconds, threads)
        command += ["-in", "pw.in"]
        record.update(status="running", started_utc=utc_now(), command=command)
        save_json(directory / "attempt.json", record)
        environment = os.environ.copy()
        environment.update(record["threads"])
        code, exception = None, None
        with (directory / "pw.out").open("wb") as output, (directory / "pw.err").open("wb") as error:
            process = None
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
        result = parse_output((directory / "pw.out").read_text(encoding="utf-8", errors="replace"), record["conv_thr_Ry"], code)
        if exception:
            result["complete"] = False
            result["reasons"].append(exception)
        record.update(status="complete" if result["complete"] else "incomplete", finished_utc=utc_now(),
                      returncode=code, result=result, output_sha256=digest_file(directory / "pw.out"),
                      stderr_sha256=digest_file(directory / "pw.err"))
        save_json(directory / "attempt.json", record)
    return {"attempt": str(directory), **result}


def inspect_attempt(path, verified):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("qe_portable_attempt") != 1:
        return None
    case_id = record.get("case_id")
    row = {"case_id": case_id, "attempt": str(path.parent), "host": record.get("host"),
           "finished_utc": record.get("finished_utc", ""), "complete": False, "reasons": []}
    if case_id not in verified["cases"]:
        row["reasons"].append("Case not in current manifest")
        return row
    case = verified["cases"][case_id]
    for key, expected in {"protocol_sha256": verified["protocol_sha256"], "scientific_sha256": case["scientific_sha256"],
                          "pseudo_sha256": verified["pseudo_sha256"]}.items():
        if record.get(key) != expected:
            row["reasons"].append(f"{key} differs from current campaign")
    directory = path.parent
    for filename, key in (("pw.in", "runtime_input_sha256"), ("pw.out", "output_sha256"), ("pw.err", "stderr_sha256")):
        file = directory / filename
        if not file.is_file() or record.get(key) != digest_file(file):
            row["reasons"].append(f"Missing or modified {filename}")
    pseudo = directory / "pseudo" / str(record.get("pseudo_filename", ""))
    if not pseudo.is_file() or digest_file(pseudo) != verified["pseudo_sha256"]:
        row["reasons"].append("Missing or modified pseudopotential snapshot")
    if record.get("status") != "complete" or record.get("returncode") != 0:
        row["reasons"].append("Attempt not recorded as successfully complete")
    if row["reasons"]:
        return row
    parsed = parse_input((directory / "pw.in").read_text(encoding="utf-8"))
    if digest_json({"input": parsed["scientific"], "pseudo_sha256": verified["pseudo_sha256"]}) != case["scientific_sha256"]:
        row["reasons"].append("Runtime scientific input does not match current case")
        return row
    output = (directory / "pw.out").read_text(encoding="utf-8", errors="replace")
    row.update(parse_output(output, parsed["conv_thr_Ry"], record.get("returncode")))
    return row


def inspect_archived(evidence, verified):
    case_id = evidence["case"]
    if case_id not in verified["cases"]:
        raise CampaignError("Archived result case not in campaign")
    files = {}
    for key in ("input", "output", "result"):
        files[key] = campaign_file(verified["root"], evidence[key])
        if digest_file(files[key]) != evidence[key + "_sha256"]:
            raise CampaignError(f"Archived {key} hash mismatch for {case_id}")
    if evidence["pseudo_sha256"] != verified["pseudo_sha256"]:
        raise CampaignError("Archived pseudopotential mismatch")
    parsed = parse_input(files["input"].read_text(encoding="utf-8-sig"), allow_restart=True)
    science = digest_json({"input": parsed["scientific"], "pseudo_sha256": verified["pseudo_sha256"]})
    if science != verified["cases"][case_id]["scientific_sha256"]:
        raise CampaignError("Archived scientific settings/geometry differ from campaign")
    record = json.loads(files["result"].read_text(encoding="utf-8-sig"))
    row = parse_output(files["output"].read_text(encoding="utf-8", errors="replace"), parsed["conv_thr_Ry"], record.get("exit_code"))
    if (record.get("status") != "scf_converged" or not row["complete"]
            or abs(record.get("energy_Ry", float("inf")) - row["energy_Ry"]) > 5e-9):
        raise CampaignError(f"Archived output does not support recorded energy: {case_id}")
    return {**row, "case_id": case_id, "attempt": str(files["output"].parent),
            "finished_utc": record.get("finished_utc", ""), "host": "archived source machine",
            "source": "verified archived result", "protocol_sha256": verified["protocol_sha256"]}


def collect(campaign, work_root):
    verified = verify_campaign(campaign)
    root = work_root_checked(work_root, verified)
    attempts = [inspect_archived(evidence, verified) for evidence in verified["manifest"].get("archived_results", [])]
    for path in sorted(root.glob("*/*/attempt.json")):
        if not contained(path, root):
            continue
        try:
            row = inspect_attempt(path, verified)
        except (ValueError, OSError, KeyError) as exc:
            row = {"attempt": str(path.parent), "complete": False, "reasons": [str(exc)]}
        if row is not None:
            attempts.append(row)
    selected = {}
    for row in sorted(attempts, key=lambda item: (item.get("finished_utc", ""), item["attempt"])):
        if row["complete"]:
            selected[row["case_id"]] = row
    pairs = []
    for pair in verified["manifest"].get("pairs", []):
        missing = [key for key in (pair["left"], pair["right"]) if key not in selected]
        result = {**pair, "complete": not missing, "definition": "E(left) - E(right)",
                  "delta_Ry": None, "delta_eV": None, "delta_meV": None}
        if missing:
            result["reason"] = "No verified complete runner attempt: " + ", ".join(missing)
        elif (not selected[pair["left"]].get("qe_version")
              or selected[pair["left"]].get("qe_version") != selected[pair["right"]].get("qe_version")):
            result.update(complete=False, reason="QE versions differ or are missing; reconcile versions before comparison")
        else:
            delta = selected[pair["left"]]["energy_Ry"] - selected[pair["right"]]["energy_Ry"]
            result.update(delta_Ry=delta, delta_eV=delta * RY_TO_EV, delta_meV=delta * RY_TO_EV * 1000,
                          left_attempt=selected[pair["left"]]["attempt"], right_attempt=selected[pair["right"]]["attempt"])
        pairs.append(result)
    report = {"collected_utc": utc_now(), "protocol_sha256": verified["protocol_sha256"],
              "selection": "latest verified complete attempt per case; failed newer attempts are retained below",
              "legacy_evidence": "Only explicitly declared, hash-verified input/output/result archives are accepted",
              "attempts": attempts, "selected": selected, "pairs": pairs}
    destination = root / "collections" / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "summary.json", report)
    columns = ["id", "left", "right", "complete", "definition", "delta_Ry", "delta_eV", "delta_meV", "reason", "left_attempt", "right_attempt"]
    with (destination / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    return {"report": str(destination / "summary.json"), "pairs_csv": str(destination / "pairs.csv"),
            "complete_cases": list(selected), "pairs": pairs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("verify", "prepare", "run", "collect"):
        child = subparsers.add_parser(action)
        child.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
        if action != "verify":
            child.add_argument("--work-root", type=Path, required=True)
        if action in {"prepare", "run"}:
            child.add_argument("--case", required=True)
            child.add_argument("--max-seconds", type=float)
            child.add_argument("--threads", type=int, default=1, help="OMP threads; BLAS threads remain 1")
        if action == "run":
            child.add_argument("--pw", help="Executable argv, e.g. 'srun pw.x'; default PW_COMMAND or pw.x")
            child.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "verify":
            verified = verify_campaign(args.campaign)
            result = {"verified": True, "campaign": str(verified["root"]), "protocol_sha256": verified["protocol_sha256"],
                      "pseudo_sha256": verified["pseudo_sha256"],
                      "cases": [{"id": key, "nat": case["parsed"]["nat"], "status": case["status"],
                                 "scientific_sha256": case["scientific_sha256"]} for key, case in verified["cases"].items()]}
        elif args.action == "prepare":
            result = prepare(args.campaign, args.work_root, args.case, args.max_seconds, args.threads)
        elif args.action == "run":
            result = run(args.campaign, args.work_root, args.case, execute=args.execute, pw=args.pw,
                         max_seconds=args.max_seconds, threads=args.threads)
        else:
            result = collect(args.campaign, args.work_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 1 if args.action == "run" and not result["complete"] else 0
    except (CampaignError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"qe_portable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
