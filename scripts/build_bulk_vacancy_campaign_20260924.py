"""Build and validate ideal FCC starting geometries; never runs a DFT code.

Standard library only. Existing files are accepted only if byte-identical.
The common first vacancy is exactly at the cell centre after a rigid lattice
translation. Parent site IDs remain stable across all eight structures.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


CAMPAIGN = "al_bulk_vacancy_relax_20260924"
A_REF = 3.9545804060131293
REPEAT = 3
HALF_GRID = 2 * REPEAT
LENGTH = REPEAT * A_REF
ORIGIN_SHIFT_HALF = (1, 1, 1)
FIRST_PARENT_HALF = (2, 2, 2)
PAIR_SPECS = (
    ("2V_1NN_D110", 1, (1, 1, 0)),
    ("2V_2NN_D100", 2, (2, 0, 0)),
    ("2V_3NN_D211", 3, (2, 1, 1)),
    ("2V_4NN_D110", 4, (2, 2, 0)),
    ("2V_5NN_D310", 5, (3, 1, 0)),
    ("2V_6NN_D111", 6, (2, 2, 2)),
)


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image_shells(site_a, site_b):
    """Enumerate the 27 neighbouring translations, sufficient for this cube."""
    raw = [b - a for a, b in zip(site_a["fractional"], site_b["fractional"])]
    images = []
    for shift in itertools.product((-1, 0, 1), repeat=3):
        vector = [(raw[i] + shift[i]) * LENGTH for i in range(3)]
        images.append({
            "image_translation_of_second_site": list(shift),
            "vector_A": vector,
            "distance_A": norm(vector),
        })
    images.sort(key=lambda value: (round(value["distance_A"], 10), value["image_translation_of_second_site"]))
    shells = []
    for entry in images:
        if not shells or not math.isclose(entry["distance_A"], shells[-1]["distance_A"], abs_tol=1e-10):
            shells.append({"distance_A": entry["distance_A"], "images": []})
        shells[-1]["images"].append(entry)
    for shell in shells:
        shell["multiplicity"] = len(shell["images"])
    return shells


def poscar(case):
    lines = [f"{CAMPAIGN} {case['case_id']} ideal FCC; parent IDs in geometry.json", "1.0"]
    lines.extend(" ".join(f"{value:.16f}" for value in vector) for vector in case["cell_A"])
    lines += ["Al", str(case["n_atoms"]), "Direct"]
    lines.extend(" ".join(f"{value:.16f}" for value in coordinate) for coordinate in case["fractional"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def extxyz(case):
    lattice = " ".join(f"{value:.16f}" for row in case["cell_A"] for value in row)
    lines = [str(case["n_atoms"]),
             f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:parent_site_id:I:1 pbc="T T T" case_id={case["case_id"]}']
    lines.extend("Al " + " ".join(f"{value:.16f}" for value in coordinate) + f" {site_id}"
                 for site_id, coordinate in zip(case["remaining_parent_ids"], case["positions_A"]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def build():
    parent_half = [point for point in itertools.product(range(HALF_GRID), repeat=3) if sum(point) % 2 == 0]
    sites = []
    for site_id, original in enumerate(parent_half):
        centered = [(original[i] + ORIGIN_SHIFT_HALF[i]) % HALF_GRID for i in range(3)]
        fractional = [value / HALF_GRID for value in centered]
        sites.append({
            "site_id_0based": site_id,
            "parent_half_lattice_integer": list(original),
            "translated_half_lattice_integer": centered,
            "fractional": fractional,
            "cartesian_A": [value * LENGTH for value in fractional],
        })
    lookup = {tuple(site["parent_half_lattice_integer"]): site for site in sites}
    first = lookup[FIRST_PARENT_HALF]
    cell = [[LENGTH if i == j else 0.0 for j in range(3)] for i in range(3)]
    cases = {}
    specifications = [("0V", []), ("1V", [first])]
    pair_metadata = {}
    for case_id, shell, vector in PAIR_SPECS:
        target = tuple((FIRST_PARENT_HALF[i] + vector[i]) % HALF_GRID for i in range(3))
        second = lookup[target]
        specifications.append((case_id, [first, second]))
        shells = image_shells(first, second)
        divisor = math.gcd(*vector)
        direction = [value // divisor for value in vector]
        expected_distance = A_REF * norm(vector) / 2
        require(math.isclose(shells[0]["distance_A"], expected_distance, abs_tol=1e-10), f"Incorrect MIC distance: {case_id}")
        pair_metadata[case_id] = {
            "fcc_neighbour_shell": shell,
            "requested_vector_half_lattice_integer": list(vector),
            "requested_vector_A": [value * A_REF / 2 for value in vector],
            "direction_indices": direction,
            "direction_label": "[" + " ".join(map(str, direction)) + "]",
            "distance_definition": "Initial removed-site minimum-image distance; not a relaxed distance or a reaction coordinate.",
            "distance_A": shells[0]["distance_A"],
            "shortest_image_multiplicity": shells[0]["multiplicity"],
            "shortest_images": shells[0]["images"],
            "next_distinct_image_shell": shells[1],
            "periodic_image_shells_within_27_translations": shells,
            "nearest_same_vacancy_image_distance_A": LENGTH,
            "half_cell_image_degeneracy": shells[0]["multiplicity"] > 1,
            "role": "periodic-image diagnostic" if shell == 5 else "core pair comparison",
            "interpretation": "Finite periodic pair; not an isolated or infinitely separated reference.",
        }
    for case_id, removed in specifications:
        removed_ids = [site["site_id_0based"] for site in removed]
        remaining = [site for site in sites if site["site_id_0based"] not in removed_ids]
        case = {
            "case_id": case_id,
            "n_parent_sites": len(sites),
            "n_atoms": len(remaining),
            "vacancy_count": len(removed),
            "q": len(removed),
            "vacancy_fraction": len(removed) / len(sites),
            "cell_A": cell,
            "pbc": [True, True, True],
            "removed_parent_ids": removed_ids,
            "removed_indices_0based": removed_ids,
            "removed_sites": removed,
            "remaining_parent_ids": [site["site_id_0based"] for site in remaining],
            "remaining_parent_indices_0based": [site["site_id_0based"] for site in remaining],
            "fractional": [site["fractional"] for site in remaining],
            "positions_A": [site["cartesian_A"] for site in remaining],
            "structure_files": {
                "poscar": f"structures/{case_id}/POSCAR",
                "extxyz": f"structures/{case_id}/structure.extxyz",
            },
            "status": "ideal_start_geometry_only_not_calculated",
        }
        if case_id in pair_metadata:
            case["pair"] = pair_metadata[case_id]
        cases[case_id] = case
    manifest = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN,
        "purpose": "Matched ideal 0V, 1V and six 2V initial geometries for fixed-cell ionic relaxation.",
        "geometry_status": "generated_and_validated_not_calculated",
        "element": "Al",
        "crystal": "fcc; conventional cubic supercell",
        "a_ref_A": A_REF,
        "a_ref_meaning": "Retained common reference from the prior controlled comparison, not newly asserted equilibrium a0.",
        "repeat": [REPEAT] * 3,
        "n_parent_sites": len(sites),
        "cell_edge_A": LENGTH,
        "cell_A": cell,
        "pbc": [True, True, True],
        "lattice_origin_translation_A": [value * A_REF / 2 for value in ORIGIN_SHIFT_HALF],
        "site_id_rule": "0-based lexicographic (hx,hy,hz), integers 0..5 with even hx+hy+hz before rigid origin translation.",
        "first_vacancy_site_id": first["site_id_0based"],
        "first_vacancy_fractional": first["fractional"],
        "sites": sites,
        "case_order": list(cases),
        "cases": cases,
        "excluded_from_first_round": ["[110] vector (3,3,0) a/2 at 8.38893 A: boundary-tied farthest case"],
        "generator": "scripts/build_bulk_vacancy_campaign_20260924.py",
        "geometry_caveats": [
            "The 108-site periodic cell is approximately 12 A, not demonstrated infinite-size convergence.",
            "Pair positions are FCC lattice sites removed from one common reference; relaxation output must preserve parent-site mapping.",
            "5NN [310] has two equally close periodic images and is retained only as a labelled diagnostic.",
            "No electronic, force, stress, strain-field or energy result is implied by generated structures.",
        ],
    }
    validate(manifest)
    files = {}
    for case in cases.values():
        for kind, contents in (("poscar", poscar(case)), ("extxyz", extxyz(case))):
            path = case["structure_files"][kind]
            files[path] = contents
            case.setdefault("structure_sha256", {})[kind] = sha256(contents)
    manifest["validation"] = {
        "passed": True,
        "checks": ["108 unique FCC sites", "12 nearest neighbours per pristine site", "8 correct atom counts",
                   "common cell and centred first vacancy", "complete disjoint parent-site mapping",
                   "analytical shell distances and full periodic MIC enumeration", "5NN-only shortest-image degeneracy"],
    }
    files["geometry.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return manifest, files


def validate(manifest):
    sites = manifest["sites"]
    require(len(sites) == 108, "FCC site count must be 108")
    require(len({tuple(site["fractional"]) for site in sites}) == 108, "Duplicate sites")
    require(manifest["first_vacancy_fractional"] == [0.5, 0.5, 0.5], "First vacancy not centred")
    for site in sites:
        integers = site["parent_half_lattice_integer"]
        require(sum(integers) % 2 == 0, "Non-FCC site")
        require(all(0 <= value < 1 for value in site["fractional"]), "Site outside primary cell")
        # This tests FCC topology after wrapping and translation, independent of the pair labels.
        nn_count = 0
        for other in sites:
            delta = [b - a for a, b in zip(site["fractional"], other["fractional"])]
            distance = LENGTH * norm([value - round(value) for value in delta])
            if math.isclose(distance, A_REF / math.sqrt(2), abs_tol=1e-10):
                nn_count += 1
        require(nn_count == 12, f"Site {site['site_id_0based']} has {nn_count} nearest neighbours")
    require(len(manifest["cases"]) == 8, "Expected eight structures")
    all_ids = set(range(108))
    for case in manifest["cases"].values():
        removed, remaining = case["removed_parent_ids"], case["remaining_parent_ids"]
        require(len(remaining) == case["n_atoms"] == 108 - case["vacancy_count"], "Atom count mismatch")
        require(len(set(removed)) == len(removed), "Duplicate removed site")
        require(len(set(remaining)) == len(remaining), "Duplicate remaining site")
        require(set(removed).isdisjoint(remaining) and set(removed) | set(remaining) == all_ids, "Broken parent-site mapping")
        require(case["cell_A"] == manifest["cell_A"], "Inconsistent cells")
        require(len(case["positions_A"]) == len(case["fractional"]) == len(remaining), "Coordinate count mismatch")
        if removed:
            require(removed[0] == manifest["first_vacancy_site_id"], "First vacancy differs")
        if "pair" in case:
            pair = case["pair"]
            expected = 2 if pair["fcc_neighbour_shell"] == 5 else 1
            require(pair["shortest_image_multiplicity"] == expected, "Unexpected periodic-image degeneracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "campaigns" / CAMPAIGN)
    parser.add_argument("--validate-only", action="store_true", help="Check geometry and exact existing structure bytes without writing.")
    args = parser.parse_args()
    manifest, expected = build()
    root = args.output_root.resolve()
    for relative, contents in expected.items():
        target = root / relative
        if target.exists():
            require(target.read_bytes() == contents, f"Refusing changed existing file: {target}")
        elif args.validate_only:
            raise FileNotFoundError(target)
    if not args.validate_only:
        for relative, contents in expected.items():
            target = root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(contents)
    print(json.dumps({
        "status": "validated" if args.validate_only else "generated_and_validated",
        "output_root": str(root), "files": len(expected), "cell_edge_A": LENGTH,
        "first_vacancy_site_id": manifest["first_vacancy_site_id"],
        "cases": [{"case_id": case["case_id"], "n_atoms": case["n_atoms"],
                   "distance_A": case.get("pair", {}).get("distance_A"),
                   "shortest_image_multiplicity": case.get("pair", {}).get("shortest_image_multiplicity")}
                  for case in manifest["cases"].values()],
    }, indent=2))


if __name__ == "__main__":
    main()
