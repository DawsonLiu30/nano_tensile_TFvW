"""Shared, minimum-image geometry for conventional cubic FCC divacancies.

Fixed-direction scans and neighbour-shell comparisons are distinct protocols.
At a half-cell boundary every equally short periodic image is considered, so
the [110] half-diagonal remains a valid minimum-image pair.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
from ase.build import bulk

DFTPY_A0_A = 3.9545804060131293


def parse_repeat(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in text.lower().replace(',', 'x').split('x'))
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('repeat must look like 3x3x3') from exc
    if len(values) != 3 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError('repeat must contain three positive integers')
    return values


def parse_direction(text: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(v) for v in text.replace('[', '').replace(']', '').replace(',', ' ').split())
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('direction must look like 1,1,0 or [1 1 0]') from exc
    if len(values) != 3 or values == (0, 0, 0):
        raise argparse.ArgumentTypeError('direction must contain three nonzero-together integer indices')
    divisor = math.gcd(*values)
    values = tuple(v // divisor for v in values)
    if next(v for v in values if v) < 0:
        values = tuple(-v for v in values)
    return values


def direction_label(direction) -> str:
    # Delimit multi-digit indices; [110] remains compatible with prior manifests.
    separator = ' ' if any(abs(v) >= 10 for v in direction) else ''
    return '[' + separator.join(str(int(v)) for v in direction) + ']'


def validate_positive(**values) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')


def prepare_output_directory(path: Path) -> None:
    """Never erase an earlier calculation when preparing new inputs."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output already exists and is nonempty: {path}. Choose a new run directory.')
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_centered_pristine(a0: float, repeat: tuple[int, int, int]):
    validate_positive(a0=a0)
    if len(repeat) != 3 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in repeat):
        raise ValueError('repeat must contain three positive integers')
    atoms = bulk('Al', 'fcc', a=a0, cubic=True).repeat(repeat)
    scaled = atoms.get_scaled_positions(wrap=True)
    target = np.array([0.5, 0.5, 0.5])
    # Preserve the historical centred structure and atom indices in cubic scans.
    center_index = int(np.argmin(np.sum((scaled - target) ** 2, axis=1)))
    shift = target - scaled[center_index]
    atoms.set_scaled_positions((scaled + shift) % 1.0)
    atoms.wrap()
    return atoms, center_index, shift.tolist()


def minimum_image_vectors(delta, cell, pbc, tolerance=1e-8):
    """All shortest images in a positive, axis-aligned conventional cell."""
    cell = np.asarray(cell, dtype=float)
    lengths = np.diag(cell)
    if not np.allclose(cell, np.diag(lengths), atol=1e-10) or np.any(lengths <= 0):
        raise ValueError('pair preparation requires an axis-aligned conventional cubic/orthorhombic cell')
    options = []
    for value, length, periodic in zip(delta, lengths, pbc):
        wrapped = float(value - math.floor(value / length + 0.5) * length) if periodic else float(value)
        if periodic and abs(abs(wrapped) - length / 2) <= tolerance:
            options.append((-length / 2, length / 2))
        else:
            options.append((wrapped,))
    return [np.asarray(v) for v in itertools.product(*options)]


def _lattice_direction(delta, a0, tolerance=1e-5):
    lattice = np.asarray(delta) * 2 / a0
    integers = np.rint(lattice).astype(int)
    if not np.allclose(lattice, integers, atol=tolerance, rtol=0) or not np.any(integers):
        raise ValueError('vacancy displacement is not an FCC lattice vector at the declared a0')
    if int(integers.sum()) % 2:
        raise ValueError('vacancy displacement is not an FCC translation (odd index sum)')
    return parse_direction(','.join(map(str, integers))), integers


def _cubic_family(direction):
    return tuple(sorted(map(abs, direction), reverse=True))


def enumerate_pairs(pristine, center_index, *, selection='fixed_direction', direction=(1, 1, 0),
                    z_tol=1e-6, distance_tol=1e-4, direction_tol=1e-6):
    """Return (distance, atom index, MIC displacement) representatives.

    shells includes 3D neighbours and retains inequivalent cubic direction
    families at the same radius. It is not a fixed-direction radial curve.
    """
    validate_positive(z_tol=z_tol, distance_tol=distance_tol, direction_tol=direction_tol)
    if selection not in ('fixed_direction', 'shells'):
        raise ValueError('pair selection must be fixed_direction or shells')
    if center_index < 0 or center_index >= len(pristine):
        raise ValueError('center_index is outside the pristine structure')
    direction = parse_direction(','.join(map(str, direction)))
    axis = np.asarray(direction, dtype=float)
    axis /= np.linalg.norm(axis)
    candidates = []
    for idx, pos in enumerate(pristine.positions):
        if idx == center_index:
            continue
        for delta in minimum_image_vectors(pos - pristine.positions[center_index], pristine.cell, pristine.pbc):
            distance = float(np.linalg.norm(delta))
            if distance <= distance_tol:
                continue
            if selection == 'fixed_direction':
                if direction[2] == 0 and abs(delta[2]) > z_tol:
                    continue
                if np.linalg.norm(delta - np.dot(delta, axis) * axis) > direction_tol:
                    continue
            candidates.append((distance, idx, delta))
    grouped = []
    # Direction family is derived from normalized Cartesian components; all
    # generated cells are conventional FCC, so these ratios are integral.
    family_keys = []
    def representative_order(item):
        distance, idx, delta = item
        nonzero = delta[np.abs(delta) > 1e-10]
        canonical = delta if nonzero[0] > 0 else -delta
        # Prefer [110] over symmetry-equivalent [101], [310] over [301],
        # and positive conventional indices when choosing shell examples.
        family_score = float(np.linalg.norm(canonical - np.sort(np.abs(delta))[::-1]))
        return (round(distance, 10), round(family_score, 9) if selection == 'shells' else 0, idx, tuple(delta))

    for distance, idx, delta in sorted(candidates, key=representative_order):
        family = tuple(np.round(np.sort(np.abs(delta / distance)), 9))
        if any(abs(distance - old[0]) <= distance_tol and
               (selection == 'fixed_direction' or family == key)
               for old, key in zip(grouped, family_keys)):
            continue
        grouped.append((distance, idx, delta))
        family_keys.append(family)
    return grouped


def fcc_shell_index(lattice_vector) -> int:
    """Infinite FCC radial-shell rank, independent of finite-cell availability."""
    n2 = sum(int(v) ** 2 for v in lattice_vector)
    bound = math.isqrt(n2)
    radii = {h*h + k*k + l*l for h in range(bound + 1) for k in range(bound + 1)
             for l in range(bound + 1) if (h+k+l) % 2 == 0 and 0 < h*h+k*k+l*l <= n2}
    return len(radii)


def pair_geometry_metadata(pristine, center_index, second_index, delta, a0):
    direction, lattice = _lattice_direction(delta, a0)
    family = _cubic_family(direction)
    actual = pristine.positions[second_index] - pristine.positions[center_index]
    images = minimum_image_vectors(actual, pristine.cell, pristine.pbc)
    if not any(np.allclose(delta, v, atol=1e-7, rtol=0) for v in images):
        raise ValueError('declared pair displacement is not a minimum image of the removed sites')
    return {
        'pair_direction_family': direction_label(direction),
        'pair_direction_indices': list(direction),
        'pair_direction_family_cubic': direction_label(family).replace('[', '<').replace(']', '>'),
        'pair_distance_A': float(np.linalg.norm(delta)),
        'pair_delta_A': [float(v) for v in delta],
        'pair_distance_convention': 'initial minimum-image distance under PBC',
        'fcc_shell_index': fcc_shell_index(lattice),
        'fcc_lattice_vector_half_a0': lattice.tolist(),
        'minimum_image_degeneracy': len(images),
    }


def remove_two_atoms(atoms, first_index, second_index):
    if first_index == second_index or any(i < 0 or i >= len(atoms) for i in (first_index, second_index)):
        raise ValueError('two distinct valid vacancy indices are required')
    divacancy = atoms.copy()
    for idx in sorted((first_index, second_index), reverse=True):
        del divacancy[idx]
    return divacancy
