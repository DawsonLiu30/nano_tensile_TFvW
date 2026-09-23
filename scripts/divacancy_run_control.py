"""Validate a prepared run before launching any expensive calculation."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
from divacancy_geometry import parse_direction


def validate(root: Path) -> list[str]:
    root = root.resolve()
    settings = (root / 'settings_pair_scan.txt').read_text().splitlines()
    settings = [s.strip() for s in settings if s.strip()]
    if not settings or len(set(settings)) != len(settings):
        raise ValueError('Settings must be nonempty and unique')
    for setting in settings:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', setting) or '..' in setting:
            raise ValueError(f'Unsafe setting: {setting!r}')
        case = (root / 'pair_scan' / setting).resolve()
        if case.parent != (root / 'pair_scan').resolve():
            raise ValueError('Case path escaped pair_scan')
        m = json.loads((case / 'point_manifest.json').read_text(encoding='utf-8-sig'))
        if m.get('setting') != setting or m.get('scan_type') != 'pair':
            raise ValueError(f'{setting}: manifest identity mismatch')
        for key in ('spacing_A', 'fmax_eV_per_A'):
            value = float(m[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{setting}: invalid {key}')
        if int(m['pristine_n_atoms']) - int(m['vacancy_n_atoms']) != 2:
            raise ValueError(f'{setting}: expected exactly two vacancies')
        if int(m['relax_steps']) <= 0:
            raise ValueError(f'{setting}: invalid relax_steps')
        # Overrides must not silently change a prepared scientific protocol.
        for env, key in (('KEDF_X', 'kedf_x'), ('KEDF_Y', 'kedf_y'),
                         ('FMAX', 'fmax_eV_per_A'), ('RELAX_STEPS', 'relax_steps')):
            if env in os.environ and not math.isclose(float(os.environ[env]), float(m[key]), abs_tol=1e-12, rel_tol=0):
                raise ValueError(f'{setting}: {env} disagrees with prepared {key}; prepare a new run')
        if 'DIRECTION' in os.environ:
            expected = list(parse_direction(os.environ['DIRECTION']))
            actual = m.get('pair_direction_indices')
            if m.get('pair_selection') != 'fixed_direction' or actual != expected:
                raise ValueError(f'{setting}: DIRECTION disagrees with manifest; prepare a new run')
        if 'A0_START' in os.environ:
            value = m.get('a0_A', m.get('a0_start_A', m.get('a0')))
            if value is None or not math.isclose(float(os.environ['A0_START']), float(value), abs_tol=1e-10):
                raise ValueError(f'{setting}: A0_START cannot be matched to manifest; prepare a new run')
    return settings


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    for setting in validate(args.root):
        print(setting)
