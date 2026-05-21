# Structure Audit After VESTA Issue

Date: 2026-05-21

## Immediate Conclusion

The latest pulled QE/DFTpy convergence data were re-audited separately from old
proposal/legacy structure folders.

Result:

- Latest professor pull: PASS
- Old local `finite_grip_*` / proposal-preview structures: FAIL / quarantine

The current QE/DFTpy bulk and vacancy convergence structures do not show atom
count errors, short Al-Al contacts, suspicious vacancy-cell atom counts, or
large axial vacuum gaps in the programmatic audit.

The structural problem is concentrated in legacy finite-name/proposal files that
contain large z-direction empty intervals of about 14-20 A. These files conflict
with the corrected thesis definition that nanocolumns and nanocrystals are
axially periodic, infinitely extended structures.

## Latest Professor Pull Audit

Root checked:

`C:\Users\dawso\Desktop\latest_professor_pull_20260511`

Command:

```powershell
python scripts\audit_qe_dftpy_structures.py --root "C:\Users\dawso\Desktop\latest_professor_pull_20260511" --outdir outputs\structure_audit_latest_pull_20260521 --max-vesta-files 120
```

Summary:

| Work item | Files | Read failures | Geometry/count flags |
|---|---:|---:|---:|
| QE bulk B/EOS | 171 | 0 | 0 |
| QE vacancy convergence | 28 | 0 | 0 |
| DFTpy vacancy spacing, fixed QE-a0 | 46 | 0 | 0 |
| DFTpy vacancy spacing, DFTpy own-a0 | 46 | 0 | 0 |
| DFTpy primitive-size fixed QE-a0 | 36 | 0 | 0 |

Total latest-pull files scanned: 327

Total latest-pull flagged files: 0

Latest-pull audit outputs:

- `outputs\structure_audit_latest_pull_20260521\STRUCTURE_AUDIT_REPORT.md`
- `outputs\structure_audit_latest_pull_20260521\structure_audit_all_files.csv`
- `outputs\structure_audit_latest_pull_20260521\structure_audit_flagged_files.csv`

## Full Local Repo Audit

Roots checked:

- `C:\Users\dawso\Desktop\latest_professor_pull_20260511`
- `C:\Users\dawso\nano_tensile_TFvW\cases`
- `C:\Users\dawso\nano_tensile_TFvW\results\professor_review`

Command:

```powershell
python scripts\audit_qe_dftpy_structures.py --outdir outputs\structure_audit_20260521 --max-vesta-files 160
```

Summary:

| Work item | Files | Read failures | Geometry/count flags |
|---|---:|---:|---:|
| QE bulk B/EOS | 171 | 0 | 0 |
| QE vacancy convergence | 28 | 0 | 0 |
| DFTpy vacancy spacing, fixed QE-a0 | 46 | 0 | 0 |
| DFTpy vacancy spacing, DFTpy own-a0 | 46 | 0 | 0 |
| DFTpy primitive-size fixed QE-a0 | 36 | 0 | 0 |
| Periodic nanowire/nanoprism | 227 | 0 | 0 |
| Legacy finite-name folder | 489 | 0 | 489 |
| Other old/proposal review files | 36 | 0 | 30 |

Total local files scanned: 1079

Total local flagged files: 519

All 519 flags are:

`LARGE_Z_EMPTY_INTERVAL_GT_5A`

No current QE/DFTpy convergence work item is flagged.

## Quarantined Files

Do not use the following old structure families as formal thesis structures:

- `cases\finite_grip_111_1.0nm_vacancy_tfvw\...`
- `results\professor_review\chapter_4_models\generated_r4_orientation_models\...`
- `results\professor_review\ppt_ready\data\...`

Representative failures:

| File family | Typical issue |
|---|---|
| `finite_grip_111_1.0nm_vacancy_tfvw\inputs\*.vasp` | z empty interval about 20 A |
| `finite_grip_111_1.0nm_vacancy_tfvw\results\cycle_*.xyz` | z empty interval about 14-20 A |
| `chapter_4_models\generated_r4_orientation_models\*.vasp` | finite/vacuum-z proposal geometry |

These are legacy/proposal structures and are not consistent with the corrected
definition:

Nanocolumns and nanocrystals are both axially periodic infinite structures. The
difference is the lateral cross-section shape, not finite versus infinite axial
length.

## Representative VESTA Files

VESTA is installed at:

`C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe`

Open a current safe DFTpy structure:

```powershell
Start-Process "C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe" "C:\Users\dawso\Desktop\latest_professor_pull_20260511\dftpy_vacancy_size_primitive_qe_a0_spacing020_20260511\size_scan\prim_04x04x04\vacancy_start.vasp"
```

Open a current safe QE vacancy structure:

```powershell
Start-Process "C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe" "C:\Users\dawso\Desktop\latest_professor_pull_20260511\qe_vacancy_convergence_20260506\vacancy_start.vasp"
```

Open a quarantined legacy failure for comparison:

```powershell
Start-Process "C:\Users\dawso\AppData\Local\Microsoft\WinGet\Packages\KoichiMomma.VESTA_Microsoft.Winget.Source_8wekyb3d8bbwe\VESTA-win64\VESTA.exe" "C:\Users\dawso\nano_tensile_TFvW\cases\finite_grip_111_1.0nm_vacancy_tfvw\inputs\vacancy_start.vasp"
```

## Action Rule From Now On

Before showing any structure to the professor:

1. Use only the latest-pull QE/DFTpy convergence folders or the verified
   periodic nanowire/nanoprism folders.
2. Do not show any file from `finite_grip_*` or old proposal-preview folders.
3. Run `scripts\audit_qe_dftpy_structures.py` on the folder.
4. Open at least one pristine and one vacancy structure in VESTA for visual
   inspection.
5. If a structure is meant to be a nanocolumn/nanocrystal, verify there is no
   large vacuum gap along z.

