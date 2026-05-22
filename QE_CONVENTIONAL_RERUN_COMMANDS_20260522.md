# QE conventional vacancy rerun commands

This note records the commands for the 2026-05-22 QE conventional 2x2x4
vacancy rerun.

## Remote work item

```text
/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522
```

This rerun uses a conventional orthorhombic fcc 2x2x4 cell:

- pristine atoms: 64
- vacancy atoms: 63
- cell angles: 90, 90, 90 degrees
- central vacancy at fractional coordinate (0.5, 0.5, 0.5)
- QE relaxation force threshold: 0.002 eV/A
- cutoff scan includes the professor-requested 800 eV point

## Submit all cases on ct56

Use this if ct56 submission capacity is available:

```bash
cd /gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522 && for d in ecut_scan/ecut_0300eV ecut_scan/ecut_0400eV ecut_scan/ecut_0500eV ecut_scan/ecut_0600eV ecut_scan/ecut_0800eV kmesh_scan/k_01x01x01 kmesh_scan/k_02x02x02 kmesh_scan/k_03x03x03 kmesh_scan/k_04x04x04 dense_k05_ecut_scan/ecut_0400eV_k05 dense_k05_ecut_scan/ecut_0500eV_k05 dense_k05_ecut_scan/ecut_0600eV_k05 dense_k05_ecut_scan/ecut_0800eV_k05 kmesh_scan/k_05x05x05 kmesh_scan/k_06x06x06; do echo "[SUBMIT] $d"; (cd "$d" && sbatch -p ct56 group_job.sh); done
```

Check queue:

```bash
squeue -u dawson666
```

## Pull results after the jobs finish

Run this from WSL on the local machine:

```bash
cd /mnt/c/Users/dawso/nano_tensile_TFvW && bash scripts/pull_qe_vacancy_conventional_results.sh
```

The script will:

- rsync the full work item back to
  `/mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522`
- exclude QE `tmp` folders but keep inputs, outputs, logs, scripts, manifests,
  CSV files, and figures
- run `scripts/collect_all_qe_vacancy_recursive.py`
- report completed and pending cases
- create
  `/mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522_no_tmp.zip`

## Manual one-line pull command

If the helper script is not available, use:

```bash
rsync -avhP --exclude '*/tmp/***' iservice:/gpfs-work/dawson666/qe_cases/qe_runs/qe_vacancy_conventional_2x2x4_20260522/ /mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522/
```

Then collect locally:

```bash
python /mnt/c/Users/dawso/nano_tensile_TFvW/scripts/collect_all_qe_vacancy_recursive.py --rootdir /mnt/c/Users/dawso/Desktop/qe_conventional_pull_20260522/qe_vacancy_conventional_2x2x4_20260522
```
