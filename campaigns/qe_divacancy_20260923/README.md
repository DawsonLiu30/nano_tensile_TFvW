# QE divacancy transfer snapshot

Inputs and status audited on 2026-09-23. Portable tooling prepared on 2026-09-24.
Run instructions: [cross-device guide](../../docs/CROSS_DEVICE_QE.md).

| Case | Source status | Transfer action |
|---|---|---|
| `2V_1NN_D110` | SCF converged, −444.04662686 Ry | Preserve verified input/output/result reference |
| `2V_2NN_D100` | Interrupted, no converged energy | Fresh SCF on destination |
| `2V_D310_r1` | Prepared, not started | Fresh SCF on destination |
| `2V_D110_r2` | Prepared, not started | Fresh SCF on destination |

All four inputs contain the exact corresponding TFvW-relaxed coordinates.
No ionic or cell relaxation is requested. The shared protocol is 106 Al atoms,
LDA, Al.pz-vbc.UPF, 60/240 Ry, 3×3×3 unshifted k points, MV 0.02 Ry,
200 bands, nosym=true, noinv=false, conv_thr=1e-9 Ry.

`inputs/` contains fresh-SCF templates with portable paths. `provenance/`
preserves the original source inputs and source manifests. Historical absolute
paths inside provenance/evidence are records, not active destinations.
`evidence/2V_1NN_D110` contains the actual converged output and its input/result.
The interrupted 2NN output is retained explicitly as incomplete evidence.
`pseudo/` contains the exact 30,963-byte UPF from the
[official QE download](https://pseudopotentials.quantum-espresso.org/upf_files/Al.pz-vbc.UPF).
Its SHA-256 is `4eab06b63f87f07ede2d5a193e6d993a09107167fd6b8647afa807342501d6e5`.

`manifest.json` selects two comparisons: E(1NN)−E(2NN), and
E([310])−E(middle [110]). The latter differs in distance as well as direction.
The collector verifies archived 1NN evidence before using it. New complete
attempts take precedence. A pair remains blank until both energies pass checks
and the printed QE versions match. Matching versions still do not establish
identical compiler/MPI numerical behaviour; reproduce a reference endpoint when
changing builds and test convergence of the meV-scale differences.

The source reference converged using QE 7.5. Its last SCF error was 3.9e-10 Ry.
The latest interrupted 2NN binary scratch is not included: its restart records
are incomplete. All source scratch remains on the original computer.

`comparison/TFvW_controlled_energies.csv` is a separate completed DFTpy dataset,
including 108/256-site formation and binding energies. It is not QE output and
does not validate the August variable-cell numbers. Its distances label the
initial removed sites. Do not subtract absolute QE and DFTpy energies.

SHA256SUMS.json protects exact snapshot bytes. Git line-ending conversion is
disabled for this campaign so clones retain those identities. Changes to
scientific settings need a new named campaign and new evidence.
