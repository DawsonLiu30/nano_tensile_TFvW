R13 0V axial line-search continuation, 2026-10-18

This revision continues the completed ionic step 12 of job 2254539,
attempt_004_axial_linesearch. That bounded 6600-s trial stopped during the next
line search. Only the atomic geometry and density in the completed step-12
bundle are reused; lower-energy unfinished trials are not restart checkpoints.
The parent checkpoint SHA256 is d30d37f9c33e7ef7be05a5ac653232f348774304636eb6691163f5b508f5297f.
Parent result, provenance, axial source manifest and all source files are pinned.

One new attempt only: attempt_005_axial_linesearch_continue.
ct56, 1 task / 1 CPU, 24 GiB, Slurm 6 h; internal 21000 s,
at most 120 completed ionic steps and 360 logical electronic calls.
No automatic further continuation and no scheduled monitoring.

Reused unchanged: scientific recovery_base.py, electronic_trace.py, API smoke
and the six actual-runtime source records. TN, econv=1e-11 Ha/atom, ncheck=3,
one HEG retry only after TN native flag 1; LDA/local TF=.9/vW=.1,
140x140x96 grid, Ne=228, full PBC, z-only FrechetCellFilter all retained.
Atomic and filter Fmax <= .001 eV/Angstrom; |sigma_zz| <= .05 GPa.
Original stress and R13 TN diagnostic gates are reverified with their scopes.
They do not certify numerical convergence for all new geometries.

Ionic method remains native ASE BFGSLineSearch with c1=.23,c2=.46,alpha=10,
stpmax=50. The line-search increment parameter maxstep changes .005 -> .02.
This is not an outer-step atomic-displacement cap. Every completed line search
must return CONVERGENCE, not WARNING or no_update-only fallback. The existing
repeated-cycle guard is retained. The preceding 12 steps reduced energy but
force did not reach .001: this is a bounded attempt to finish, not an accepted
physical result. This revision's speed or convergence is not yet demonstrated.

Fresh inverse Hessian: the parent did not save full optimizer history.
No old Hessian is loaded or reconstructed. After every complete core checkpoint,
optimizer_state.json is atomically written with the exact bundle SHA, step,
geometry, density bundle reference, filter orig_cell/mask/exp_cell_factor,
current generalized positions and H/r0/g0/p/e0/alpha_k/no_update and policy.
g0 is minus generalized force divided by alpha. A crash between the core bundle
and state write can leave an older state; future recovery must reject a hash,
step or geometry mismatch and explicitly start with fresh H. This revision has
no automatic optimizer-state loader. The state is retained for future review.

Checks: synthetic energy/force/filter tests plus saved-state coherence checks,
actual-runtime API smoke (no DFT), and read-only checkpoint/source preflight.
Program tests do not certify a new relaxed Al endpoint. Submit only after root
review and archive the actual scheduler receipt. No new QE calculations here.

Synthetic QA: the inherited farther harmonic-cell target (Lz=13.45) with
maxstep=.02 reaches native WARNING: STP = maxstep. A regression test retains
this as a rejected bounded stop. A nearer target (13.41) checks successful
axial convergence and checkpoint-state matching. No native line-search warning
is converted into acceptance. The physical continuation can likewise stop if
native Wolfe convergence is not established.
