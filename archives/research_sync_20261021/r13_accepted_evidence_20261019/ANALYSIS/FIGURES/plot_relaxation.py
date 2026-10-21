from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
a=parser.parse_args(); root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
frames=read(root/"relax.traj",index=":")
e=np.array([x.get_potential_energy() for x in frames])
f=np.array([np.linalg.norm(x.get_forces(),axis=1).max() for x in frames])
lz=np.array([x.cell[2,2] for x in frames]);steps=np.arange(len(frames))
fig,ax=plt.subplots(3,1,figsize=(7.4,7.2),sharex=True,constrained_layout=True)
ax[0].plot(steps,e-e[-1],"o-",ms=3,lw=1.4,color="#2166ac");ax[0].set_ylabel("Energy above final (eV)")
ax[1].semilogy(steps,f,"o-",ms=3,lw=1.4,color="#b35806");ax[1].axhline(.001,color="#32744c",ls="--",label="Acceptance: 0.001 eV / Å");ax[1].set_ylabel("Maximum force (eV / Å)");ax[1].legend(fontsize=9)
ax[2].plot(steps,lz,"o-",ms=3,lw=1.4,color="#636363");ax[2].set_ylabel("Axial length (Å)");ax[2].set_xlabel("Completed ionic step (initial = 0)")
for x in ax:x.grid(alpha=.2)
fig.suptitle("R13 pristine column: BFGS line-search continuation\nDFTpy, job 2255218; completed steps only",fontsize=13)
fig.savefig(out/"R13_convergence_2255218.png",dpi=180);plt.close(fig)
(out/"R13_convergence_plot_data.json").write_text(json.dumps({"source":"relax.traj","steps":steps.tolist(),"energy_eV":e.tolist(),"atomic_fmax_eV_A":f.tolist(),"Lz_A":lz.tolist()},indent=2))
print(json.dumps({"frames":len(frames),"energy_drop_eV":float(e[0]-e[-1]),"final_force":float(f[-1]),"figure":str(out/"R13_convergence_2255218.png")}))
