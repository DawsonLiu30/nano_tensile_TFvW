set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/dawson666/miniconda3/envs/dftpy-env/bin/python -B - <<'PY'
import importlib,inspect,hashlib,tarfile,sys,io,json
from pathlib import Path
mods=['ase.optimize.bfgslinesearch','ase.optimize.optimize','ase.filters','ase.utils.linesearch','dftpy.utils']
manifest=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in mods:
  m=importlib.import_module(name);p=Path(inspect.getsourcefile(m));data=p.read_bytes();arc=name+'.py'
  manifest.append({'module':name,'source':str(p),'archive':arc,'sha256':hashlib.sha256(data).hexdigest()})
  info=tarfile.TarInfo(arc);info.size=len(data);t.addfile(info,io.BytesIO(data))
 data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
PY
