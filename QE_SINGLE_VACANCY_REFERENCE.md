# QE single-vacancy reference

## Primary corrected reference

The primary KSDFT comparison uses the professor-requested corrected workflow:

- conventional cubic fcc `3x3x3` supercell
- pristine/vacancy atoms: `108 -> 107`
- centered vacancy, concentration `0.925926%`
- QE/PBE with `Al_PAW_PBE.UPF`
- `calculation = 'vc-relax'` for both structures
- cutoff `800 eV`
- densest completed k mesh: `5x5x5`
- vacancy formation energy: `0.6389122264 eV`

The completed dense-k values retain visible metallic k-point sensitivity:

| k mesh | Vacancy formation energy (eV) |
|---|---:|
| `3x3x3` | `0.6449419043` |
| `4x4x4` | `0.6778746999` |
| `5x5x5` | `0.6389122264` |

Report the `5x5x5` value as the best completed dense-k point, together with the
`0.638912-0.677875 eV` dense-k range. Do not call this a perfectly locked
single-value k-point limit.

## Historical value

The old `0.6011-0.6012 eV` value belongs to a different conventional `2x2x4`
cell (`64 -> 63` atoms, `5x5x5`, `600-800 eV`). It remains useful as historical
provenance but must not be labeled as the corrected `3x3x3` reference.

The previous hard-coded target `0.601167 eV` was inherited from that historical
dataset. DFTpy scan energies do not depend on this target; completed scans can
be re-ranked against the corrected reference without rerunning DFTpy.
