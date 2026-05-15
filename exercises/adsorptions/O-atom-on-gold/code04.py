from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write
import numpy as np

# --- 1. ISOLATED ADSORBATE ---
adsorbate = Atoms('O')
adsorbate.calc = EMT()
e_o = adsorbate.get_potential_energy()

# --- 2. CLEAN SLAB ---
slab = fcc111('Au', size=(6, 6, 3), vacuum=12.0)
slab.calc = EMT()

# Relax clean slab
dyn_slab = BFGS(slab, trajectory='clean_slab.traj', logfile=None)
dyn_slab.run(fmax=0.05)
e_slab = slab.get_potential_energy()

# Save relaxed clean slab
write('Au111_clean.vasp', slab, format='vasp', direct=True, sort=True)

# --- 3. ADSORPTION SYSTEM ---
add_adsorbate(slab, 'O', height=1.5, position='fcc')

# Relax adsorption system
dyn_ads = BFGS(slab, trajectory='adsorption.traj', logfile=None)
dyn_ads.run(fmax=0.05)

# --- 4. RESULTS ---
e_total = slab.get_potential_energy()
e_bind_calc = e_total - (e_slab + e_o)

# Literature reference (RPBE)
lit_val = -2.77

# --- 5. GEOMETRY ANALYSIS ---
o_index = len(slab) - 1
o_z = slab[o_index].position[2]

au_indices = [atom.index for atom in slab if atom.symbol == 'Au']
au_z_coords = [slab[i].position[2] for i in au_indices]
tol = 0.1
top_layer_atoms = [z for z in au_z_coords if abs(z - max(au_z_coords)) < tol]
top_layer_avg_z = np.mean(top_layer_atoms)
o_height = o_z - top_layer_avg_z

# --- 6. SAVE ADSORPTION SYSTEM ---
write('O_on_Au111.vasp', slab, format='vasp', direct=True, sort=True)

# --- 7. REPORT ---
print(f"--- ASE ADSORPTION REPORT ---")
print(f"Calculated Binding Energy (EMT): {e_bind_calc:8.3f} eV")
print(f"Literature Value (DFT)     :     {lit_val:8.3f} eV")
print(f"Difference (vs RPBE):           {abs(e_bind_calc - lit_val):8.3f} eV")
print(f"Relaxed O height above Au(111): {o_height:8.3f} Å")
print("-" * 29)

if e_bind_calc < 0:
    print("Status: Adsorption is EXOTHERMIC (stable).")
else:
    print("Status: Adsorption is ENDOTHERMIC (unstable).")

