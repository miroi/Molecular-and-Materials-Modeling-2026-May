from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.optimize import BFGS

# --- 1. SET UP CALCULATIONS ---
# Isolated Oxygen atom (Reference)
adsorbate = Atoms('O')
adsorbate.calc = EMT()
e_o = adsorbate.get_potential_energy()

# Clean Gold surface (Reference)
slab = fcc111('Au', size=(6, 6, 3), vacuum=12.0)
slab.calc = EMT()

# --- 2. RELAX CLEAN SLAB ---
dyn_slab = BFGS(slab, trajectory='clean_slab.traj', logfile=None)
dyn_slab.run(fmax=0.05)
e_slab = slab.get_potential_energy()

# --- 3. ADSORPTION SYSTEM ---
# Place O atom at fcc site (literature stable site)
add_adsorbate(slab, 'O', height=1.5, position='fcc')

# --- 4. RELAX ADSORPTION SYSTEM ---
dyn_ads = BFGS(slab, trajectory='adsorption.traj', logfile=None)
dyn_ads.run(fmax=0.05)

# --- 5. RESULTS & COMPARISON ---
e_total = slab.get_potential_energy()  # energy O@Au(111)
e_bind_calc = e_total - (e_slab + e_o) # adsorption energy

# Published data for O/Au(111) at fcc site:
lit_val = -2.77  # RPBE reference

print(f"--- ASE ADSORPTION REPORT ---")
print(f"Calculated Binding Energy (EMT): {e_bind_calc:8.3f} eV")
print(f"Literature Value (DFT)     :     {lit_val:8.3f} eV")
print(f"Difference (vs RPBE):           {abs(e_bind_calc - lit_val):8.3f} eV")
print("-" * 29)

if e_bind_calc < 0:
    print("Status: Adsorption is EXOTHERMIC (stable).")
else:
    print("Status: Adsorption is ENDOTHERMIC (unstable).")

