import torch
import torchani
from ase import Atoms
from ase.optimize import BFGS
from ase.vibrations import Vibrations
import os

# 1. Setup H2S with slightly more realistic initial positions
atoms = Atoms('SH2', positions=[[0.0, 0.0, 0.0], 
                                [0.0, 0.9, 1.0], 
                                [0.0, -0.9, 1.0]])
calculator = torchani.models.ANI2x().ase()
atoms.calc = calculator # Updated syntax to avoid FutureWarning

# 2. High-precision Optimization
opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.0001) # Even tighter for frequency stability

# 3. Frequency Calculation
vib = Vibrations(atoms, name='h2s_vib')
vib.run()
freqs = vib.get_frequencies() 

# 4. Filter and Sort
# For a non-linear 3-atom molecule, there are exactly 3 vibrational modes.
# We take the 3 highest frequencies.
real_vibs = sorted([f.real for f in freqs])[-3:] 

# 5. Comparison Data
exp_freqs = [1183, 2615, 2626] # v2, v1, v3
mode_names = ["Bending (v2)", "Symm. Stretch (v1)", "Asymm. Stretch (v3)"]

print("-" * 55)
print(f"{'Mode':<20} | {'ANI-2x (cm^-1)':<15} | {'Exp. (cm^-1)'}")
print("-" * 55)

for i in range(3):
    print(f"{mode_names[i]:<20} | {real_vibs[i]:<15.1f} | {exp_freqs[i]}")

print("-" * 55)

# Clean up
vib.clean()
if os.path.exists('h2s_vib'):
    import shutil
    shutil.rmtree('h2s_vib')

