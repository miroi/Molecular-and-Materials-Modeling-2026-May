import numpy as np
from ase import Atoms
from ase.calculators.lj import Lennard_Jones
from ase.optimize.basinhopping import BasinHopping

# 1. Setup: Create 7 Argon atoms in a random starting configuration
n_atoms = 7
positions = np.random.rand(n_atoms, 3) * 5.0 
atoms = Atoms('Ar' * n_atoms, positions=positions)

# 2. Assign Calculator: Use the built-in Lennard-Jones potential
atoms.calc = Lennard_Jones()

# 3. Initialize Global Optimizer:
# 'temperature' (in eV) controls the probability of accepting higher-energy steps
# 'dr' is the maximum distance an atom can be moved during a jump
bh = BasinHopping(atoms, 
                  temperature=0.05, 
                  dr=0.5, 
                  optimizer_logfile=None)

# 4. Execute the search:
# For Ar7, the global minimum is a pentagonal bipyramid (~ -16.5 eV)
bh.run(steps=50)

# 5. Results
print(f"Lowest Energy Found: {atoms.get_potential_energy():.4f} eV")
print("Atomic Positions:\n", atoms.get_positions())

