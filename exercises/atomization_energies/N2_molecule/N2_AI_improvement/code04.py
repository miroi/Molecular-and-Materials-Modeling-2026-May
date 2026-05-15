# Extended ASE-EMT workflow for N2 potential curve
# By Miroslav

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write
import numpy as np
import matplotlib.pyplot as plt

# Single atom reference
atom = Atoms('N', calculator=EMT())
e_atom = atom.get_potential_energy()

# Initial N2 molecule
d = 1.1  # experimental bond length ~1.098 Å
molecule = Atoms('2N', [(0., 0., 0.), (0., 0., d)], calculator=EMT())

print('\n\n Running ASE-EMT calculations on N2/N systems :')

# Geometry optimization with trajectory recording
opt = BFGS(molecule, trajectory='N2_optimization.traj')
print('\n running geometry optimization of the N2 molecule with the initial distance d(N-N)=', d)
opt.run(fmax=0.01)

# Print optimized bond length
print('\n d(N-N)optimiz =', molecule.get_distance(0, 1), ' Å (experiment is 1.098 Å)')

# Print final geometry
print('\n Final optimized geometry of N2:')
for i, (symbol, pos) in enumerate(zip(molecule.get_chemical_symbols(), molecule.positions)):
    print(f" Atom {i} ({symbol}): x={pos[0]:.4f} Å, y={pos[1]:.4f} Å, z={pos[2]:.4f} Å")

# Save optimized geometry
write("N2_optimized.xyz", molecule)
write("N2_optimized.traj", molecule)

# Energies
e_molecule = molecule.get_potential_energy()
e_atomization = (2 * e_atom) - e_molecule

print('\n Nitrogen atom energy: %5.2f eV' % e_atom)
print(' Nitrogen molecule energy: %5.2f eV' % e_molecule)
print(' Atomization energy: %5.2f eV' % e_atomization, ' (experiment ~9.76 eV)')

# --- Potential energy curve ---
print("\n Calculating potential energy curve for N2...")

# Extended range for bond distances
distances = np.linspace(0.8, 3.0, 50)  # Å
energies = []

for d in distances:
    mol = Atoms('2N', [(0., 0., 0.), (0., 0., d)], calculator=EMT())
    energies.append(mol.get_potential_energy())

energies = np.array(energies)
energies_norm = energies - energies.min()  # normalized relative to minimum

# Save curve data to text file
with open("N2_potential_curve.txt", "w") as f:
    f.write("# N2 potential energy curve (ASE-EMT)\n")
    f.write("# Distance (Å)   Energy (eV)   Normalized (eV)\n")
    for d, e, en in zip(distances, energies, energies_norm):
        f.write(f"{d:.4f}   {e:.6f}   {en:.6f}\n")

# Plot absolute energies
plt.figure(figsize=(6,4))
plt.plot(distances, energies, 'o-', label='ASE-EMT (absolute)')
plt.axvline(1.098, color='r', linestyle='--', label='Experimental bond length')
plt.xlabel('N–N distance (Å)')
plt.ylabel('Energy (eV)')
plt.title('Potential Energy Curve of N2 (EMT)')
plt.legend()
plt.tight_layout()
plt.savefig("N2_potential_curve_absolute.png", dpi=150)

# Plot normalized energies
plt.figure(figsize=(6,4))
plt.plot(distances, energies_norm, 'o-', label='ASE-EMT (normalized)')
plt.axvline(1.098, color='r', linestyle='--', label='Experimental bond length')
plt.xlabel('N–N distance (Å)')
plt.ylabel('Energy relative to minimum (eV)')
plt.title('Normalized Potential Energy Curve of N2 (EMT)')
plt.legend()
plt.tight_layout()
plt.savefig("N2_potential_curve_normalized.png", dpi=150)

plt.show()

