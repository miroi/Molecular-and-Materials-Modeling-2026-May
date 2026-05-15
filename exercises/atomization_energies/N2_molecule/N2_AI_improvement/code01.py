from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.io import write   # <-- import writer

atom = Atoms('N', calculator=EMT())
e_atom = atom.get_potential_energy()

d = 1.1
molecule = Atoms('2N', [(0., 0., 0.), (0., 0., d)], calculator=EMT())

print('\n\n Running ASE-EMT calculations on N2/N systems :')

opt = BFGS(molecule)
print('\n running geometry optimization of the N2 molecule with the initial distance d(N-N)=', d)
opt.run(fmax=0.01)

# Print out optimal internuclear distance
print('\n d(N-N)optimiz =', molecule.get_distance(0,1), ' Ang (experiment is 1.098 Ang)')

# Print final geometry
print('\n Final optimized geometry of N2:')
for i, (symbol, pos) in enumerate(zip(molecule.get_chemical_symbols(), molecule.positions)):
    print(f" Atom {i} ({symbol}): x={pos[0]:.4f} Å, y={pos[1]:.4f} Å, z={pos[2]:.4f} Å")

# Save optimized geometry to file
write("N2_optimized.xyz", molecule)   # XYZ format
write("N2_optimized.traj", molecule) # ASE trajectory format

e_molecule = molecule.get_potential_energy()
e_atomization = (2 * e_atom) - e_molecule

print('\n Nitrogen atom energy: %5.2f eV' % e_atom)
print(' Nitrogen molecule energy: %5.2f eV' % e_molecule)
print(' Atomization energy: %5.2f eV' % e_atomization, ' (experiment cca 9.76 eV) ')

