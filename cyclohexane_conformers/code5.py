import numpy as np
import warnings
from ase import Atoms
from ase.optimize import BFGS
from ase.io import write
from chgnet.model.dynamics import CHGNetCalculator

# Suppress PyTorch/CHGNet warnings
warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")

def get_base_cyclohexane():
    """Returns a chair cyclohexane in a 15A vacuum cell."""
    coords = [
        [0.000, 1.454, 0.228],   [1.259, 0.727, -0.228],
        [1.259, -0.727, 0.228],  [0.000, -1.454, -0.228],
        [-1.259, -0.727, 0.228], [-1.259, 0.727, -0.228],
        [0.000, 2.499, -0.108],  [0.000, 1.454, 1.328],
        [2.164, 1.250, 0.155],   [1.259, 0.727, -1.328],
        [2.164, -1.250, -0.155], [1.259, -0.727, 1.328],
        [0.000, -2.499, 0.108],  [0.000, -1.454, -1.328],
        [-2.164, -1.250, 0.155], [-1.259, -0.727, 1.328],
        [-2.164, 1.250, -0.155], [-1.259, 0.727, -1.328]
    ]
    mol = Atoms('C6H12', positions=coords)
    mol.set_cell([15, 15, 15])
    mol.center()
    return mol

def analyze(atoms, name):
    """Calculates energy and ring dihedrals."""
    # This list ensures we can loop through the ring (C0-C1-C2-C3... back to C2)
    idx = [0, 1, 2, 3, 4, 5, 0, 1, 2] 
    dihedrals = []
    for i in range(6):
        d = atoms.get_dihedral(*idx[i:i+4])
        if d > 180: d -= 360
        dihedrals.append(round(d, 1))
    
    energy = atoms.get_potential_energy()
    print(f"{name:12} | Energy: {energy:.4f} eV | Dihedrals: {dihedrals}")
    return energy

def run_conformers():
    calc = CHGNetCalculator(use_device='cpu')
    to_kcal = 23.061

    # 1. CHAIR
    chair = get_base_cyclohexane()
    chair.calc = calc
    print("Optimizing Chair...")
    BFGS(chair, logfile=None).run(fmax=0.01)
    e_chair = analyze(chair, "Chair")
    write('chair.xyz', chair)

    # 2. PURE BOAT (Transition State)
    # We flip Carbon 3 and its hydrogens (12, 13) up and keep it symmetric
    pure_boat = chair.copy()
    pos = pure_boat.get_positions()
    pos[[3, 12, 13], 2] += 2.2 
    pure_boat.set_positions(pos)
    pure_boat.calc = calc
    # No optimization here to stay at the high-energy point
    e_pure = analyze(pure_boat, "Pure Boat")
    write('pure_boat.xyz', pure_boat)

    # 3. TWIST-BOAT (Local Minimum)
    twist_boat = pure_boat.copy()
    t_pos = twist_boat.get_positions()
    # Break symmetry by twisting C1 and C4 slightly in opposite directions
    t_pos[1, 0] += 0.3 
    t_pos[4, 0] -= 0.3 
    twist_boat.set_positions(t_pos)
    twist_boat.calc = calc
    print("\nOptimizing Twist-Boat...")
    BFGS(twist_boat, logfile=None).run(fmax=0.01)
    e_twist = analyze(twist_boat, "Twist-Boat")
    write('twist_boat.xyz', twist_boat)

    # Summary
    print("\n" + "="*45)
    print("RESULTS (kcal/mol relative to Chair)")
    print(f"Twist-Boat: { (e_twist - e_chair)*to_kcal :.2f} (Exp: ~5.5)")
    print(f"Pure Boat:  { (e_pure - e_chair)*to_kcal :.2f} (Exp: ~7.0)")
    print("="*45)

if __name__ == "__main__":
    run_conformers()

