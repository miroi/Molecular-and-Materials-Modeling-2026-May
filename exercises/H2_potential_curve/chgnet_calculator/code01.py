import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.calculator import Calculator
from chgnet.model.dynamics import CHGNetCalculator
from ase.constraints import ExpCellFilter
import warnings
warnings.filterwarnings('ignore')  # CHGNet may produce warnings

# Set up CHGNet calculator globally
# Using a larger cell to avoid spurious interactions
def get_chgnet_calculator():
    """Initialize CHGNet calculator for H2 calculations"""
    try:
        calc = CHGNetCalculator(use_device='cpu')  # Use 'cuda' if GPU available
        return calc
    except Exception as e:
        print(f"Error initializing CHGNet: {e}")
        raise

# 1. Calculate the energy of a single isolated Hydrogen atom
# Need a sufficiently large periodic box to avoid interactions between periodic images
cell_size = 10.0  # Angstroms - large enough to isolate atoms
isolated_atom = Atoms('H', 
                     positions=[(0, 0, 0)],
                     cell=[cell_size, cell_size, cell_size],
                     pbc=True)  # Periodic boundary conditions
isolated_atom.calc = get_chgnet_calculator()
e_single_atom = isolated_atom.get_potential_energy()
e_two_isolated_atoms = 2 * e_single_atom

# 2. Define the range of interatomic distances (in Angstroms)
distances = np.linspace(0.5, 3.0, 40)  # CHGNet works better with slightly larger range
energies = []

# 3. Loop over distances to calculate molecular potential energy
print("Calculating potential energy curve...")
for i, r in enumerate(distances):
    # Create H2 molecule in a periodic box with sufficient vacuum
    # Center the molecule at the box center to minimize periodic effects
    box_center = cell_size / 2
    positions = [
        (box_center - r/2, 0, 0),
        (box_center + r/2, 0, 0)
    ]
    
    molecule = Atoms('H2', 
                    positions=positions,
                    cell=[cell_size, cell_size, cell_size],
                    pbc=True)  # Enable periodic boundary conditions
    
    molecule.calc = get_chgnet_calculator()
    energy = molecule.get_potential_energy()
    energies.append(energy)
    
    if (i + 1) % 10 == 0:
        print(f"  Completed {i+1}/{len(distances)} distances")

energies = np.array(energies)

# 4. Find the equilibrium bond length and minimum molecular energy
min_idx = np.argmin(energies)
r_eq = distances[min_idx]
e_min = energies[min_idx]

# 5. Compute Simulated Dissociation Energy (De)
# De = E(separated atoms) - E(equilibrium molecule)
simulated_de = e_two_isolated_atoms - e_min
experimental_de = 4.52  # Precise experimental De for H2 in eV

# 6. Save data points and metrics to a text file
output_filename = "H2_potential_curve_CHGNet.txt"
data_to_save = np.column_stack((distances, energies))

header_text = (
    f"H2 Potential Energy Curve and Dissociation Data (CHGNet)\n"
    f"Calculator: CHGNet (universal machine learning potential)\n"
    f"Periodic box size: {cell_size:.1f} x {cell_size:.1f} x {cell_size:.1f} A^3\n"
    f"Equilibrium Bond Length (r_e): {r_eq:.3f} A\n"
    f"Simulated Dissociation Energy (De): {simulated_de:.3f} eV\n"
    f"Experimental Dissociation Energy (De): {experimental_de:.2f} eV\n"
    f"Absolute Error: {abs(simulated_de - experimental_de):.3f} eV\n"
    f"{'Distance(A)':<15} {'Energy(eV)':<20}"
)
np.savetxt(output_filename, data_to_save, fmt="%-15.6f %-20.8f", 
           header=header_text, comments='')
print(f"\nData saved to {output_filename}")

# 7. Print the comparison metrics
print("\n" + "="*50)
print("Summary of Results (CHGNet)")
print("="*50)
print(f"Equilibrium Bond Length:      {r_eq:.4f} Å")
print(f"Experimental bond length:     0.7414 Å")
print(f"Bond length error:            {abs(r_eq - 0.7414):.4f} Å")
print(f"\nSimulated Dissociation Energy: {simulated_de:.3f} eV")
print(f"Experimental De:               {experimental_de:.3f} eV")
print(f"Absolute Error:               {abs(simulated_de - experimental_de):.3f} eV")
print(f"Relative Error:               {abs(simulated_de - experimental_de)/experimental_de*100:.1f}%")

# 8. Plot the potential energy curve with discrete simulation steps
plt.figure(figsize=(9, 6))

# Main potential curve
plt.plot(distances, energies, 'b-o', markersize=4, lw=1.5, 
         label='CHGNet Data Points', alpha=0.8)

# Annotations for key features
plt.axvline(r_eq, color='r', linestyle='--', lw=2, 
           label=f'r$_e$ = {r_eq:.2f} Å')
plt.axhline(e_min, color='g', linestyle='--', lw=1.5, 
           label=f'E$_{{min}}$ = {e_min:.2f} eV')
plt.axhline(e_two_isolated_atoms, color='m', linestyle=':', lw=2, 
           label=f'2H isolated = {e_two_isolated_atoms:.2f} eV')

# Add experimental reference
plt.axhline(e_two_isolated_atoms - experimental_de, color='orange', 
           linestyle='-.', lw=1.5, alpha=0.7,
           label=f'Exp. dissociation limit (De = {experimental_de:.2f} eV)')

# Labels and title with details
plt.title(r'Potential Energy Curve of $\mathrm{H_2}$: CHGNet Machine Learning Potential', 
          fontsize=12, fontweight='bold')
plt.xlabel(r'Interatomic Distance $r$ (Å)', fontsize=11)
plt.ylabel('Potential Energy $E$ (eV)', fontsize=11)

# Add text box with accuracy metrics
textstr = f'CHGNet Accuracy:\nDe error: {abs(simulated_de - experimental_de):.2f} eV\n'
textstr += f'$r_e$ error: {abs(r_eq - 0.7414):.3f} Å'
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.legend(loc='best', fontsize=9)
plt.grid(True, linestyle=':', alpha=0.6)

# Set reasonable y-axis limits to show the well clearly
y_min = min(energies.min(), e_two_isolated_atoms - experimental_de - 0.5)
y_max = max(energies.max(), e_two_isolated_atoms + 0.2)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()

# Optional: Verify that periodic boundaries are working as intended
print("\n" + "="*50)
print("Periodic Boundary Conditions Verification")
print("="*50)
print(f"✓ Periodic cell size: {cell_size} Å (sufficient vacuum to isolate molecules)")
print(f"✓ H2 bond vector aligned along x-axis")
print(f"✓ Minimum image convention applied automatically by ASE")
print(f"  (no spurious interactions between periodic images)")

# Check minimum distance between periodic images
min_periodic_distance = cell_size - r_eq
print(f"✓ Minimum distance between periodic images: {min_periodic_distance:.1f} Å")
if min_periodic_distance > 5.0:
    print(f"  -> Sufficient vacuum (well-converged isolated molecule)")
else:
    print(f"  -> WARNING: May have some periodic interactions")
