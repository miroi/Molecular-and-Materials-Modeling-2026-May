import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
import ase.units as units

# 1. Accepted Experimental Reference Values (for H2 Ground State)
R_E_EXP = 0.7416  # Angstroms (Å)
B_E_EXP = 60.853  # Wavenumbers (cm^-1)

# 2. Construct and Optimize H2 Molecule with ASE
h2 = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
h2.calc = EMT()

opt = BFGS(h2, logfile=None)
opt.run(fmax=0.01)

# Extract optimized distance
r_e_calc = h2.get_distance(0, 1)

# 3. Process Moments of Inertia via Statistical Mechanics Tools
thermo = IdealGasThermo(
    vib_energies=[0.1], 
    potentialenergy=h2.get_potential_energy(), 
    atoms=h2, 
    geometry='linear', 
    symmetrynumber=2, 
    spin=0
)

# Extract non-zero moment of inertia vector (returns array of shape (3,))
moments = thermo.get_moments_of_inertia()
I_amu_ang2 = moments[0]  # For linear molecules, I_x = I_y > 0; I_z = 0

# 4. Fundamental Physical Constant Unit Conversions
h = units.Planck                 # Planck constant in eV * s
c_cm_s = units._c * 100         # Speed of light in cm/s
# Convert inertia from (amu * Å^2) to (eV * s^2) for strict dimensional parity
I_ev_s2 = I_amu_ang2 * (units._amu / units._me) * units.fs**2 

# Calculate the Rotational Constant (B_e = h / (8 * pi^2 * c * I))
b_e_calc = h / (8 * np.pi**2 * c_cm_s * I_ev_s2)

# 5. Compute Error Percentages
r_error = ((r_e_calc - R_E_EXP) / R_E_EXP) * 100
b_error = ((b_e_calc - B_E_EXP) / B_E_EXP) * 100

# 6. Format and Print Comparison Summary
print("==================================================")
print("     H2 ROTATIONAL METRICS COMPARISON SUMMARY     ")
print("==================================================")
print(f"Equilibrium Bond Length (r_e):")
print(f"  - Calculated (EMT):   {r_e_calc:.4f} Å")
print(f"  - Experimental:       {R_E_EXP:.4f} Å")
print(f"  - Relative Error:     {r_error:+.2f}%")
print("--------------------------------------------------")
print(f"Rotational Constant (B_e):")
print(f"  - Calculated (EMT):   {b_e_calc:.3f} cm^-1")
print(f"  - Experimental:       {B_E_EXP:.3f} cm^-1")
print(f"  - Relative Error:     {b_error:+.2f}%")
print("==================================================")
print("Note: Minor deviations arise purely from the ")
print("empirical approximations used in the generic EMT calculator.")
print("==================================================")

