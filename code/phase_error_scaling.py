"""
Phase error from pulsar distance uncertainty.

delta_Phi = 2 * pi * f * delta_L / c

Coherent matching requires delta_Phi < 1 rad,
so delta_L < c / (2 * pi * f).
"""
import numpy as np

c = 2.998e8          # m/s
pc = 3.0857e16       # m per parsec

# Maximum allowed distance error for 1 rad of phase error
for f_nHz in [10, 30, 50, 100]:
    f = f_nHz * 1e-9
    dL_max = c / (2 * np.pi * f)
    print(f"f = {f_nHz:>3d} nHz:  dL_max = {dL_max/pc:.4f} pc  ({dL_max/pc*1e3:.1f} mpc)")

print()

# Phase error for delta_L = 100 pc at various frequencies
dL = 100 * pc
for f_nHz in [10, 30, 50, 100]:
    f = f_nHz * 1e-9
    dphi = 2 * np.pi * f * dL / c
    print(f"f = {f_nHz:>3d} nHz, dL = 100 pc:  delta_Phi = {dphi:.0f} rad  ({dphi/(2*np.pi):.0f} cycles)")
