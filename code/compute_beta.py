"""
Compute the dimensionless chirp parameter beta for fiducial echo sources.

beta = 4 fdot L_p / (3 f_E c)

where fdot is the leading-order (Newtonian) GW frequency derivative:
    fdot = (96/5) pi^(8/3) (G Mc / c^3)^(5/3) f_E^(11/3)

and Mc = eta^(3/5) M is the chirp mass, eta = m1 m2 / M^2 the symmetric
mass ratio, M = m1 + m2 the total mass.
"""

import numpy as np

# Physical constants
G = 6.674e-11       # m^3 kg^-1 s^-2
c = 2.998e8          # m/s
M_sun = 1.989e30     # kg
pc = 3.086e16        # m

def compute_beta(M_tot_Msun, f_E_Hz, L_p_pc, eta=0.25):
    """
    Parameters
    ----------
    M_tot_Msun : float
        Total binary mass in solar masses.
    f_E_Hz : float
        Earth-term GW frequency in Hz.
    L_p_pc : float
        Pulsar distance in parsecs.
    eta : float
        Symmetric mass ratio (0.25 for equal mass).

    Returns
    -------
    beta : float
        Dimensionless chirp parameter.
    fdot : float
        Leading-order frequency derivative in Hz/s.
    theta_opt_deg : float or None
        Optimal source-pulsar angle in degrees (None if no peak).
    """
    M = M_tot_Msun * M_sun
    Mc = eta**(3.0 / 5) * M
    f = f_E_Hz
    L = L_p_pc * pc

    fdot = (96.0 / 5) * np.pi**(8.0 / 3) * (G * Mc / c**3)**(5.0 / 3) * f**(11.0 / 3)
    beta = 4 * fdot * L / (3 * f * c)

    if beta > 2:
        sin2 = (beta - 2) / (9 * beta)
        theta_opt = 2 * np.arcsin(np.sqrt(sin2))
        theta_opt_deg = np.degrees(theta_opt)
    else:
        theta_opt_deg = None

    return beta, fdot, theta_opt_deg


if __name__ == "__main__":
    print("Dimensionless chirp parameter beta for fiducial echo sources")
    print("Equal mass (eta = 0.25)")
    print("=" * 70)

    cases = [
        ("Optimistic",    1e9,   1e-6, 1000),
        ("Typical",       5e8,   1e-6, 1000),
        ("Conservative",  1e8,   1e-5, 1000),
    ]

    for name, M, fE, Lp in cases:
        beta, fdot, theta_opt = compute_beta(M, fE, Lp)
        print(f"\n{name}: M_tot = {M:.0e} M_sun, f_E = {fE:.0e} Hz, L_p = {Lp} pc")
        print(f"  Mc      = {0.25**(3./5) * M:.3e} M_sun")
        print(f"  fdot    = {fdot:.3e} Hz/s")
        print(f"  beta    = {beta:.1f}")
        if theta_opt is not None:
            print(f"  theta_opt = {theta_opt:.1f} deg")
        else:
            print(f"  theta_opt: no peak (beta < 2)")
