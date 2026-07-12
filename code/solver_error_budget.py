"""Quantify convergence of fixed-count Picard/Newton inversion of the TaylorT2
time equation vs a high-precision bisection reference (proxy for the
authoritative brentq path, smbhb_evolution.py:541-554). Pure stdlib."""
import math

GMsun = 1.327124400e20
c = 299792458.0
Tsun = GMsun / c**3          # 4.9255e-6 s
YR = 365.25 * 86400.0

def coeffs(eta, beta_so, sigma_ss):
    t2 = 743.0/252 + 11.0*eta/3
    t3 = -(32.0/5)*math.pi + beta_so*8.0/5.0
    t4 = (3058673.0/508032 + 5429.0*eta/504 + 617.0*eta**2/72) - 2.0*sigma_ss
    return t2, t3, t4

def bisect(f, a, b, n=200):
    fa = f(a)
    for _ in range(n):
        m = 0.5*(a+b)
        fm = f(m)
        if fa*fm <= 0: b = m
        else: a, fa = m, fm
    return 0.5*(a+b)

def run(Mtot_msun, eta, chi, fE, tau_yr):
    Ms = Mtot_msun * Tsun
    t2, t3, t4 = coeffs(eta, 47.0/6.0*chi, 0.0)  # equal-mass aligned SO, SS off
    K = 5.0*Ms/(256.0*eta)
    vE = (math.pi*Ms*fE)**(1.0/3)
    tau = tau_yr*YR
    T  = lambda v: K*v**-8*(1 + t2*v**2 + t3*v**3 + t4*v**4)
    Tp = lambda v: K*(-8*v**-9 - 6*t2*v**-7 - 5*t3*v**-6 - 4*t4*v**-5)
    target = T(vE) + tau

    v_exact = bisect(lambda v: T(v)-target, vE*1e-3, vE*0.9999)
    f_exact = v_exact**3/(math.pi*Ms)

    # 0PN closed-form seed (same form as enterprise cw_delay:415-417)
    v0 = (vE**-8 + 256.0*eta*tau/(5.0*Ms))**(-1.0/8)

    v = v0; eP = []
    for _ in range(4):
        v = (K*(1 + t2*v**2 + t3*v**3 + t4*v**4)/target)**(1.0/8)
        eP.append(abs(v**3/(math.pi*Ms) - f_exact)*1e9)
    v = v0; eN = []
    for _ in range(4):
        v = v - (T(v)-target)/Tp(v)
        eN.append(abs(v**3/(math.pi*Ms) - f_exact)*1e9)
    err0 = abs(v0**3/(math.pi*Ms) - f_exact)*1e9
    return vE, v_exact, f_exact*1e9, (fE - f_exact)*1e9, err0, eP, eN

cases = [
    ("2x1e9 Msun chi=0.9, fE=100 nHz, tau=6500 yr (~1 kpc worst-case)", 2e9, 0.25, 0.9, 100e-9, 6500),
    ("2x1e9 Msun chi=0.9, fE=50 nHz,  tau=6500 yr",                     2e9, 0.25, 0.9, 50e-9, 6500),
    ("2x1e9 Msun chi=0.9, fE=20 nHz,  tau=20000 yr (~3 kpc)",           2e9, 0.25, 0.9, 20e-9, 20000),
    ("2x5e9 Msun chi=0.9, fE=50 nHz,  tau=6500 yr",                     1e10, 0.25, 0.9, 50e-9, 6500),
]
for name, M, eta, chi, fE, tau in cases:
    vE, vP, fP, dfPE, err0, eP, eN = run(M, eta, chi, fE, tau)
    print(name)
    print(f"  v_E={vE:.4f} v_P={vP:.4f} f_P={fP:.3f} nHz, f_E-f_P={dfPE:.3f} nHz")
    print(f"  |f_P err| 0PN closed-form seed: {err0:.3e} nHz")
    print("  Picard k=1..4: " + " ".join(f"{e:.2e}" for e in eP) + " nHz")
    print("  Newton k=1..4: " + " ".join(f"{e:.2e}" for e in eN) + " nHz")
