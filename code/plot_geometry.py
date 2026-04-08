"""
plot_geometry.py

Illustrate the precession geometry and Thomas precession phase shift
for an SMBHB system, following the conventions of Mingarelli et al.
(2012), PRL 109, 081104, and the simple-precession approximation
of Apostolatos et al. (1994), PRD 49, 6274.

Figure 1: Simple precession — J, L, S vectors, precession cone,
          and the tilted binary orbital plane.

Figure 2: Thomas precession context — observer line of sight
          inside vs outside the precession cone, with the correct
          solid-angle phase shift 2pi(1 - cos lambda_L) per
          precession cycle.

Uses physical parameters from SMBHBEvolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smbhb_evolution import SMBHBEvolution, G_SI, C_SI, M_SUN

plt.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


def compute_vectors(binary):
    """
    Compute the angular momentum vectors from the SMBHBEvolution object.

    Returns normalised display vectors (scaled for plotting) and
    the physical precession cone half-angle lambda_L.
    """
    m1, m2 = binary.m1, binary.m2
    M = binary.M
    mu = binary.mu
    eta = binary.eta
    f = binary.f_gw_earth

    # Orbital angular momentum magnitude at leading order
    # L = mu (G M)^{2/3} / (pi f_GW)^{1/3}
    L_mag = mu * (G_SI * M) ** (2.0 / 3) / (np.pi * f) ** (1.0 / 3)

    # Spin angular momenta  S_i = chi_i G m_i^2 / c
    S1_mag = binary.chi1 * G_SI * m1 ** 2 / C_SI
    S2_mag = binary.chi2 * G_SI * m2 ** 2 / C_SI

    # Spin vectors (in a frame where L starts along z before precession)
    # kappa_i = angle between S_i and L
    S1_vec = S1_mag * np.array([
        np.sin(binary.kappa1), 0, np.cos(binary.kappa1)
    ])
    S2_vec = S2_mag * np.array([
        0, np.sin(binary.kappa2), np.cos(binary.kappa2)
    ])
    S_vec = S1_vec + S2_vec

    # L along z in the pre-precession frame
    L_vec = np.array([0.0, 0.0, L_mag])

    # Total angular momentum
    J_vec = L_vec + S_vec

    # Precession cone half-angle: angle between L and J
    cos_lambda = np.dot(L_vec, J_vec) / (np.linalg.norm(L_vec) * np.linalg.norm(J_vec))
    lambda_L = np.arccos(np.clip(cos_lambda, -1, 1))

    return L_vec, S_vec, J_vec, S1_vec, S2_vec, lambda_L, L_mag


def plot_precession_geometry(binary, alpha_deg=60, lambda_L_override=None):
    """
    Figure 1: Simple precession dynamics.

    Shows J (fixed), L precessing on a cone of half-angle lambda_L,
    S = J - L, the precession cone, and the tilted binary orbit.

    Parameters
    ----------
    lambda_L_override : float or None
        If given, override the physical lambda_L (in radians) for
        visual clarity.  The physical value (~1-2 deg) is too small
        to see the cone structure.
    """
    L_vec, S_vec, J_vec, S1_vec, S2_vec, lambda_L_phys, L_mag = compute_vectors(binary)
    lambda_L = lambda_L_override if lambda_L_override is not None else lambda_L_phys

    # Normalise for display: set |J| = 1
    scale = 1.0 / np.linalg.norm(J_vec)
    J_d = J_vec * scale
    L_d_mag = np.linalg.norm(L_vec) * scale

    # Scale individual spin vectors for display
    S1_d = S1_vec * scale
    S2_d = S2_vec * scale

    # Rotate L to an azimuthal angle alpha on the cone (around J)
    alpha = np.radians(alpha_deg)
    J_hat = J_d / np.linalg.norm(J_d)

    # Build L on the cone at azimuth alpha
    # First get a perpendicular basis in the plane normal to J
    if abs(J_hat[0]) < 0.9:
        perp = np.cross(J_hat, np.array([1, 0, 0]))
    else:
        perp = np.cross(J_hat, np.array([0, 1, 0]))
    e1 = perp / np.linalg.norm(perp)
    e2 = np.cross(J_hat, e1)

    L_d = L_d_mag * (
        np.cos(lambda_L) * J_hat
        + np.sin(lambda_L) * (np.cos(alpha) * e1 + np.sin(alpha) * e2)
    )
    S_d = J_d - L_d  # by definition J = L + S

    # Rotate S1, S2 into the same frame as L (apply the same rotation
    # that placed L on the cone at azimuth alpha)
    # Original frame: L along z. Build rotation matrix from z-hat to L_hat.
    L_hat_d = L_d / np.linalg.norm(L_d)
    z_hat = np.array([0.0, 0.0, 1.0])
    # Rodrigues rotation from z_hat to L_hat_d
    v_cross = np.cross(z_hat, L_hat_d)
    c_dot = np.dot(z_hat, L_hat_d)
    if np.linalg.norm(v_cross) > 1e-10:
        vx = np.array([[0, -v_cross[2], v_cross[1]],
                        [v_cross[2], 0, -v_cross[0]],
                        [-v_cross[1], v_cross[0], 0]])
        R_mat = np.eye(3) + vx + vx @ vx / (1 + c_dot)
    else:
        R_mat = np.eye(3)
    S1_d_rot = R_mat @ S1_d
    S2_d_rot = R_mat @ S2_d

    # ---- Figure ----
    fig = plt.figure(figsize=(5, 5.5))
    # Oversized axes to defeat matplotlib 3D padding
    ax = fig.add_axes([-0.3, -0.05, 1.6, 1.2], projection='3d')

    origin = np.array([0, 0, 0])

    # Vectors
    def draw_vec(v, color, label, fontsize=16, start=None):
        o = origin if start is None else start
        ax.quiver(*o, *v, color=color, linewidth=2.5,
                  arrow_length_ratio=0.08)
        tip = o + v
        ax.text(*(tip * 1.08 if start is None else tip + 0.03),
                label, color=color, fontsize=fontsize,
                fontweight='bold')

    draw_vec(J_d, 'black', r'$\mathbf{J}$')
    draw_vec(L_d, 'C0', r'$\mathbf{L}$')

    # ---- Precession cone ----
    theta_ring = np.linspace(0, 2 * np.pi, 120)
    R_ring = L_d_mag * np.sin(lambda_L)
    Z_ring = L_d_mag * np.cos(lambda_L)
    ring = np.array([
        R_ring * np.cos(theta_ring),
        R_ring * np.sin(theta_ring),
        np.full_like(theta_ring, Z_ring)
    ])
    # Rotate ring into J-aligned frame
    rot = np.column_stack([e1, e2, J_hat])  # columns = basis vectors
    ring_rot = rot @ ring
    ax.plot(ring_rot[0], ring_rot[1], ring_rot[2], 'C0--', alpha=0.5, lw=1)

    # Cone surface (light shading)
    n_gen = 30
    z_gen = np.linspace(0, Z_ring, n_gen)
    cone_x, cone_y, cone_z = [], [], []
    for zz in z_gen:
        r = zz * np.tan(lambda_L)
        pts = np.array([
            r * np.cos(theta_ring),
            r * np.sin(theta_ring),
            np.full_like(theta_ring, zz)
        ])
        pts_rot = rot @ pts
        cone_x.append(pts_rot[0])
        cone_y.append(pts_rot[1])
        cone_z.append(pts_rot[2])
    cone_x = np.array(cone_x)
    cone_y = np.array(cone_y)
    cone_z = np.array(cone_z)
    ax.plot_surface(cone_x, cone_y, cone_z, color='C0', alpha=0.06,
                    edgecolor='none')

    # ---- Angle arc for lambda_L ----
    arc_frac = np.linspace(0, lambda_L, 25)
    arc_r = 0.35
    arc_pts = arc_r * (
        np.cos(arc_frac)[:, None] * J_hat[None, :]
        + np.sin(arc_frac)[:, None] * (
            np.cos(alpha) * e1[None, :] + np.sin(alpha) * e2[None, :]
        )
    )
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2], 'k-', lw=1.2)
    mid = len(arc_frac) // 2
    ax.text(arc_pts[mid, 0] * 1.15, arc_pts[mid, 1] * 1.15,
            arc_pts[mid, 2] * 1.15, r'$\lambda_L$', fontsize=15)

    # ---- Angle arc for alpha ----
    arc_alpha = np.linspace(0, alpha, 35)
    r_a = R_ring * 0.6
    alpha_pts = (
        Z_ring * J_hat[None, :]
        + r_a * (np.cos(arc_alpha)[:, None] * e1[None, :]
                 + np.sin(arc_alpha)[:, None] * e2[None, :])
    )
    ax.plot(alpha_pts[:, 0], alpha_pts[:, 1], alpha_pts[:, 2],
            'C2-', lw=1.8)
    # Arrowhead at the end of the alpha arc
    arrow_dir = alpha_pts[-1] - alpha_pts[-3]
    arrow_dir = arrow_dir / np.linalg.norm(arrow_dir) * 0.06
    ax.quiver(*alpha_pts[-1], *arrow_dir, color='C2', linewidth=1.8,
              arrow_length_ratio=0.6)
    # Reference line at alpha=0
    ref_pt = Z_ring * J_hat + r_a * e1
    ax.plot(*zip(Z_ring * J_hat, ref_pt), 'C2--', alpha=0.4, lw=1)
    amid = len(arc_alpha) // 2
    ax.text(alpha_pts[amid, 0] * 1.05, alpha_pts[amid, 1] * 1.05,
            alpha_pts[amid, 2], r'$\alpha$', color='C2', fontsize=15)

    # ---- Total spin S at the origin ----
    s_display_scale = 4.5
    S_total_d = (S1_d_rot + S2_d_rot) * s_display_scale
    draw_vec(S_total_d, 'C3', r'$\mathbf{S}$', fontsize=16)

    lim = 0.75
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-0.35, lim + 0.1])
    ax.set_axis_off()
    ax.view_init(elev=22, azim=40)

    return fig


def plot_thomas_precession(binary, lambda_L_override=None):
    """
    Figure 2: Thomas precession — inside vs outside the precession cone.

    The accumulated Thomas phase per precession cycle is:
      - 2 pi                    if Omega_hat is inside the cone
      - 2 pi (1 - cos lambda_L) if Omega_hat is outside the cone

    (Apostolatos et al. 1994; Mingarelli et al. 2012 Eq. for phi_T)
    For weak-field sources lambda_L << 1, so the outside case is
    suppressed by ~ lambda_L^2 / 2.
    """
    _, _, J_vec, _, _, lambda_L_phys, L_mag = compute_vectors(binary)
    lambda_L = lambda_L_override if lambda_L_override is not None else lambda_L_phys
    scale = 1.0 / np.linalg.norm(J_vec)
    L_d_mag = L_mag * scale
    J_hat = np.array([0, 0, 1.0])  # J along z for simplicity

    e1 = np.array([1.0, 0, 0])
    e2 = np.array([0, 1.0, 0])

    fig = plt.figure(figsize=(14, 7))

    for idx, inside in enumerate([True, False]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        # ---- Precession cone ----
        theta_ring = np.linspace(0, 2 * np.pi, 100)
        R_ring = L_d_mag * np.sin(lambda_L)
        Z_ring = L_d_mag * np.cos(lambda_L)

        ax.plot(R_ring * np.cos(theta_ring),
                R_ring * np.sin(theta_ring),
                np.full_like(theta_ring, Z_ring),
                'C0--', alpha=0.6, lw=1)

        # Cone surface
        z_gen = np.linspace(0, Z_ring, 20)
        for zz in z_gen:
            r = zz * np.tan(lambda_L)
            ax.plot(r * np.cos(theta_ring), r * np.sin(theta_ring),
                    np.full_like(theta_ring, zz), 'C0-', alpha=0.03, lw=0.3)

        # J vector
        ax.quiver(0, 0, 0, 0, 0, 1.0, color='black', linewidth=2.5,
                  arrow_length_ratio=0.08)
        ax.text(0, 0, 1.06, r'$\mathbf{J}$', fontsize=16, fontweight='bold')

        # L vector (on the cone at alpha=0)
        L_d = L_d_mag * np.array([np.sin(lambda_L), 0, np.cos(lambda_L)])
        ax.quiver(0, 0, 0, *L_d, color='C0', linewidth=2.5,
                  arrow_length_ratio=0.08)
        ax.text(*(L_d * 1.1), r'$\mathbf{L}$', color='C0', fontsize=16,
                fontweight='bold')

        # Omega_hat (GW propagation direction)
        O_len = 0.85
        if inside:
            # Inside the cone: theta_Omega < lambda_L
            theta_O = lambda_L * 0.4
            phi_O = np.pi / 3
        else:
            # Outside the cone: theta_Omega > lambda_L
            theta_O = lambda_L + np.radians(35)
            phi_O = np.pi * 1.2

        Omega = O_len * np.array([
            np.sin(theta_O) * np.cos(phi_O),
            np.sin(theta_O) * np.sin(phi_O),
            np.cos(theta_O)
        ])
        ax.quiver(0, 0, 0, *Omega, color='C1', linewidth=2.5,
                  arrow_length_ratio=0.08)
        ax.text(*(Omega * 1.12), r'$\hat{\Omega}$', color='C1',
                fontsize=16, fontweight='bold')

        # lambda_L arc
        arc_frac = np.linspace(0, lambda_L, 20)
        arc_r = 0.4
        ax.plot(arc_r * np.sin(arc_frac),
                np.zeros_like(arc_frac),
                arc_r * np.cos(arc_frac), 'k-', lw=1.2)
        mid = len(arc_frac) // 2
        ax.text(arc_r * np.sin(arc_frac[mid]) * 1.2, 0,
                arc_r * np.cos(arc_frac[mid]) * 1.1,
                r'$\lambda_L$', fontsize=14)

        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-0.05, 1.15])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    # Use parameters from the paper's fiducial system
    binary = SMBHBEvolution(
        m1=1e9, m2=1e9,
        chi1=0.5, chi2=0.5,
        kappa1=0.4, kappa2=0.8,   # different misalignments for visual separation
        f_gw_earth=100e-9,
        D_L=100.0,
        iota=np.pi / 4,
    )

    outdir = Path(__file__).resolve().parent

    # Exaggerate lambda_L for visual clarity (physical value ~1.5 deg
    # is too small to see cone structure)
    lambda_L_display = np.radians(25)

    from PIL import Image

    def save_cropped(fig, stem, dpi=300, pad=10):
        """Render to PNG, auto-crop whitespace, save PNG + PDF."""
        tmp = outdir / f"_{stem}_raw.png"
        fig.savefig(tmp, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        img = Image.open(tmp)
        # Find bounding box of non-white pixels
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = Image.fromarray(
            np.abs(np.array(img).astype(int)
                   - np.array(bg).astype(int)).astype(np.uint8)
        )
        bbox = diff.getbbox()
        if bbox:
            bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                    min(img.width, bbox[2] + pad),
                    min(img.height, bbox[3] + pad))
            img = img.crop(bbox)
        img.save(outdir / f"{stem}.png")
        # Also save a tight PDF by re-rendering from the cropped image
        # (matplotlib PDF with 3D axes can't be cropped the same way)
        img.save(outdir / f"{stem}.pdf")
        try:
            tmp.unlink()
        except OSError:
            pass  # sandbox may block deletion
        print(f"Saved: {outdir}/{stem}.pdf  ({img.width}x{img.height})")

    fig1 = plot_precession_geometry(binary, alpha_deg=60,
                                    lambda_L_override=lambda_L_display)
    save_cropped(fig1, "fig_precession_geometry", dpi=300)

    fig2 = plot_thomas_precession(binary,
                                  lambda_L_override=lambda_L_display)
    save_cropped(fig2, "fig_thomas_precession", dpi=300)

    plt.close('all')

    # Print physical parameters for reference
    _, S_vec, J_vec, S1_vec, S2_vec, lambda_L, L_mag = compute_vectors(binary)
    print(f"\nPhysical parameters:")
    print(f"  |L|       = {L_mag:.3e} kg m^2/s")
    print(f"  |S1|      = {np.linalg.norm(S1_vec):.3e} kg m^2/s")
    print(f"  |S2|      = {np.linalg.norm(S2_vec):.3e} kg m^2/s")
    print(f"  |S|/|L|   = {np.linalg.norm(S_vec)/L_mag:.4f}")
    print(f"  lambda_L  = {np.degrees(lambda_L):.2f} deg")
    print(f"  zeta_L    = {np.degrees(binary.zeta_L):.2f} deg")
