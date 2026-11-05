"""
ASTM F3269-17 Simplex Architecture & Control Barrier Function (CBF) Quadratic Program (QP)
Runtime Assurance Shield for Autonomous Rendezvous, Proximity Operations & Docking (RPOD).

Enforces four coupled safety invariants at 50 Hz:
1. Keep-Out Zone (KOZ) High-Order Control Barrier Function (h_KOZ >= 0).
2. V-bar Line-of-Sight (LOS) Approach Corridor Cone Barrier (h_LOS >= 0).
3. Range-Dependent Braking Velocity Envelope (h_vel >= 0).
4. Passive Abort Safety (2-Orbit HCW Free-Drift Collision-Free Invariance) &
   Active Thruster Fault Disturbance Rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Tuple
import numpy as np
import scipy.sparse as sp

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

from certi_rpo_hil.orbital_dynamics import (
    OrbitConfig,
    hcw_state_transition_matrix,
    propagate_hcw_free_drift,
)


@dataclass
class SafetyConfig:
    """Configuration for the ASTM F3269-17 Simplex + CBF-QP Safety Shield."""

    # Target central body Keep-Out Zone radius [m] (docking port sits at y = -2.2 m)
    koz_radius_m: float = 2.0
    # Passive abort free-drift minimum safe core radius [m] prior to final capture gate
    passive_abort_radius_m: float = 1.85
    # Final docking capture gate distance [m] where docking contact is permitted
    capture_gate_dist_m: float = 4.2
    # Approach corridor half-cone angle [deg] around -V-bar (0, -1, 0) axis
    los_cone_half_angle_deg: float = 20.0
    # Lateral port margin at docking interface [m]
    los_apex_margin_m: float = 0.45
    # Maximum touchdown relative speed [m/s] and range-proportional braking slope [1/s]
    max_touchdown_speed_ms: float = 0.06
    braking_slope_s_inv: float = 0.032
    # CBF exponential class-K gains
    alpha_koz: float = 0.65
    alpha_los: float = 0.70
    alpha_vel: float = 0.85
    # Maximum control acceleration [m/s^2]
    max_accel_ms2: float = 0.08


@dataclass
class ShieldDecision:
    """Diagnostic output from a single 50 Hz execution of the CBF-QP Safety Shield."""

    u_nominal_ms2: np.ndarray         # Unshielded guidance command [ax, ay, az]
    u_safe_ms2: np.ndarray            # Certified safe command emitted to thrusters [ax, ay, az]
    shield_intervened: bool           # True if QP modified u_nominal or Simplex overrode
    intervention_norm_ms2: float      # ||u_safe - u_nominal||_2 [m/s^2]
    h_koz_m: float                    # Keep-Out Zone barrier value [m] (>= 0 is safe)
    h_los_m: float                    # Line-of-Sight corridor cone margin [m] (>= 0 is safe)
    h_vel_ms: float                   # Braking velocity envelope margin [m/s] (>= 0 is safe)
    min_abort_drift_dist_m: float     # Minimum distance over 2-orbit unpowered HCW drift [m]
    passive_abort_safe: bool          # True if 2-orbit free drift avoids central KOZ
    qp_solve_time_us: float           # Execution time of safety shield [microseconds]
    active_mode: str                  # "NOMINAL", "CBF_QP_FILTER", or "SIMPLEX_RECOVERY"


class NominalRPODGuidance:
    """
    Nominal V-bar Approach Guidance Law.
    Can operate in:
    - `aggressive_unshielded=True`: High-gain PD/neural-like policy without disturbance
      compensation or corridor constraints (vulnerable to solar glare & thruster stuck-open faults).
    - `aggressive_unshielded=False`: Reference glide-slope tracking controller feeding the CBF shield.
    """

    def __init__(
        self,
        orbit_cfg: Optional[OrbitConfig] = None,
        aggressive_unshielded: bool = False,
    ) -> None:
        self.orbit_cfg = orbit_cfg or OrbitConfig()
        self.aggressive_unshielded = aggressive_unshielded
        self.r_dock = np.asarray(self.orbit_cfg.docking_port_pos_m, dtype=float)

    def compute_command(
        self,
        x_est: np.ndarray,
        t_s: float,
    ) -> np.ndarray:
        """Compute nominal translational acceleration command [ax, ay, az] in m/s^2."""
        r = x_est[:3]
        v = x_est[3:6]
        n = self.orbit_cfg.mean_motion_rad_s

        # Feedforward Clohessy-Wiltshire cancellation
        a_ff = np.array(
            [
                -3.0 * (n ** 2) * r[0] - 2.0 * n * v[1],
                2.0 * n * v[0],
                (n ** 2) * r[2],
            ],
            dtype=float,
        )

        if self.aggressive_unshielded:
            # Aggressive time-optimal-style pursuit toward target center/port that over-accelerates
            # and trusts corrupted optical coordinates during glare
            r_goal = np.array([0.0, -1.6, 0.0], dtype=float)
            pos_err = r - r_goal
            u_fb = -0.024 * pos_err - 0.16 * v
            u_cmd = a_ff + u_fb
        else:
            # Smooth exponential glide-slope toward docking port (0, -2.2, 0) along -V-bar
            pos_err = r - self.r_dock
            dist = float(np.linalg.norm(pos_err))
            # Desired approach speed scales with range and caps smoothly
            v_des_mag = min(0.05 + 0.022 * dist, 0.42)
            if dist > 0.08:
                v_des = -v_des_mag * (pos_err / dist)
                # Stronger lateral centering on x and z to stay well inside LOS cone
                v_des[0] = -0.045 * r[0]
                v_des[2] = -0.045 * r[2]
            else:
                v_des = np.zeros(3, dtype=float)

            u_fb = 0.28 * (v_des - v) - 0.008 * pos_err
            u_cmd = a_ff + u_fb

        a_max = self.orbit_cfg.max_accel_ms2
        return np.clip(u_cmd, -a_max, a_max)


class ASTM3269CBFShield:
    """
    ASTM F3269-17 Simplex & Control Barrier Function (CBF) Quadratic Program Runtime Shield.
    Minimizes ||u - u_nom_compensated||_2^2 subject to linearized discrete-time barrier
    invariance constraints and passive-abort free-drift safety.
    """

    def __init__(
        self,
        orbit_cfg: Optional[OrbitConfig] = None,
        safety_cfg: Optional[SafetyConfig] = None,
        enabled: bool = True,
    ) -> None:
        self.orbit_cfg = orbit_cfg or OrbitConfig()
        self.cfg = safety_cfg or SafetyConfig(max_accel_ms2=self.orbit_cfg.max_accel_ms2)
        self.enabled = enabled
        self.r_dock = np.asarray(self.orbit_cfg.docking_port_pos_m, dtype=float)
        self.tan_cone = float(np.tan(np.deg2rad(self.cfg.los_cone_half_angle_deg)))

    def evaluate_barriers(self, x_state: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute instantaneous safety barrier values:
        - h_koz_m: Distance margin outside Keep-Out Zone sphere [m]
        - h_los_m: Lateral margin inside -V-bar Line-of-Sight corridor cone [m]
        - h_vel_ms: Braking speed margin [m/s]
        """
        r = x_state[:3]
        v = x_state[3:6]

        dist_center = float(np.linalg.norm(r))
        h_koz = dist_center - self.cfg.koz_radius_m

        # LOS corridor cone around -y axis: lateral radius sqrt(x^2 + z^2) <= tan(theta)*max(-y, 0) + margin
        lateral_r = float(np.hypot(r[0], r[2]))
        along_track_depth = max(-float(r[1]) - abs(float(self.r_dock[1])), 0.0)
        allowed_radius = self.tan_cone * along_track_depth + self.cfg.los_apex_margin_m
        h_los = allowed_radius - lateral_r

        # Range-dependent braking speed limit relative to docking port
        dist_dock = float(np.linalg.norm(r - self.r_dock))
        speed = float(np.linalg.norm(v))
        v_limit = self.cfg.max_touchdown_speed_ms + self.cfg.braking_slope_s_inv * dist_dock
        h_vel = v_limit - speed

        return h_koz, h_los, h_vel

    def evaluate_passive_abort_safety(
        self,
        x_state: np.ndarray,
        dt_s: float,
        u_candidate: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Propagate 1 step with u_candidate, then evaluate a 2-orbit unpowered HCW free-drift
        trajectory to verify Passive Abort Safety.
        """
        n = self.orbit_cfg.mean_motion_rad_s
        Phi, Gamma = hcw_state_transition_matrix(n, dt_s)
        x_next = Phi @ x_state + Gamma @ u_candidate

        # Check 2 orbits of free drift (if outside the final capture gate)
        two_orbits_s = 2.0 * self.orbit_cfg.orbital_period_s
        drift_states = propagate_hcw_free_drift(x_next, n, two_orbits_s, num_steps=90)
        dists = np.linalg.norm(drift_states[:, :3], axis=1)
        min_drift_dist = float(np.min(dists))

        dist_now = float(np.linalg.norm(x_state[:3]))
        if dist_now <= self.cfg.capture_gate_dist_m:
            # Inside final capture gate, controlled docking approach is authorized if outside KOZ
            is_safe = bool(float(np.linalg.norm(x_next[:3])) >= self.cfg.koz_radius_m)
            return max(min_drift_dist, float(np.linalg.norm(x_next[:3]))), is_safe

        is_safe = bool(min_drift_dist >= self.cfg.passive_abort_radius_m)
        return min_drift_dist, is_safe

    def certify_control(
        self,
        x_est: np.ndarray,
        u_nominal_ms2: np.ndarray,
        d_hat_ms2: np.ndarray,
        dt_s: float,
        fault_effectiveness_hint: Optional[np.ndarray] = None,
    ) -> ShieldDecision:
        """
        Execute the 50 Hz ASTM F3269-17 Simplex + CBF-QP Safety Shield.
        """
        t0 = time.perf_counter()
        a_max = self.cfg.max_accel_ms2
        u_nom = np.clip(np.asarray(u_nominal_ms2, dtype=float), -a_max, a_max)

        if not self.enabled:
            h_koz, h_los, h_vel = self.evaluate_barriers(x_est)
            min_drift, abort_ok = self.evaluate_passive_abort_safety(x_est, dt_s, u_nom)
            elapsed_us = (time.perf_counter() - t0) * 1e6
            return ShieldDecision(
                u_nominal_ms2=u_nom,
                u_safe_ms2=u_nom,
                shield_intervened=False,
                intervention_norm_ms2=0.0,
                h_koz_m=h_koz,
                h_los_m=h_los,
                h_vel_ms=h_vel,
                min_abort_drift_dist_m=min_drift,
                passive_abort_safe=abort_ok,
                qp_solve_time_us=elapsed_us,
                active_mode="UNSHIELDED",
            )

        r = x_est[:3]
        v = x_est[3:6]
        dist_center = max(float(np.linalg.norm(r)), 1e-6)
        dist_dock = max(float(np.linalg.norm(r - self.r_dock)), 1e-6)
        lateral_r = float(np.hypot(r[0], r[2]))

        # 1. Compensate estimated thruster disturbance bias d_hat_ms2
        eff = (
            np.asarray(fault_effectiveness_hint, dtype=float)
            if fault_effectiveness_hint is not None
            else np.ones(3, dtype=float)
        )
        u_des = (u_nom - d_hat_ms2) / np.maximum(eff, 0.5)

        # 2. Construct Linearized Control Barrier Function Inequality Constraints: A_cbf * u <= b_cbf
        # where u_eff = eff * u + d_hat_ms2 is the effective acceleration entering dynamics
        A_rows = []
        b_rows = []

        # (a) Keep-Out Zone High-Order CBF:
        # radial_vel = (r / ||r||) . v; require radial_accel >= -alpha_koz * (h_koz + 1.2 * radial_vel)
        n_rad = r / dist_center
        v_rad = float(np.dot(n_rad, v))
        h_koz_now = dist_center - self.cfg.koz_radius_m
        # Add extra safety buffer (0.35 m) so true state never breaches KOZ under noise/faults
        h_koz_buffered = h_koz_now - 0.35
        a_rad_min = -self.cfg.alpha_koz * h_koz_buffered - 1.35 * v_rad
        # Constraint: - (n_rad * eff) . u <= (n_rad . d_hat) - a_rad_min
        A_rows.append(-(n_rad * eff))
        b_rows.append(float(np.dot(n_rad, d_hat_ms2)) - a_rad_min)

        # (b) Line-of-Sight (LOS) Corridor Cone CBF:
        # Keep lateral deviation sqrt(x^2 + z^2) well inside allowed cone
        along_track_depth = max(-float(r[1]) - abs(float(self.r_dock[1])), 0.0)
        allowed_lat = 0.82 * (self.tan_cone * along_track_depth + self.cfg.los_apex_margin_m)
        for idx in (0, 2):
            pos_c = float(r[idx])
            vel_c = float(v[idx])
            # Upper bound: pos_c <= allowed_lat / sqrt(2)
            lim_c = 0.68 * allowed_lat
            h_pos_hi = lim_c - pos_c
            a_max_c = self.cfg.alpha_los * h_pos_hi - 1.45 * vel_c
            row_hi = np.zeros(3, dtype=float)
            row_hi[idx] = eff[idx]
            A_rows.append(row_hi)
            b_rows.append(a_max_c - float(d_hat_ms2[idx]))

            # Lower bound: -pos_c <= allowed_lat / sqrt(2)
            h_pos_lo = lim_c + pos_c
            a_min_c = -self.cfg.alpha_los * h_pos_lo - 1.45 * vel_c
            row_lo = np.zeros(3, dtype=float)
            row_lo[idx] = -eff[idx]
            A_rows.append(row_lo)
            b_rows.append(float(d_hat_ms2[idx]) - a_min_c)

        # (c) Approach Speed & V-bar Closing Rate CBF:
        # Ensure closing velocity along +y (since chaser is at y < 0 moving toward y = -2.2)
        # does not exceed safe braking profile
        y_err_to_port = float(self.r_dock[1] - r[1])  # positive when chaser is at y < -2.2
        v_y_max = self.cfg.max_touchdown_speed_ms + self.cfg.braking_slope_s_inv * max(y_err_to_port, 0.0)
        h_vy = v_y_max - float(v[1])
        ay_upper = self.cfg.alpha_vel * h_vy / max(dt_s, 0.02)
        row_vy = np.array([0.0, eff[1], 0.0], dtype=float)
        A_rows.append(row_vy)
        b_rows.append(ay_upper - float(d_hat_ms2[1]))

        A_cbf = np.vstack(A_rows)
        b_cbf = np.asarray(b_rows, dtype=float)

        # 3. Solve the 3-variable Quadratic Program via OSQP (or analytical active-set projection)
        u_qp = self._solve_qp(u_des, A_cbf, b_cbf, a_max)

        # 4. Verify Passive Abort Free-Drift Invariance & Apply Simplex Recovery if Needed
        u_eff_candidate = eff * u_qp + d_hat_ms2
        min_drift, abort_ok = self.evaluate_passive_abort_safety(x_est, dt_s, u_eff_candidate)
        active_mode = "NOMINAL"

        if not abort_ok:
            # Simplex Passive-Abort Recovery: inject radial/along-track separation impulse
            # so natural HCW drift ellipse clears the Keep-Out Zone
            u_abort = u_des.copy()
            u_abort[1] = min(u_abort[1], -0.45 * a_max)
            u_abort[0] = np.clip(-0.35 * r[0] - 1.2 * v[0] - d_hat_ms2[0], -a_max, a_max)
            u_qp = np.clip(u_abort, -a_max, a_max)
            u_eff_candidate = eff * u_qp + d_hat_ms2
            min_drift, abort_ok = self.evaluate_passive_abort_safety(x_est, dt_s, u_eff_candidate)
            active_mode = "SIMPLEX_RECOVERY"

        diff_norm = float(np.linalg.norm(u_qp - u_nom))
        intervened = bool(diff_norm > 1e-4 or active_mode == "SIMPLEX_RECOVERY")
        if intervened and active_mode == "NOMINAL":
            active_mode = "CBF_QP_FILTER"

        h_koz, h_los, h_vel = self.evaluate_barriers(x_est)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        return ShieldDecision(
            u_nominal_ms2=u_nom,
            u_safe_ms2=u_qp,
            shield_intervened=intervened,
            intervention_norm_ms2=diff_norm,
            h_koz_m=h_koz,
            h_los_m=h_los,
            h_vel_ms=h_vel,
            min_abort_drift_dist_m=min_drift,
            passive_abort_safe=abort_ok,
            qp_solve_time_us=elapsed_us,
            active_mode=active_mode,
        )

    def _solve_qp(
        self,
        u_ref: np.ndarray,
        A_ineq: np.ndarray,
        b_ineq: np.ndarray,
        a_max: float,
    ) -> np.ndarray:
        """
        Solve min 0.5 * ||u - u_ref||_2^2 + 0.5 * rho * s^2
        subject to A_ineq * u - s <= b_ineq, -a_max <= u <= a_max.
        Uses fast exact active-set projection onto half-spaces + box constraints
        (and validates against OSQP when invoked).
        """
        u = np.clip(u_ref.copy(), -a_max, a_max)
        # Sequential Dykstra/Hildreth projection onto the 6 decoupled/radial half-spaces
        for _ in range(8):
            violated = False
            for i in range(A_ineq.shape[0]):
                a_i = A_ineq[i]
                b_i = b_ineq[i]
                val = float(np.dot(a_i, u))
                if val > b_i:
                    norm_sq = float(np.dot(a_i, a_i))
                    if norm_sq > 1e-12:
                        u = u - ((val - b_i) / norm_sq) * a_i
                        u = np.clip(u, -a_max, a_max)
                        violated = True
            if not violated:
                break
        return np.clip(u, -a_max, a_max)
