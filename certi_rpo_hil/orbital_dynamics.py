"""
6-DOF Coupled Relative Orbital & Attitude Dynamics in the LVLH (Hill) Frame.

Includes:
1. Exact nonlinear two-body + J2-perturbed relative translational dynamics in the
   Chief-centered Local-Vertical Local-Horizontal (LVLH: x=R-bar, y=V-bar, z=H-bar) frame.
2. Closed-form Hill-Clohessy-Wiltshire (HCW) state-transition matrices (Phi, Gamma)
   for high-rate onboard MEKF propagation and 2-orbit Passive Abort Free-Drift verification.
3. 6-DOF Euler-Poinsot rigid-body rotational dynamics for both Chaser and a tumbling
   non-cooperative Target spacecraft, including Solar Radiation Pressure (SRP) and
   RCS thruster plume impingement disturbances.
4. Realistic Thruster Fault Injection (stuck-open bias + degraded actuator authority).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

# WGS-84 Earth Astrodynamic Constants
MU_EARTH: float = 3.986004418e14      # Earth gravitational parameter [m^3/s^2]
R_EARTH: float = 6378137.0            # Earth equatorial radius [m]
J2_EARTH: float = 1.08262668e-3       # Earth second zonal harmonic [-]


@dataclass
class OrbitConfig:
    """Configuration for Chief reference orbit and spacecraft mass/inertia properties."""

    altitude_m: float = 550_000.0
    inclination_rad: float = float(np.deg2rad(51.6))
    arg_latitude_0_rad: float = 0.0
    chaser_mass_kg: float = 180.0
    chaser_inertia_kgm2: Tuple[float, float, float] = (18.5, 22.0, 25.0)
    target_inertia_kgm2: Tuple[float, float, float] = (120.0, 145.0, 160.0)
    max_thrust_n: float = 14.4  # Gives max control acceleration a_max = 0.08 m/s^2
    srp_accel_ms2: float = 4.5e-7
    plume_impingement_coeff: float = 1.5e-4
    docking_port_pos_m: Tuple[float, float, float] = (0.0, -2.2, 0.0)

    @property
    def semi_major_axis_m(self) -> float:
        return R_EARTH + self.altitude_m

    @property
    def mean_motion_rad_s(self) -> float:
        return float(np.sqrt(MU_EARTH / (self.semi_major_axis_m ** 3)))

    @property
    def orbital_period_s(self) -> float:
        return float(2.0 * np.pi / self.mean_motion_rad_s)

    @property
    def max_accel_ms2(self) -> float:
        return float(self.max_thrust_n / self.chaser_mass_kg)


@dataclass
class ThrusterFaultConfig:
    """Configuration for stuck-open and degraded RCS thruster faults."""

    enabled: bool = True
    start_time_s: float = 70.0
    end_time_s: float = 115.0
    # Stuck-open parasitic acceleration [ax, ay, az] in m/s^2 pushing toward target (+y) and outside cone (+x)
    stuck_bias_accel_ms2: Tuple[float, float, float] = (0.018, 0.032, -0.010)
    # Degraded actuator effectiveness per axis in (0, 1]
    effectiveness_scale: Tuple[float, float, float] = (0.78, 0.72, 0.85)

    def is_active(self, t_s: float) -> bool:
        return bool(self.enabled and (self.start_time_s <= t_s <= self.end_time_s))

    def get_fault_effects(self, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (effectiveness_diag_vec, stuck_bias_accel_vec) at time t_s."""
        if self.is_active(t_s):
            return (
                np.asarray(self.effectiveness_scale, dtype=float),
                np.asarray(self.stuck_bias_accel_ms2, dtype=float),
            )
        return np.ones(3, dtype=float), np.zeros(3, dtype=float)


@dataclass
class SixDOFState:
    """Full 6-DOF relative translational and attitude state of Chaser & Target."""

    # Relative translational state in LVLH frame: [x, y, z, vx, vy, vz] (m, m/s)
    x_rel: np.ndarray
    # Chaser attitude quaternion [qw, qx, qy, qz] (LVLH to Chaser body)
    q_chaser: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    # Chaser body angular velocity [wx, wy, wz] (rad/s)
    w_chaser: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))
    # Target attitude quaternion [qw, qx, qy, qz] (LVLH to Target body)
    q_target: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    # Target tumbling angular velocity [wx, wy, wz] (rad/s)
    w_target: np.ndarray = field(default_factory=lambda: np.array([0.004, -0.003, 0.008], dtype=float))

    @property
    def pos_m(self) -> np.ndarray:
        return self.x_rel[:3]

    @property
    def vel_ms(self) -> np.ndarray:
        return self.x_rel[3:6]

    def copy(self) -> "SixDOFState":
        return SixDOFState(
            x_rel=self.x_rel.copy(),
            q_chaser=self.q_chaser.copy(),
            w_chaser=self.w_chaser.copy(),
            q_target=self.q_target.copy(),
            w_target=self.w_target.copy(),
        )


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize scalar-first quaternion [qw, qx, qy, qz] with positive scalar convention."""
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q_norm = q / norm
    if q_norm[0] < 0.0:
        q_norm = -q_norm
    return q_norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions [qw, qx, qy, qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of scalar-first quaternion [qw, qx, qy, qz]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_error_angle_deg(q_est: np.ndarray, q_true: np.ndarray) -> float:
    """Geodesic attitude error angle in degrees between two unit quaternions."""
    dq = quat_multiply(quat_conjugate(normalize_quaternion(q_true)), normalize_quaternion(q_est))
    qw = float(np.clip(abs(dq[0]), 0.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(qw)))


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Time derivative dq/dt = 0.5 * q (x) [0, omega]."""
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
    return 0.5 * quat_multiply(q, omega_quat)


def hcw_state_transition_matrix(n: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact closed-form Hill-Clohessy-Wiltshire (HCW) discrete-time state-transition
    matrix Phi(dt) in R^(6x6) and zero-order-hold input matrix Gamma(dt) in R^(6x3).

    State vector: x = [x, y, z, vx, vy, vz]^T in LVLH frame.
    Control input: u = [ax, ay, az]^T in m/s^2.
    """
    nt = n * dt
    s = float(np.sin(nt))
    c = float(np.cos(nt))

    Phi = np.array(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [6.0 * n * (c - 1.0), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )

    n2 = n * n
    Gamma = np.array(
        [
            [(1.0 - c) / n2, 2.0 * (nt - s) / n2, 0.0],
            [-2.0 * (nt - s) / n2, 4.0 * (1.0 - c) / n2 - 1.5 * dt * dt, 0.0],
            [0.0, 0.0, (1.0 - c) / n2],
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [-2.0 * (1.0 - c) / n, 4.0 * s / n - 3.0 * dt, 0.0],
            [0.0, 0.0, s / n],
        ],
        dtype=float,
    )
    return Phi, Gamma


def propagate_hcw_free_drift(
    x0: np.ndarray,
    n: float,
    duration_s: float,
    num_steps: int = 120,
) -> np.ndarray:
    """
    Vectorized analytical propagation of the natural HCW free-drift trajectory (u = 0)
    over [0, duration_s] for Passive Abort Safety verification.

    Returns array of shape (num_steps, 6).
    """
    t_grid = np.linspace(0.0, duration_s, num_steps, dtype=float)
    nt = n * t_grid
    s = np.sin(nt)
    c = np.cos(nt)

    x_init, y_init, z_init, vx_init, vy_init, vz_init = x0

    x_t = (4.0 - 3.0 * c) * x_init + (s / n) * vx_init + (2.0 * (1.0 - c) / n) * vy_init
    y_t = (
        6.0 * (s - nt) * x_init
        + y_init
        - (2.0 * (1.0 - c) / n) * vx_init
        + ((4.0 * s - 3.0 * nt) / n) * vy_init
    )
    z_t = c * z_init + (s / n) * vz_init

    vx_t = 3.0 * n * s * x_init + c * vx_init + 2.0 * s * vy_init
    vy_t = 6.0 * n * (c - 1.0) * x_init - 2.0 * s * vx_init + (4.0 * c - 3.0) * vy_init
    vz_t = -n * s * z_init + c * vz_init

    return np.column_stack([x_t, y_t, z_t, vx_t, vy_t, vz_t])


class NonlinearJ2SixDOFDynamics:
    """
    High-fidelity 6-DOF nonlinear Chief-Deputy relative orbital and attitude dynamics
    with exact two-body differential gravity, J2 zonal harmonic perturbations,
    solar radiation pressure (SRP), thruster plume impingement, and RCS fault injection.
    """

    def __init__(
        self,
        orbit_cfg: Optional[OrbitConfig] = None,
        fault_cfg: Optional[ThrusterFaultConfig] = None,
    ) -> None:
        self.orbit_cfg = orbit_cfg or OrbitConfig()
        self.fault_cfg = fault_cfg or ThrusterFaultConfig(enabled=False)
        self.j_chaser = np.asarray(self.orbit_cfg.chaser_inertia_kgm2, dtype=float)
        self.j_target = np.asarray(self.orbit_cfg.target_inertia_kgm2, dtype=float)

    def compute_translational_accel(
        self,
        x_rel: np.ndarray,
        u_applied: np.ndarray,
        t_s: float,
    ) -> np.ndarray:
        """
        Compute exact nonlinear two-body + Schweighart-Sedwick J2 relative acceleration
        plus SRP and plume impingement in the LVLH frame.
        """
        r0 = self.orbit_cfg.semi_major_axis_m
        n = self.orbit_cfg.mean_motion_rad_s
        inc = self.orbit_cfg.inclination_rad
        theta = self.orbit_cfg.arg_latitude_0_rad + n * t_s

        x, y, z, vx, vy, vz = x_rel

        # Exact nonlinear two-body relative acceleration in LVLH frame
        rc_vec = np.array([r0 + x, y, z], dtype=float)
        rc_norm = float(np.linalg.norm(rc_vec))
        mu_rc3 = MU_EARTH / (rc_norm ** 3)

        ax_2b = 2.0 * n * vy + (n ** 2) * (r0 + x) - mu_rc3 * (r0 + x)
        ay_2b = -2.0 * n * vx + (n ** 2) * y - mu_rc3 * y
        az_2b = -mu_rc3 * z

        # J2 differential perturbation (Schweighart-Sedwick / differential J2 gradient)
        j2_factor = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / (r0 ** 5)
        sin_i = float(np.sin(inc))
        sin_th = float(np.sin(theta))
        cos_th = float(np.cos(theta))
        sin2_i_sin2_th = (sin_i ** 2) * (sin_th ** 2)

        ax_j2 = j2_factor * (
            (4.0 - 12.0 * sin2_i_sin2_th) * x
            - (8.0 * (sin_i ** 2) * sin_th * cos_th) * y
            - (8.0 * sin_i * float(np.cos(inc)) * sin_th) * z
        )
        ay_j2 = j2_factor * (
            -(8.0 * (sin_i ** 2) * sin_th * cos_th) * x
            - (1.0 - 7.0 * sin2_i_sin2_th) * y
            + (2.0 * sin_i * float(np.cos(inc)) * cos_th) * z
        )
        az_j2 = j2_factor * (
            -(8.0 * sin_i * float(np.cos(inc)) * sin_th) * x
            + (2.0 * sin_i * float(np.cos(inc)) * cos_th) * y
            - (3.0 - 5.0 * sin2_i_sin2_th) * z
        )

        # Differential Solar Radiation Pressure (SRP)
        sun_dir = np.array(
            [np.cos(0.015 * t_s), np.sin(0.015 * t_s), 0.25 * np.sin(0.008 * t_s)],
            dtype=float,
        )
        sun_dir /= np.linalg.norm(sun_dir)
        a_srp = self.orbit_cfg.srp_accel_ms2 * sun_dir

        # Plume impingement back-reaction when close to target (< 15 m)
        dist = max(float(np.linalg.norm(x_rel[:3])), 1.0)
        a_plume = -self.orbit_cfg.plume_impingement_coeff * u_applied / (dist ** 2)

        return (
            np.array([ax_2b + ax_j2, ay_2b + ay_j2, az_2b + az_j2], dtype=float)
            + a_srp
            + a_plume
            + u_applied
        )

    def euler_poinsot_rhs(
        self,
        omega: np.ndarray,
        inertia_diag: np.ndarray,
        torque: np.ndarray,
    ) -> np.ndarray:
        """Euler's rigid body rotational equation: J * d(omega)/dt = -omega x (J * omega) + torque."""
        j_omega = inertia_diag * omega
        gyro = np.cross(omega, j_omega)
        return (torque - gyro) / inertia_diag

    def apply_thruster_model(
        self,
        u_cmd_ms2: np.ndarray,
        t_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clip commanded acceleration to physical saturation limits and inject any active
        stuck-open / degraded thruster fault.

        Returns (u_actual_ms2, fault_perturbation_ms2).
        """
        a_max = self.orbit_cfg.max_accel_ms2
        u_clipped = np.clip(np.asarray(u_cmd_ms2, dtype=float), -a_max, a_max)
        eff_scale, stuck_bias = self.fault_cfg.get_fault_effects(t_s)
        u_actual = eff_scale * u_clipped + stuck_bias
        fault_delta = u_actual - u_clipped
        return u_actual, fault_delta

    def step(
        self,
        state: SixDOFState,
        u_cmd_ms2: np.ndarray,
        t_s: float,
        dt_s: float,
        chaser_torque_nm: Optional[np.ndarray] = None,
    ) -> Tuple[SixDOFState, np.ndarray, np.ndarray]:
        """
        Advance the 6-DOF nonlinear dynamics by one step dt_s using 4th-order Runge-Kutta (RK4).

        Returns:
            (next_state, u_actual_ms2, fault_delta_ms2)
        """
        u_actual, fault_delta = self.apply_thruster_model(u_cmd_ms2, t_s)

        # Default PD attitude pointing controller keeping Chaser aligned with LVLH docking axis
        if chaser_torque_nm is None:
            q_vec = state.q_chaser[1:4] * np.sign(state.q_chaser[0] if state.q_chaser[0] != 0 else 1.0)
            chaser_torque_nm = -2.4 * q_vec - 3.8 * state.w_chaser

        # Plume torque on Target when Chaser fires thrusters near Target
        dist = max(float(np.linalg.norm(state.pos_m)), 1.5)
        target_plume_torque = 0.002 * np.cross(state.pos_m / dist, u_actual) / (dist ** 2)

        def trans_deriv(x_vec: np.ndarray, tau: float) -> np.ndarray:
            acc = self.compute_translational_accel(x_vec, u_actual, tau)
            return np.concatenate([x_vec[3:6], acc])

        # RK4 for 6-DOF translational state
        x0 = state.x_rel
        k1 = trans_deriv(x0, t_s)
        k2 = trans_deriv(x0 + 0.5 * dt_s * k1, t_s + 0.5 * dt_s)
        k3 = trans_deriv(x0 + 0.5 * dt_s * k2, t_s + 0.5 * dt_s)
        k4 = trans_deriv(x0 + dt_s * k3, t_s + dt_s)
        x_next = x0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # RK4 for Chaser & Target attitude + angular rates
        def att_step(
            q_in: np.ndarray,
            w_in: np.ndarray,
            j_diag: np.ndarray,
            trq: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            kq1 = quaternion_derivative(q_in, w_in)
            kw1 = self.euler_poinsot_rhs(w_in, j_diag, trq)

            kq2 = quaternion_derivative(q_in + 0.5 * dt_s * kq1, w_in + 0.5 * dt_s * kw1)
            kw2 = self.euler_poinsot_rhs(w_in + 0.5 * dt_s * kw1, j_diag, trq)

            kq3 = quaternion_derivative(q_in + 0.5 * dt_s * kq2, w_in + 0.5 * dt_s * kw2)
            kw3 = self.euler_poinsot_rhs(w_in + 0.5 * dt_s * kw2, j_diag, trq)

            kq4 = quaternion_derivative(q_in + dt_s * kq3, w_in + dt_s * kw3)
            kw4 = self.euler_poinsot_rhs(w_in + dt_s * kw3, j_diag, trq)

            q_out = normalize_quaternion(q_in + (dt_s / 6.0) * (kq1 + 2.0 * kq2 + 2.0 * kq3 + kq4))
            w_out = w_in + (dt_s / 6.0) * (kw1 + 2.0 * kw2 + 2.0 * kw3 + kw4)
            return q_out, w_out

        q_c_next, w_c_next = att_step(state.q_chaser, state.w_chaser, self.j_chaser, chaser_torque_nm)
        q_t_next, w_t_next = att_step(state.q_target, state.w_target, self.j_target, target_plume_torque)

        next_state = SixDOFState(
            x_rel=x_next,
            q_chaser=q_c_next,
            w_chaser=w_c_next,
            q_target=q_t_next,
            w_target=w_t_next,
        )
        return next_state, u_actual, fault_delta
