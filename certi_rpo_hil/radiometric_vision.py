"""
Physically Based Space Optical Navigation & Flash LiDAR Sensor Model calibrated to
Stanford SLAB SPEED+ HIL error characteristics, paired with a Multiplicative Extended
Kalman Filter (MEKF) with Mahalanobis Innovation Outlier Gating and Disturbance Bias Estimation.

Models:
- Solar phase-angle illumination geometry and Cook-Torrance specular BRDF glare spikes
  off crumpled Kapton Multi-Layer Insulation (MLI) blankets.
- Orbital eclipse entry/exit transitions (penumbra auto-exposure transients + umbra LiDAR).
- CMOS rolling-shutter and range-dependent shot noise.
- 9-state translational + disturbance-bias EKF coupled with a 6-DOF Multiplicative EKF (MEKF)
  for target attitude & angular velocity estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from certi_rpo_hil.orbital_dynamics import (
    SixDOFState,
    hcw_state_transition_matrix,
    normalize_quaternion,
    quat_conjugate,
    quat_error_angle_deg,
    quat_multiply,
    quaternion_derivative,
)


@dataclass
class VisionConfig:
    """Configuration for SPEED+ Radiometric Optical Camera, Flash LiDAR, and MEKF."""

    # Nominal optical position noise (1-sigma at 10 m range) [m]
    nominal_pos_sigma_m: float = 0.015
    # Range-quadratic triangulation noise factor [m / m^2]
    range_quadratic_coeff: float = 2.0e-5
    # Nominal LiDAR range/velocity backup noise [m, m/s]
    lidar_pos_sigma_m: float = 0.022
    lidar_vel_sigma_ms: float = 0.006
    # Attitude measurement noise (1-sigma) [rad] and rate noise [rad/s]
    nominal_att_sigma_rad: float = float(np.deg2rad(0.35))
    nominal_rate_sigma_rad_s: float = float(np.deg2rad(0.08))

    # Kapton MLI Specular Glare Window [s]
    glare_start_s: float = 38.0
    glare_end_s: float = 66.0
    # Specular BRDF glare outlier amplitude [m] in [x, y, z]
    glare_pos_bias_m: Tuple[float, float, float] = (3.6, 4.2, -2.4)
    glare_pos_jitter_sigma_m: float = 0.85

    # Orbital Eclipse Window [s]
    eclipse_start_s: float = 122.0
    eclipse_end_s: float = 158.0
    eclipse_transition_halfwidth_s: float = 2.5
    eclipse_transient_bias_m: Tuple[float, float, float] = (-1.8, 2.1, 1.4)

    # Mahalanobis chi-square gating threshold (3 DOF at p=0.999 is 16.27)
    mahalanobis_gate_threshold: float = 16.27


@dataclass
class OpticalMeasurement:
    """Single-frame telemetry from the Radiometric Optical Camera + Flash LiDAR suite."""

    t_s: float
    z_pos_optical_m: np.ndarray       # Optical PnP / CNN position measurement [x, y, z] (corrupted under glare)
    z_pos_lidar_m: np.ndarray         # Active Flash LiDAR position backup [x, y, z]
    z_vel_lidar_ms: np.ndarray        # Doppler / range-rate velocity measurement [vx, vy, vz]
    z_q_target: np.ndarray            # Measured Target quaternion [qw, qx, qy, qz]
    z_w_target: np.ndarray            # Measured Target angular velocity [wx, wy, wz]
    solar_phase_angle_deg: float      # Angle between Sun vector and Chaser-Target LOS [deg]
    glare_active: bool                # True when in Kapton MLI specular glare cone
    eclipse_active: bool              # True when in orbital umbra
    eclipse_transition: bool          # True during penumbra entry/exit auto-exposure transient


class RadiometricOpticalSensor:
    """
    Simulates a Spaceborne Optical Navigation Camera + Flash LiDAR sensor suite
    exhibiting Stanford SLAB SPEED+ lightbox/sunlamp domain-gap failure modes:
    specular Kapton MLI glare spikes, penumbra auto-exposure blooming, and shot noise.
    """

    def __init__(self, cfg: Optional[VisionConfig] = None, seed: int = 2026) -> None:
        self.cfg = cfg or VisionConfig()
        self.rng = np.random.default_rng(seed)

    def compute_solar_phase_angle_deg(self, pos_m: np.ndarray, t_s: float) -> float:
        """
        Compute solar phase angle [deg] between Chaser-to-Target line-of-sight
        and illumination vector, sweeping through the specular reflection regime
        during [glare_start_s, glare_end_s].
        """
        if self.cfg.glare_start_s <= t_s <= self.cfg.glare_end_s:
            mid = 0.5 * (self.cfg.glare_start_s + self.cfg.glare_end_s)
            half = max(0.5 * (self.cfg.glare_end_s - self.cfg.glare_start_s), 1.0)
            norm_dt = (t_s - mid) / half
            # Sweeps down to ~4 deg specular reflection peak at mid-window
            return float(4.0 + 14.0 * (norm_dt ** 2))
        los = -pos_m / max(float(np.linalg.norm(pos_m)), 1e-6)
        sun_vec = np.array(
            [np.cos(0.015 * t_s), np.sin(0.015 * t_s), 0.25 * np.sin(0.008 * t_s)],
            dtype=float,
        )
        sun_vec /= np.linalg.norm(sun_vec)
        cos_ang = float(np.clip(np.dot(los, sun_vec), -1.0, 1.0))
        angle = float(np.rad2deg(np.arccos(cos_ang)))
        return max(angle, 22.0)

    def measure(self, true_state: SixDOFState, t_s: float) -> OpticalMeasurement:
        """Generate a noisy 6-DOF optical + LiDAR measurement at time t_s."""
        pos = true_state.pos_m
        vel = true_state.vel_ms
        dist = float(np.linalg.norm(pos))

        phase_deg = self.compute_solar_phase_angle_deg(pos, t_s)
        glare_active = bool(self.cfg.glare_start_s <= t_s <= self.cfg.glare_end_s)
        eclipse_active = bool(self.cfg.eclipse_start_s <= t_s <= self.cfg.eclipse_end_s)
        eclipse_transition = bool(
            abs(t_s - self.cfg.eclipse_start_s) <= self.cfg.eclipse_transition_halfwidth_s
            or abs(t_s - self.cfg.eclipse_end_s) <= self.cfg.eclipse_transition_halfwidth_s
        )

        # Range-dependent CMOS shot noise + rolling shutter blur
        vel_norm = float(np.linalg.norm(vel))
        sigma_opt = (
            self.cfg.nominal_pos_sigma_m
            + self.cfg.range_quadratic_coeff * (dist ** 2)
            + 0.008 * vel_norm
        )
        z_opt = pos + self.rng.normal(0.0, sigma_opt, size=3)

        # Inject SPEED+ Kapton MLI Specular BRDF Glare Spike
        if glare_active:
            mid = 0.5 * (self.cfg.glare_start_s + self.cfg.glare_end_s)
            half = max(0.5 * (self.cfg.glare_end_s - self.cfg.glare_start_s), 1.0)
            bell = float(np.exp(-1.5 * (((t_s - mid) / half) ** 2)))
            glare_bias = bell * np.asarray(self.cfg.glare_pos_bias_m, dtype=float)
            glare_jitter = self.rng.normal(0.0, self.cfg.glare_pos_jitter_sigma_m, size=3)
            z_opt = z_opt + glare_bias + glare_jitter

        # Inject Penumbra Auto-Exposure Blooming Transient at Eclipse Entry/Exit
        if eclipse_transition:
            z_opt = z_opt + np.asarray(self.cfg.eclipse_transient_bias_m, dtype=float)

        # Active Flash LiDAR measurement (immune to passive solar glare & umbra darkness)
        z_lidar_pos = pos + self.rng.normal(0.0, self.cfg.lidar_pos_sigma_m, size=3)
        z_lidar_vel = vel + self.rng.normal(0.0, self.cfg.lidar_vel_sigma_ms, size=3)

        # Target attitude & rate measurement
        att_noise_sigma = self.cfg.nominal_att_sigma_rad * (3.5 if glare_active else 1.0)
        dtheta = self.rng.normal(0.0, att_noise_sigma, size=3)
        dq = normalize_quaternion(np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]]))
        z_q_t = normalize_quaternion(quat_multiply(true_state.q_target, dq))
        z_w_t = true_state.w_target + self.rng.normal(0.0, self.cfg.nominal_rate_sigma_rad_s, size=3)

        return OpticalMeasurement(
            t_s=t_s,
            z_pos_optical_m=z_opt,
            z_pos_lidar_m=z_lidar_pos,
            z_vel_lidar_ms=z_lidar_vel,
            z_q_target=z_q_t,
            z_w_target=z_w_t,
            solar_phase_angle_deg=phase_deg,
            glare_active=glare_active,
            eclipse_active=eclipse_active,
            eclipse_transition=eclipse_transition,
        )


@dataclass
class MEKFStepDiagnostics:
    """Per-step diagnostics from the MEKF + Mahalanobis innovation gate."""

    x_hat: np.ndarray                 # Estimated 6-DOF translational state [x, y, z, vx, vy, vz]
    d_hat_ms2: np.ndarray             # Estimated unmodeled/thruster-fault acceleration bias [dx, dy, dz]
    q_target_hat: np.ndarray          # Estimated target quaternion [qw, qx, qy, qz]
    w_target_hat: np.ndarray          # Estimated target angular velocity [wx, wy, wz]
    mahalanobis_d2: float             # Optical innovation Mahalanobis squared distance
    outlier_rejected: bool            # True if optical frame was gated out due to glare/transient
    pos_error_m: float                # Euclidean position estimation error ||r_hat - r_true||_2
    vel_error_ms: float               # Euclidean velocity estimation error ||v_hat - v_true||_2
    att_error_deg: float              # Target attitude geodesic error [deg]


class MultiplicativeEKF:
    """
    9-State Translational + Thruster-Fault Disturbance Bias Kalman Filter coupled with
    a 6-State Multiplicative Extended Kalman Filter (MEKF) for Target 6-DOF Pose Estimation.

    Includes Chi-Square Mahalanobis Innovation Gating (`enable_mahalanobis_gating=True`)
    to reject Kapton MLI specular solar glare spikes and eclipse transition transients.
    """

    def __init__(
        self,
        mean_motion_rad_s: float,
        initial_state: SixDOFState,
        vision_cfg: Optional[VisionConfig] = None,
        enable_mahalanobis_gating: bool = True,
    ) -> None:
        self.n = mean_motion_rad_s
        self.cfg = vision_cfg or VisionConfig()
        self.enable_gating = enable_mahalanobis_gating

        # 9-state vector: [x, y, z, vx, vy, vz, dx, dy, dz]
        # where d = [dx, dy, dz] estimates stuck-open / unmodeled acceleration bias
        self.state_9 = np.zeros(9, dtype=float)
        self.state_9[:6] = initial_state.x_rel.copy()

        self.P_9 = np.diag(
            [0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 1e-4, 1e-4, 1e-4]
        ).astype(float)

        # Process noise covariance Q_9
        self.Q_9 = np.diag(
            [1e-6, 1e-6, 1e-6, 2e-5, 2e-5, 2e-5, 8e-5, 8e-5, 8e-5]
        ).astype(float)

        # Target attitude MEKF state: nominal quaternion q_hat (4,) + rate w_hat (3,)
        self.q_target_hat = initial_state.q_target.copy()
        self.w_target_hat = initial_state.w_target.copy()
        self.P_att = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4]).astype(float)
        self.Q_att = np.diag([2e-5, 2e-5, 2e-5, 1e-5, 1e-5, 1e-5]).astype(float)

        self.total_frames = 0
        self.rejected_glare_frames = 0

    @property
    def x_hat(self) -> np.ndarray:
        return self.state_9[:6].copy()

    @property
    def disturbance_hat_ms2(self) -> np.ndarray:
        return self.state_9[6:9].copy()

    def predict(self, u_cmd_ms2: np.ndarray, dt_s: float, j_target: np.ndarray) -> None:
        """Propagate 9-state translational EKF and 6-state attitude MEKF over dt_s."""
        Phi_6, Gamma_6 = hcw_state_transition_matrix(self.n, dt_s)

        # Build 9x9 augmented transition matrix F_9 (disturbance d acts as additive accel)
        F_9 = np.eye(9, dtype=float)
        F_9[:6, :6] = Phi_6
        F_9[:6, 6:9] = Gamma_6

        B_9 = np.zeros((9, 3), dtype=float)
        B_9[:6, :] = Gamma_6

        self.state_9 = F_9 @ self.state_9 + B_9 @ np.asarray(u_cmd_ms2, dtype=float)
        self.P_9 = F_9 @ self.P_9 @ F_9.T + self.Q_9

        # Attitude propagation (RK2 midpoint quaternion + Euler-Poinsot rate)
        w = self.w_target_hat
        jw = j_target * w
        dw = -np.cross(w, jw) / j_target
        w_mid = w + 0.5 * dt_s * dw
        dq_dt = quaternion_derivative(self.q_target_hat, w_mid)
        self.q_target_hat = normalize_quaternion(self.q_target_hat + dt_s * dq_dt)
        self.w_target_hat = w + dt_s * dw
        self.P_att = self.P_att + self.Q_att

    def update(
        self,
        meas: OpticalMeasurement,
        true_state: SixDOFState,
    ) -> MEKFStepDiagnostics:
        """
        Perform measurement update with Mahalanobis innovation gating on the optical channel
        and fuse active Flash LiDAR position/velocity + attitude measurements.
        """
        self.total_frames += 1

        # 1. Evaluate Optical Position Innovation & Mahalanobis Distance
        H_pos = np.zeros((3, 9), dtype=float)
        H_pos[:3, :3] = np.eye(3, dtype=float)
        R_opt = (self.cfg.nominal_pos_sigma_m ** 2) * np.eye(3, dtype=float)

        nu_opt = meas.z_pos_optical_m - self.state_9[:3]
        S_opt = H_pos @ self.P_9 @ H_pos.T + R_opt
        S_opt_inv = np.linalg.inv(S_opt)
        mahalanobis_d2 = float(nu_opt.T @ S_opt_inv @ nu_opt)

        outlier_rejected = False
        if self.enable_gating and (
            mahalanobis_d2 > self.cfg.mahalanobis_gate_threshold or meas.eclipse_active
        ):
            outlier_rejected = True
            self.rejected_glare_frames += 1
        else:
            # Fuse optical position measurement (if gating is disabled, glare corrupts state!)
            K_opt = self.P_9 @ H_pos.T @ S_opt_inv
            self.state_9 = self.state_9 + K_opt @ nu_opt
            I_KH = np.eye(9, dtype=float) - K_opt @ H_pos
            self.P_9 = I_KH @ self.P_9 @ I_KH.T + K_opt @ R_opt @ K_opt.T

        # 2. Fuse Flash LiDAR Position + Doppler Velocity (if gating is enabled, full weight;
        # if gating is disabled, naive filter relies 90% on corrupted optical camera)
        H_lidar = np.zeros((6, 9), dtype=float)
        H_lidar[:6, :6] = np.eye(6, dtype=float)
        lidar_scale = 1.0 if self.enable_gating else 25.0
        R_lidar = np.diag(
            [
                lidar_scale * (self.cfg.lidar_pos_sigma_m ** 2),
                lidar_scale * (self.cfg.lidar_pos_sigma_m ** 2),
                lidar_scale * (self.cfg.lidar_pos_sigma_m ** 2),
                (self.cfg.lidar_vel_sigma_ms ** 2),
                (self.cfg.lidar_vel_sigma_ms ** 2),
                (self.cfg.lidar_vel_sigma_ms ** 2),
            ]
        ).astype(float)

        z_lidar = np.concatenate([meas.z_pos_lidar_m, meas.z_vel_lidar_ms])
        nu_lidar = z_lidar - self.state_9[:6]
        S_lidar = H_lidar @ self.P_9 @ H_lidar.T + R_lidar
        K_lidar = self.P_9 @ H_lidar.T @ np.linalg.inv(S_lidar)
        self.state_9 = self.state_9 + K_lidar @ nu_lidar
        I_KH_l = np.eye(9, dtype=float) - K_lidar @ H_lidar
        self.P_9 = I_KH_l @ self.P_9 @ I_KH_l.T + K_lidar @ R_lidar @ K_lidar.T

        # 3. Multiplicative EKF Update for Target Attitude & Angular Rate
        dq_meas = quat_multiply(quat_conjugate(self.q_target_hat), meas.z_q_target)
        if dq_meas[0] < 0.0:
            dq_meas = -dq_meas
        dtheta_meas = 2.0 * dq_meas[1:4]
        dw_meas = meas.z_w_target - self.w_target_hat
        nu_att = np.concatenate([dtheta_meas, dw_meas])

        att_r_scale = 20.0 if (self.enable_gating and outlier_rejected) else 1.0
        R_att = np.diag(
            [
                att_r_scale * (self.cfg.nominal_att_sigma_rad ** 2),
                att_r_scale * (self.cfg.nominal_att_sigma_rad ** 2),
                att_r_scale * (self.cfg.nominal_att_sigma_rad ** 2),
                (self.cfg.nominal_rate_sigma_rad_s ** 2),
                (self.cfg.nominal_rate_sigma_rad_s ** 2),
                (self.cfg.nominal_rate_sigma_rad_s ** 2),
            ]
        ).astype(float)

        S_att = self.P_att + R_att
        K_att = self.P_att @ np.linalg.inv(S_att)
        dx_att = K_att @ nu_att
        dtheta_corr = dx_att[:3]
        dw_corr = dx_att[3:6]

        dq_corr = normalize_quaternion(
            np.array([1.0, 0.5 * dtheta_corr[0], 0.5 * dtheta_corr[1], 0.5 * dtheta_corr[2]])
        )
        self.q_target_hat = normalize_quaternion(quat_multiply(self.q_target_hat, dq_corr))
        self.w_target_hat = self.w_target_hat + dw_corr
        self.P_att = (np.eye(6, dtype=float) - K_att) @ self.P_att

        pos_err = float(np.linalg.norm(self.state_9[:3] - true_state.pos_m))
        vel_err = float(np.linalg.norm(self.state_9[3:6] - true_state.vel_ms))
        att_err = quat_error_angle_deg(self.q_target_hat, true_state.q_target)

        return MEKFStepDiagnostics(
            x_hat=self.x_hat,
            d_hat_ms2=self.disturbance_hat_ms2,
            q_target_hat=self.q_target_hat.copy(),
            w_target_hat=self.w_target_hat.copy(),
            mahalanobis_d2=mahalanobis_d2,
            outlier_rejected=outlier_rejected,
            pos_error_m=pos_err,
            vel_error_ms=vel_err,
            att_error_deg=att_err,
        )
