"""
Comprehensive Unit & Integration Test Suite for CERTI-RPO-HIL (16 Tests).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from certi_rpo_hil.orbital_dynamics import (
    NonlinearJ2SixDOFDynamics,
    OrbitConfig,
    SixDOFState,
    ThrusterFaultConfig,
    hcw_state_transition_matrix,
    normalize_quaternion,
    propagate_hcw_free_drift,
    quat_error_angle_deg,
    quat_multiply,
)
from certi_rpo_hil.radiometric_vision import (
    MultiplicativeEKF,
    RadiometricOpticalSensor,
    VisionConfig,
)
from certi_rpo_hil.safety_shield import (
    ASTM3269CBFShield,
    NominalRPODGuidance,
    SafetyConfig,
)
from certi_rpo_hil.hil_bridge import (
    APID_ACTUATOR_COMMAND,
    APID_SENSOR_TELEMETRY,
    CCSDSSpacePacketBridge,
    CCSDSTelecommandPayload,
    CCSDSTelemetryPayload,
    HILBridgeConfig,
    crc16_ccitt,
    pack_ccsds_packet,
    unpack_ccsds_packet,
)
from certi_rpo_hil.verification import run_rpo_campaign
from certi_rpo_hil.reporting import generate_all_reports


def test_01_quaternion_algebra_invariants() -> None:
    q1 = normalize_quaternion(np.array([0.8, 0.2, -0.4, 0.3]))
    assert np.isclose(np.linalg.norm(q1), 1.0, atol=1e-12)
    assert quat_error_angle_deg(q1, q1) == pytest.approx(0.0, abs=1e-7)
    q_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    prod = quat_multiply(q1, q_inv)
    assert np.allclose(prod, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)


def test_02_hcw_state_transition_identity_and_symplectic() -> None:
    cfg = OrbitConfig()
    Phi0, Gamma0 = hcw_state_transition_matrix(cfg.mean_motion_rad_s, 0.0)
    assert np.allclose(Phi0, np.eye(6), atol=1e-12)
    assert np.allclose(Gamma0, np.zeros((6, 3)), atol=1e-12)

    Phi, _ = hcw_state_transition_matrix(cfg.mean_motion_rad_s, 10.0)
    assert np.isclose(np.linalg.det(Phi), 1.0, atol=1e-9)


def test_03_hcw_free_drift_consistency_with_stm() -> None:
    cfg = OrbitConfig()
    x0 = np.array([2.5, -25.0, 1.0, 0.01, 0.05, -0.02])
    duration = 120.0
    drift = propagate_hcw_free_drift(x0, cfg.mean_motion_rad_s, duration, num_steps=25)
    Phi, _ = hcw_state_transition_matrix(cfg.mean_motion_rad_s, duration)
    assert np.allclose(drift[-1], Phi @ x0, atol=1e-9)


def test_04_nonlinear_j2_6dof_dynamics_step() -> None:
    cfg = OrbitConfig()
    fault = ThrusterFaultConfig(enabled=False)
    dyn = NonlinearJ2SixDOFDynamics(cfg, fault)
    state = SixDOFState(x_rel=np.array([0.5, -20.0, -0.3, 0.0, 0.1, 0.0]))
    next_state, u_act, f_delta = dyn.step(state, np.array([0.01, -0.02, 0.0]), t_s=10.0, dt_s=0.5)
    assert next_state.x_rel.shape == (6,)
    assert np.isclose(np.linalg.norm(next_state.q_chaser), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(next_state.q_target), 1.0, atol=1e-12)
    assert np.allclose(f_delta, np.zeros(3))
    assert np.allclose(u_act, np.array([0.01, -0.02, 0.0]))


def test_05_thruster_fault_injection_window() -> None:
    cfg = OrbitConfig()
    fault = ThrusterFaultConfig(enabled=True, start_time_s=70.0, end_time_s=115.0)
    dyn = NonlinearJ2SixDOFDynamics(cfg, fault)
    u_cmd = np.array([0.02, -0.02, 0.01])

    u_nominal, delta_nom = dyn.apply_thruster_model(u_cmd, t_s=20.0)
    assert np.allclose(u_nominal, u_cmd)
    assert np.allclose(delta_nom, np.zeros(3))

    u_faulted, delta_flt = dyn.apply_thruster_model(u_cmd, t_s=85.0)
    assert not np.allclose(u_faulted, u_cmd)
    assert np.linalg.norm(delta_flt) > 0.02


def test_06_radiometric_sensor_glare_and_eclipse_modes() -> None:
    vcfg = VisionConfig()
    sensor = RadiometricOpticalSensor(vcfg, seed=42)
    state = SixDOFState(x_rel=np.array([0.0, -18.0, 0.0, 0.0, 0.0, 0.0]))

    m_nom = sensor.measure(state, t_s=10.0)
    assert not m_nom.glare_active and not m_nom.eclipse_active

    m_glare = sensor.measure(state, t_s=50.0)
    assert m_glare.glare_active
    assert np.linalg.norm(m_glare.z_pos_optical_m - state.pos_m) > 1.5

    m_eclipse = sensor.measure(state, t_s=140.0)
    assert m_eclipse.eclipse_active


def test_07_mekf_mahalanobis_glare_outlier_rejection() -> None:
    cfg = OrbitConfig()
    vcfg = VisionConfig()
    sensor = RadiometricOpticalSensor(vcfg, seed=99)
    state = SixDOFState(x_rel=np.array([0.0, -18.0, 0.0, 0.0, 0.0, 0.0]))

    mekf_gated = MultiplicativeEKF(cfg.mean_motion_rad_s, state, vcfg, enable_mahalanobis_gating=True)
    mekf_naive = MultiplicativeEKF(cfg.mean_motion_rad_s, state, vcfg, enable_mahalanobis_gating=False)

    meas_glare = sensor.measure(state, t_s=52.0)
    diag_gated = mekf_gated.update(meas_glare, state)
    diag_naive = mekf_naive.update(meas_glare, state)

    assert diag_gated.outlier_rejected is True
    assert diag_gated.pos_error_m < 0.08
    assert diag_naive.pos_error_m > 1.0


def test_08_cbf_safety_shield_repels_koz_approach() -> None:
    orbit_cfg = OrbitConfig()
    safety_cfg = SafetyConfig(koz_radius_m=2.0)
    shield = ASTM3269CBFShield(orbit_cfg, safety_cfg, enabled=True)

    # State just outside KOZ at y = -2.25 m with dangerous inward velocity +0.35 m/s
    x_danger = np.array([0.0, -2.25, 0.0, 0.0, 0.35, 0.0])
    u_aggressive = np.array([0.0, 0.08, 0.0])  # Trying to accelerate into KOZ
    decision = shield.certify_control(x_danger, u_aggressive, np.zeros(3), dt_s=0.5)

    assert decision.shield_intervened is True
    assert decision.u_safe_ms2[1] < 0.0  # Commands braking away from target


def test_09_cbf_safety_shield_enforces_los_cone() -> None:
    orbit_cfg = OrbitConfig()
    safety_cfg = SafetyConfig()
    shield = ASTM3269CBFShield(orbit_cfg, safety_cfg, enabled=True)

    # Near cone boundary in +x with outward +vx velocity
    x_cone_edge = np.array([2.4, -8.0, 0.0, 0.20, 0.05, 0.0])
    u_outward = np.array([0.06, 0.0, 0.0])
    decision = shield.certify_control(x_cone_edge, u_outward, np.zeros(3), dt_s=0.5)

    assert decision.shield_intervened is True
    assert decision.u_safe_ms2[0] < u_outward[0]


def test_10_ccsds_crc16_and_packet_framing() -> None:
    tm = CCSDSTelemetryPayload(
        t_s=12.5,
        z_pos_optical_m=np.array([1.0, -20.0, 0.5]),
        z_pos_lidar_m=np.array([1.01, -19.98, 0.49]),
        z_vel_lidar_ms=np.array([0.0, 0.15, 0.0]),
        z_q_target=np.array([1.0, 0.0, 0.0, 0.0]),
        z_w_target=np.array([0.01, -0.01, 0.02]),
        solar_phase_angle_deg=35.0,
        flags=1,
    )
    pkt = pack_ccsds_packet(APID_SENSOR_TELEMETRY, seq_count=42, payload_bytes=tm.to_bytes(), packet_type=0)
    hdr, body, crc_ok = unpack_ccsds_packet(pkt)
    assert crc_ok is True
    assert hdr.apid == APID_SENSOR_TELEMETRY
    assert hdr.seq_count == 42
    tm_out = CCSDSTelemetryPayload.from_bytes(body)
    assert np.allclose(tm_out.z_pos_optical_m, tm.z_pos_optical_m)


def test_11_ccsds_single_bit_upset_detection() -> None:
    tc = CCSDSTelecommandPayload(
        t_s=25.0,
        u_cmd_ms2=np.array([-0.01, 0.02, 0.005]),
        shield_intervened=True,
        active_mode_code=2,
    )
    pkt = bytearray(pack_ccsds_packet(APID_ACTUATOR_COMMAND, seq_count=7, payload_bytes=tc.to_bytes(), packet_type=1))
    pkt[10] ^= 0x04  # Flip 1 bit in payload
    _, _, crc_ok = unpack_ccsds_packet(bytes(pkt))
    assert crc_ok is False


def test_12_hil_bridge_udp_socket_mode() -> None:
    hil_cfg = HILBridgeConfig(
        transport_mode="UDP_SOCKET",
        seu_bitflip_probability=0.0,
        packet_dropout_probability=0.0,
    )
    bridge = CCSDSSpacePacketBridge(hil_cfg, seed=10)
    tc = CCSDSTelecommandPayload(
        t_s=5.0,
        u_cmd_ms2=np.array([0.01, -0.03, 0.0]),
        shield_intervened=True,
        active_mode_code=2,
    )
    rx_tc, ok = bridge.transmit_telecommand(tc, inject_faults=False)
    assert ok is True
    assert np.allclose(rx_tc.u_cmd_ms2, tc.u_cmd_ms2)


def test_13_full_rpo_campaign_shielded_vs_unshielded() -> None:
    result = run_rpo_campaign()
    assert result.shielded_run.min_koz_margin_m >= 0.0
    assert result.shielded_run.min_los_margin_m >= 0.0
    assert result.unshielded_run.min_koz_margin_m < 0.0 or result.unshielded_run.min_los_margin_m < 0.0
    assert all(p.shielded_status == "PASS" for p in result.pillars)


def test_14_cryptographic_reproducibility() -> None:
    result = run_rpo_campaign(seed=2026)
    assert result.reproducibility_max_diff == 0.0
    assert len(result.campaign_sha256) == 64


def test_15_realtime_50hz_qp_timing_budget() -> None:
    result = run_rpo_campaign()
    p99_us = float(np.percentile(result.shielded_run.qp_solve_time_us, 99))
    assert p99_us < 10000.0  # Well under 20,000 us (50 Hz frame budget)


def test_16_report_generation_artifacts(tmp_path: Path) -> None:
    result = run_rpo_campaign()
    paths = generate_all_reports(result, output_dir=tmp_path)
    assert Path(paths["html"]).exists()
    assert Path(paths["markdown"]).exists()
    assert Path(paths["json"]).exists()
