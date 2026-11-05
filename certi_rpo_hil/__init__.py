"""
CERTI-RPO-HIL: Certifiable Autonomous Rendezvous, Proximity Operations & Docking (RPOD)
Radiometric Optical Navigation & Control Barrier Function (CBF) Runtime Assurance HIL Testbed.

Developed under the Digital Intelligence Maintenance for Aircraft Systems in Africa (DIMASA)
and Physics Hybrid Integrity Lab initiative.
"""

from certi_rpo_hil.orbital_dynamics import (
    MU_EARTH,
    R_EARTH,
    J2_EARTH,
    OrbitConfig,
    ThrusterFaultConfig,
    SixDOFState,
    hcw_state_transition_matrix,
    propagate_hcw_free_drift,
    NonlinearJ2SixDOFDynamics,
)
from certi_rpo_hil.radiometric_vision import (
    VisionConfig,
    OpticalMeasurement,
    RadiometricOpticalSensor,
    MultiplicativeEKF,
)
from certi_rpo_hil.safety_shield import (
    SafetyConfig,
    ShieldDecision,
    NominalRPODGuidance,
    ASTM3269CBFShield,
)
from certi_rpo_hil.hil_bridge import (
    CCSDSPrimaryHeader,
    CCSDSTelemetryPayload,
    CCSDSTelecommandPayload,
    crc16_ccitt,
    pack_ccsds_packet,
    unpack_ccsds_packet,
    HILBridgeConfig,
    CCSDSSpacePacketBridge,
)
from certi_rpo_hil.verification import (
    PillarResult,
    CampaignComparisonResult,
    run_rpo_campaign,
    evaluate_seven_pillars,
)
from certi_rpo_hil.reporting import (
    generate_all_reports,
)

__version__ = "1.0.0"
__all__ = [
    "MU_EARTH",
    "R_EARTH",
    "J2_EARTH",
    "OrbitConfig",
    "ThrusterFaultConfig",
    "SixDOFState",
    "hcw_state_transition_matrix",
    "propagate_hcw_free_drift",
    "NonlinearJ2SixDOFDynamics",
    "VisionConfig",
    "OpticalMeasurement",
    "RadiometricOpticalSensor",
    "MultiplicativeEKF",
    "SafetyConfig",
    "ShieldDecision",
    "NominalRPODGuidance",
    "ASTM3269CBFShield",
    "CCSDSPrimaryHeader",
    "CCSDSTelemetryPayload",
    "CCSDSTelecommandPayload",
    "crc16_ccitt",
    "pack_ccsds_packet",
    "unpack_ccsds_packet",
    "HILBridgeConfig",
    "CCSDSSpacePacketBridge",
    "PillarResult",
    "CampaignComparisonResult",
    "run_rpo_campaign",
    "evaluate_seven_pillars",
    "generate_all_reports",
]
