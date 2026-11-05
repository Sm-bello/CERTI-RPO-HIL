"""
End-to-End Executable TRL 4 Benchmark Campaign for CERTI-RPO-HIL.

Compares:
1. Unshielded Aggressive RPOD Guidance Policy (corrupted by Kapton MLI solar glare
   and a +0.032 m/s^2 stuck-open thruster fault)
2. CERTI-RPO-HIL Simplex + Control Barrier Function (CBF-QP) Shielded Digital Twin

Generates:
- reports/rpo_trl4_scorecard.html
- reports/TRL4_DOSSIER.md
- reports/trl4_audit.json
"""

from __future__ import annotations

from pathlib import Path
from certi_rpo_hil.verification import run_rpo_campaign
from certi_rpo_hil.reporting import generate_all_reports


def main() -> None:
    print("=" * 84)
    print(" CERTI-RPO-HIL: Autonomous RPOD Radiometric Vision & CBF-QP HIL Verification Suite")
    print(" Sponsored by DIMASA & Physics Hybrid Integrity Lab")
    print("=" * 84)

    result = run_rpo_campaign()
    project_root = Path(__file__).resolve().parent.parent
    report_paths = generate_all_reports(result, output_dir=project_root / "reports")

    print(f"\nCampaign SHA-256 Digest : {result.campaign_sha256}")
    print(f"Reproducibility Drift   : {result.reproducibility_max_diff:.2e} m\n")
    print("-" * 84)
    print(f"{'Verification Pillar':<44} | {'Shielded Twin':<14} | {'Unshielded':<14}")
    print("-" * 84)
    for p in result.pillars:
        print(f"{f'{p.pillar_id}. {p.name}':<44} | {p.shielded_status:<14} | {p.unshielded_status:<14}")
    print("-" * 84)

    print("\nGenerated TRL 4 Artifacts:")
    for k, path_str in report_paths.items():
        print(f"  [{k.upper()}] -> {path_str}")


if __name__ == "__main__":
    main()
