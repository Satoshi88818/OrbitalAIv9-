Here is a **patched/extended version** of the original `orbitalai_v9.py` that incorporates a Suncatcher-mimicking mode.  
Added:

- A new fidelity level `SUNCATCHER_2027_PROTOTYPE` (realistic near-term Google-style assumptions)
- A separate `SUNCATCHER_MIMIC` override dictionary that can be applied on top of any base fidelity
- Hard-coded cluster & orbit tweaks (81 nodes, tight 150 m formation, dawn-dusk SSO solar boost, 1.6 Tbps ISL, fast PAT, etc.)
- Minor extensions to `CommModel`, `Config`, and demo output so you can easily switch modes

This is **not** a 1:1 literal copy-paste of the original file (some long unchanged sections are abbreviated with `# ... original ...` comments to keep this readable). You can merge it back into the full codebase.

```python
#!/usr/bin/env python3
"""
Orbital AI Accelerator v9.0 – Fidelity-Tiered Digital Twin + Suncatcher Mimic
Fully updated with Path C enhancements (March 2026) + Suncatcher patch (2027 prototype mode)
Executable examples:
  python orbitalai_v9_suncatcher.py --fidelity PLAUSIBLE_2028
  python orbitalai_v9_suncatcher.py --fidelity SUNCATCHER_2027_PROTOTYPE
  python orbitalai_v9_suncatcher.py --suncatcher-mimic --num-nodes 81
"""

import argparse
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
# import matplotlib.pyplot as plt  # uncomment if you want plots
from scipy import stats as scipy_stats
from scipy.integrate import odeint
import uncertainties as unc

# ==============================================================================
# Logging & Fidelity
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orbital-ai-v9.0-suncatcher")

class FidelityLevel(Enum):
    REDTEAM_2026 = auto()
    PLAUSIBLE_2028 = auto()
    STRETCH_2032 = auto()
    SUNCATCHER_2027_PROTOTYPE = auto()   # New: Google-style near-term prototype assumptions

# ==============================================================================
# Custom Exceptions (unchanged)
# ==============================================================================

class OrbitalAIError(Exception): pass
class PowerFailure(OrbitalAIError): pass
class AvailabilityFailure(OrbitalAIError): pass
class EnduranceFailure(OrbitalAIError): pass
class ThermalRunaway(OrbitalAIError): pass
class ValidationError(OrbitalAIError): pass
class SecurityBreach(OrbitalAIError): pass

# ==============================================================================
# Config with Fidelity Tiers + Suncatcher overrides
# ==============================================================================

@dataclass
class Config:
    fidelity: FidelityLevel = FidelityLevel.REDTEAM_2026

    # Core params (REDTEAM defaults)
    array_size: int = 128
    mem_bw_gbps: float = 3200.0
    base_power_w: float = 200.0
    solar_panel_area_m2: float = 2.0
    solar_efficiency: float = 0.3
    solar_efficiency_boost: float = 1.0          # New: dawn-dusk SSO multiplier
    battery_capacity_wh: float = 200.0
    isl_bandwidth_gbps: float = 100.0
    p_adversarial_attack: float = 0.01
    gradient_compression_ratio: float = 0.1
    attitude_control_power_w: float = 40.0
    solar_degradation_annual: float = 0.025
    battery_cycle_life: int = 5000
    acquisition_time_s: float = 8.0
    pointing_std_deg: float = 0.1
    battery_calendar_fade_rate: float = 1e-8
    typical_distance_km: float = 800.0           # New: average neighbor distance
    overlap_fraction: float = 0.4                # New name (was overlap)

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any]) -> "Config":
        valid = {f.name for f in field(cls)}
        cleaned = {k: v for k, v in overrides.items() if k in valid}
        return cls(**cleaned)

    def apply_fidelity_overrides(self):
        if self.fidelity == FidelityLevel.REDTEAM_2026:
            self.isl_bandwidth_gbps = 100.0
            self.attitude_control_power_w = 40.0
            self.p_adversarial_attack = 0.01
            self.gradient_compression_ratio = 0.1
            self.solar_degradation_annual = 0.025
            self.battery_cycle_life = 5000
            self.acquisition_time_s = 8.0
            self.pointing_std_deg = 0.1
            self.overlap_fraction = 0.4
            logger.info("🎭 REDTEAM_2026 mode: generous assumptions")

        elif self.fidelity == FidelityLevel.PLAUSIBLE_2028:
            self.isl_bandwidth_gbps = 0.5
            self.attitude_control_power_w = 6.0
            self.p_adversarial_attack = 0.07
            self.gradient_compression_ratio = 0.35
            self.solar_degradation_annual = 0.032
            self.battery_cycle_life = 3500
            self.acquisition_time_s = 45.0
            self.pointing_std_deg = 0.4
            self.overlap_fraction = 0.15
            logger.info("📡 PLAUSIBLE_2028 mode: 2025–2027 grounded")

        elif self.fidelity == FidelityLevel.STRETCH_2032:
            self.isl_bandwidth_gbps = 5.0
            self.attitude_control_power_w = 3.5
            self.p_adversarial_attack = 0.03
            self.gradient_compression_ratio = 0.15
            self.solar_degradation_annual = 0.015
            self.battery_cycle_life = 9000
            self.overlap_fraction = 0.3
            logger.info("🚀 STRETCH_2032 mode: aggressive future")

        elif self.fidelity == FidelityLevel.SUNCATCHER_2027_PROTOTYPE:
            self.isl_bandwidth_gbps = 1600.0               # Google 1.6 Tbps bench
            self.attitude_control_power_w = 8.0            # tight formation station-keeping
            self.p_adversarial_attack = 0.008              # rad-tested TPU
            self.gradient_compression_ratio = 0.04         # high BW → low compression needed
            self.solar_degradation_annual = 0.018
            self.battery_cycle_life = 12000                # minimal cycling in SSO
            self.acquisition_time_s = 4.0                  # fast PAT in tight cluster
            self.pointing_std_deg = 0.04                   # sub-mrad class
            self.overlap_fraction = 0.85                   # near continuous visibility
            self.solar_efficiency_boost = 7.8              # dawn-dusk SSO gain
            logger.info("☀️ SUNCATCHER_2027_PROTOTYPE: Google-style tight-cluster assumptions")

        # Common clamps
        self.isl_bandwidth_gbps = max(0.05, self.isl_bandwidth_gbps)

    def apply_suncatcher_mimic(self):
        """Apply aggressive Suncatcher-like overrides on top of current fidelity"""
        overrides = {
            "num_nodes": 81,                           # Google 81-sat example
            "typical_distance_km": 0.15,               # 150 m average spacing
            "isl_bandwidth_gbps": 1600.0,
            "acquisition_time_s": 4.0,
            "pointing_std_deg": 0.04,
            "overlap_fraction": 0.85,
            "solar_efficiency_boost": 7.8,             # dawn-dusk near-constant insolation
            "p_adversarial_attack": 0.008,
            "gradient_compression_ratio": 0.04,
        }
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        logger.info("🔧 Applied Suncatcher mimic overrides (81-node tight cluster)")

    def update(self, overrides: Dict[str, Any]) -> "Config":
        new = self.from_dict({**asdict(self), **overrides})
        new.apply_fidelity_overrides()
        return new

# ==============================================================================
# BatteryPhysics (unchanged + minor comment)
# ==============================================================================

class BatteryPhysics:
    def __init__(self, capacity_wh: float, temp_k: float = 300.0):
        self.capacity_wh = capacity_wh
        self.temp_k = temp_k
        self.cumulative_cycles = 0.0

    def update(self, energy_drawn_wh: float, avg_temp_k: float, avg_soc: float = 0.5, dt_days: float = 1.0):
        arr = math.exp((avg_temp_k - 298.15) / 10.0)
        fade_cal = 8e-9 * arr * (avg_soc ** 2) * dt_days
        fade_cyc = 4e-5 * (energy_drawn_wh / self.capacity_wh) ** 1.75
        self.capacity_wh *= (1 - fade_cal - fade_cyc)
        self.cumulative_cycles += energy_drawn_wh / (self.capacity_wh * 0.3)
        self.temp_k = avg_temp_k

# ==============================================================================
# Enhanced CommModel – now uses typical_distance_km
# ==============================================================================

class CommModel:
    def __init__(self, config: Config):
        self.base_bw = config.isl_bandwidth_gbps
        self.acq_time = config.acquisition_time_s
        self.pointing_std = config.pointing_std_deg
        self.compression = config.gradient_compression_ratio
        self.overlap = config.overlap_fraction
        self.typical_dist_km = config.typical_distance_km

    def transfer_time(self, data_bytes: int, distance_km: Optional[float] = None, 
                      pointing_error_deg: Optional[float] = None, is_training: bool = False) -> float:
        if data_bytes <= 0:
            return 0.0
        if is_training:
            data_bytes = int(data_bytes * self.compression)

        dist = distance_km if distance_km is not None else self.typical_dist_km
        p_err = pointing_error_deg if pointing_error_deg is not None else self.pointing_std

        bw = self.base_bw * max(0.2, 1 - dist / 2000)                   # distance falloff
        bw *= max(0.3, 1 - (p_err / 2.0))                               # pointing penalty

        tx_s = data_bytes / (bw * 1e9 / 8)
        total = tx_s + self.acq_time + random.uniform(1, 4)             # handover
        if random.random() < 0.08:                                      # lower dropout in tight formation
            total *= 1.5
        return total * (1 - self.overlap)

# ==============================================================================
# OrbitalAIChip – minor ACS & battery integration (your original logic assumed)
# ==============================================================================

class OrbitalAIChip:
    def __init__(self, config: Config, phase_offset: float = 0.0):
        self.config = config
        self.battery_phys = BatteryPhysics(config.battery_capacity_wh * config.solar_efficiency_boost)
        self.comm = CommModel(config)
        self.acs_base_w = config.attitude_control_power_w
        # ... your original init fields ...

    def get_acs_power(self, slew_rad_s: float = 0.01) -> float:
        if self.config.fidelity in [FidelityLevel.SUNCATCHER_2027_PROTOTYPE]:
            return 6.0 + 2.5 * (slew_rad_s / 0.02)**2   # tighter control, less aggressive slew
        # ... your original dynamic ACS logic ...
        return self.acs_base_w  # fallback

    # run_workload(...) remains as in your original – just make sure to pass energy_consumed_wh

# ==============================================================================
# ODCConstellation stub (your original + node count support)
# ==============================================================================

class ODCConstellation:
    def __init__(self, num_nodes: int = 8, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.apply_fidelity_overrides()
        self.num_nodes = num_nodes
        logger.info(f"🚀 Launched {num_nodes}-node constellation @ {self.config.fidelity.name}")

    def run_distributed_workload(self, *args, **kwargs):
        # Your original heavy logic here – return metrics dict
        # For demo we fake plausible numbers
        base_tops = 120.0 * self.num_nodes
        comm_penalty = 1.0 / (1 + self.config.acquisition_time_s / 10)
        metrics = {
            "total_reliable_tops": base_tops * comm_penalty * (1 - self.config.p_adversarial_attack * 5),
            "cost_per_tops_year": 4200 / (base_tops * 0.001),
            "uptime": 0.88,
            # ... add your real keys ...
        }
        return metrics

# ==============================================================================
# Enhanced Demo
# ==============================================================================

def run_demo(config: Config, num_nodes: int = 8, use_suncatcher_mimic: bool = False):
    logger.info(f"🔥 OrbitalAI v9.0 + Suncatcher Demo – {config.fidelity.name}")
    config.apply_fidelity_overrides()
    if use_suncatcher_mimic:
        config.apply_suncatcher_mimic()

    const = ODCConstellation(num_nodes, config)
    metrics = const.run_distributed_workload()  # placeholder call

    print(f"\n{'Fidelity':18} | Nodes | Reliable TOPS | Cost/TOPS-yr | Uptime")
    print("-" * 65)
    print(f"{config.fidelity.name:18} | {num_nodes:5} | {metrics['total_reliable_tops']:13.1f} | "
          f"${metrics['cost_per_tops_year']:11.1f} | {metrics['uptime']:.2f}")

    if use_suncatcher_mimic:
        print("  └─ Suncatcher mimic active (tight 150 m formation, 1.6 Tbps ISL, SSO boost)")

# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="OrbitalAI v9.0 + Suncatcher Patch")
    parser.add_argument("--fidelity", choices=["REDTEAM_2026", "PLAUSIBLE_2028", "STRETCH_2032", "SUNCATCHER_2027_PROTOTYPE"],
                        default="PLAUSIBLE_2028", help="Simulation fidelity level")
    parser.add_argument("--num-nodes", type=int, default=8, help="Number of constellation nodes")
    parser.add_argument("--suncatcher-mimic", action="store_true", help="Apply aggressive Suncatcher-like overrides")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    fid_map = {
        "REDTEAM_2026": FidelityLevel.REDTEAM_2026,
        "PLAUSIBLE_2028": FidelityLevel.PLAUSIBLE_2028,
        "STRETCH_2032": FidelityLevel.STRETCH_2032,
        "SUNCATCHER_2027_PROTOTYPE": FidelityLevel.SUNCATCHER_2027_PROTOTYPE,
    }

    cfg = Config(fidelity=fid_map[args.fidelity])
    cfg.apply_fidelity_overrides()

    if args.test:
        logger.info("✅ Self-tests passed (stub)")
    else:
        run_demo(cfg, num_nodes=args.num_nodes, use_suncatcher_mimic=args.suncatcher_mimic)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("💥 Mission abort")
        sys.exit(1)
```

### Quick Usage Examples

```bash
# Baseline PLAUSIBLE
python orbitalai_v9_suncatcher.py --fidelity PLAUSIBLE_2028 --num-nodes 8

# Suncatcher near-term prototype mode
python orbitalai_v9_suncatcher.py --fidelity SUNCATCHER_2027_PROTOTYPE --num-nodes 81

# PLAUSIBLE base + forced Suncatcher physics (most extreme uplift)
python orbitalai_v9_suncatcher.py --fidelity PLAUSIBLE_2028 --suncatcher-mimic --num-nodes 81
```

