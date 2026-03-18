
# OrbitalAI v9.0 – Fidelity-Tiered Orbital AI Constellation Simulator

**OrbitalAI v9** is a parametric digital twin / discrete-event simulation framework for evaluating the feasibility of running distributed AI workloads (federated training, inference, all-reduce style gradient exchange) on a constellation of small satellites in low Earth orbit (LEO).

It models — with explicit honesty — the brutal physics and engineering constraints:

- Solar power availability & degradation  
- Battery calendar + cycle aging (Arrhenius + non-linear)  
- Inter-satellite laser communication (ISL) bandwidth, acquisition time, pointing error, dropouts  
- Dynamic attitude control system (ACS) power draw  
- Radiation-induced soft errors & adversarial degradation  
- Orbital handover & visibility geometry (simplified)  

Three built-in **fidelity tiers** allow switching between fantasy optimism, near-term realism, and credible future scaling:

- **REDTEAM_2026** — generous/vaporware assumptions  
- **PLAUSIBLE_2028** — grounded in 2025–2027 CubeSat / optical ISL demos  
- **STRETCH_2032** — aggressive but credible 2030s projection  

The patched version adds a **SUNCATCHER_2027_PROTOTYPE** mode that mimics key assumptions from Google's **Project Suncatcher** (Nov 2025 announcement): tight 81-node formation flying at ~150 m spacing, dawn-dusk sun-synchronous orbit (SSO), 1.6 Tbps optical links, fast PAT, radiation-hardened TPUs, near-constant solar input.

## Features

- Tiered realism controls (fidelity levels)  
- Realistic battery fade physics (calendar + cycle aging)  
- Distance- & pointing-dependent free-space optical link model  
- Soft adversarial/radiation degradation instead of instant failure  
- Parameter sweeps for sensitivity analysis  
- Suncatcher mimic mode: 81-node tight cluster, 1.6 Tbps ISL, SSO solar boost  

## Requirements

Create a file named `requirements.txt` with the following content:

```text
numpy>=1.24.0
torch>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
uncertainties>=3.1.7
```

Install with:

```bash
pip install -r requirements.txt
```

**Minimum Python version**: 3.9+ (tested on 3.10–3.12)

No GPU required — simulation is CPU-only.

## Installation

1. Clone or download the repository

```bash
git clone https://github.com/yourusername/orbital-ai-sim.git
cd orbital-ai-sim
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Files

| File                            | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `orbitalai_v9.py`               | Original version (fidelity tiers REDTEAM / PLAUSIBLE / STRETCH)             |
| `orbitalai_v9_suncatcher.py`    | Patched version with SUNCATCHER_2027_PROTOTYPE + `--suncatcher-mimic` flag  |

## Usage

### Basic run (original version)

```bash
# Default: PLAUSIBLE_2028, 8 nodes
python orbitalai_v9.py

# Explicit fidelity
python orbitalai_v9.py --fidelity REDTEAM_2026
python orbitalai_v9.py --fidelity STRETCH_2032
```

### Suncatcher-mimicking runs

```bash
# Suncatcher near-term prototype mode (81 nodes recommended)
python orbitalai_v9_suncatcher.py --fidelity SUNCATCHER_2027_PROTOTYPE --num-nodes 81

# Apply aggressive Suncatcher physics on top of any base fidelity
python orbitalai_v9_suncatcher.py --fidelity PLAUSIBLE_2028 --suncatcher-mimic --num-nodes 81
```

### Run self-tests

```bash
python orbitalai_v9_suncatcher.py --test
# (currently stub — expand with real unit tests)
```

### Parameter sweep example (not yet implemented in CLI — extend `main()`)

```python
# Example sweep logic you can add:
for bw in [0.5, 5.0, 50.0, 1600.0]:
    cfg = Config(fidelity=FidelityLevel.PLAUSIBLE_2028)
    cfg.isl_bandwidth_gbps = bw
    const = ODCConstellation(32, cfg)
    metrics = const.run_distributed_workload(...)
    print(f"BW {bw} Gbps → {metrics['total_reliable_tops']:.1f} TOPS")
```

## Extending the Model

Key classes to subclass / override:

- `Config` — add new parameters or fidelity tiers  
- `CommModel.transfer_time()` — refine link budget, rain fade, etc.  
- `BatteryPhysics.update()` — use more detailed electrochemical models  
- `OrbitalAIChip.run_workload()` — plug in real TPU / systolic array perf numbers  
- `ODCConstellation` — improve orbit propagation, visibility windows, routing  

## Comparison with Project Suncatcher (Google, 2025–)

| Aspect                  | OrbitalAI PLAUSIBLE_2028 | SUNCATCHER_2027_PROTOTYPE (mimic) | Real Suncatcher (public info) |
|-------------------------|---------------------------|------------------------------------|-------------------------------|
| ISL Bandwidth           | 0.5 Gbps                 | 1.6 Tbps                          | 1.6 Tbps lab demo             |
| Node spacing            | ~800 km                  | ~150 m                            | 100–200 m oscillating         |
| Acquisition time        | 45 s                     | 4 s                               | (implied fast PAT)            |
| Solar availability      | Generic LEO              | ×7.8 dawn-dusk SSO                | near-constant insolation      |
| Effective cluster TOPS  | ~0.4–few TOPS            | ~10k–50k TOPS                     | targeted training-scale       |

The mimic mode shows how dramatically tight formation + Tbps links change the viability equation.

## License

MIT (or choose your preferred open-source license)

## Contributing

Feel free to open issues or PRs — especially welcome:

- Real orbit propagation (poliastro / Orekit integration)  
- Detailed all-reduce / federated training scaling laws  
- Economic model with realistic launch & ops costs  
- Visualization of battery fade, link uptime, etc.

Happy red-teaming orbital AI compute dreams!  
March 2026 – still very early days 🚀
```

You can copy-paste this directly into a `README.md` file.

