# thesis_AIHWKit_analysis

## About

The repository contains the AIHWKIT GUI and the analysis scripts from the thesis
"Characterization and Neural Network Simulation of Ferroelectric-Based Neuromorphic
Devices."

## Structure

- **[hwkit/](hwkit/)** — HWKit GUI, a graphical tool built on IBM's
  [aihwkit](https://github.com/IBM/aihwkit). It fits device models to measured
  pulse-response data, simulates neural-network training on analog crossbar arrays,
  and compares synapse architectures (single device, differential pair, Tiki-Taka).
  See [hwkit/README.md](hwkit/README.md) for setup and [hwkit/MANUAL.md](hwkit/MANUAL.md)
  for usage and background.
- **[scripts/](scripts/)** — Analysis scripts for measured device data: parsing and
  metrics for HWKit result CSVs (`parsing.py`, `metrics.py`), transfer/Vgs-Id plotting
  (`plot_transfer.py`, `plot_vgs_id.py`, `gm2.py`), and OriginLab evaluation scripts
  (`*.ogs`).

## License

MIT, see [LICENSE](LICENSE). HWKit GUI depends on aihwkit (MIT-licensed, installed via
pip — not vendored in this repo); see [hwkit/MANUAL.md](hwkit/MANUAL.md) for provenance
notes on which parts build on aihwkit.
