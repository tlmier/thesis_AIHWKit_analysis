#!/usr/bin/env python3
"""Datei-Ein-/Ausgabe fuer die HWKit-Ergebnis-CSVs.

Trennt bewusst von metrics.py (reine Kennzahlen-Logik). Metadaten (Netztyp,
Device Mode, Seed) kommen aus dem CSV-Kommentar-Header (JSON), nicht aus dem
Ordnernamen -- der ist teils irrefuehrend (z.B. liegen unter
CNN/Results/ablation und CNN/Results/qwell tatsaechlich MLP-Laeufe, siehe
network.type im Header). Nur der Pulsschema (SVS/CVS) steht ausschliesslich
im Dateinamen, das JSON unterscheidet SVS/CVS-Varianten sonst nicht.
"""
import csv
import json
import re
from pathlib import Path

from metrics import Run

_SEED_RE = re.compile(r"^(?P<study>.+)_s(?P<seed>[123])$")


def parse_csv(path):
    """Liest eine einzelne HWKit-Ergebnisdatei.

    Rueckgabe: dict mit sim_config/network (rohes JSON), abgeleiteten Feldern
    (seed, device_mode, network_type, study_id, pulse_scheme, devnoise,
    category) und der Testgenauigkeits-Kurve 'acc' (list[float]).
    """
    path = Path(path)
    sim_config = None
    network = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# sim config:"):
                sim_config = json.loads(line.split(":", 1)[1].strip())
            elif line.startswith("# network:"):
                network = json.loads(line.split(":", 1)[1].strip())

    if sim_config is None or network is None:
        raise ValueError(f"{path}: 'sim config' oder 'network' fehlt im Header")

    acc = []
    with open(path, encoding="utf-8") as f:
        rows = csv.DictReader((l for l in f if not l.startswith("#")), delimiter=";")
        for row in rows:
            acc.append(float(row["test_acc"]))

    m = _SEED_RE.match(path.stem)
    if not m:
        raise ValueError(f"{path}: Dateiname ohne '_s1/_s2/_s3'-Suffix")
    study_id = m.group("study")
    seed_from_name = int(m.group("seed"))
    if seed_from_name != sim_config["Seed"]:
        raise ValueError(
            f"{path}: Seed im Dateinamen ({seed_from_name}) != Seed im Header "
            f"({sim_config['Seed']})"
        )

    study_base = study_id
    pulse_scheme = None
    for suffix in ("SVS", "CVS"):
        if study_base.endswith("_" + suffix):
            pulse_scheme = suffix
            study_base = study_base[: -(len(suffix) + 1)]
            break

    devnoise = study_base.endswith("_devnoise")
    if devnoise:
        study_base = study_base[: -len("_devnoise")]

    return {
        "path": path,
        "sim_config": sim_config,
        "network": network,
        "network_type": network["type"],
        "seed": sim_config["Seed"],
        "device_mode": sim_config["Device Mode"],
        "acc": acc,
        "category": path.parent.name,
        "study_id": study_id,
        "study_base": study_base,
        "pulse_scheme": pulse_scheme,
        "devnoise": devnoise,
    }


def load_runs(csv_dir, pattern="*_s[123].csv"):
    """Gruppiert alle zum Pattern passenden CSVs in csv_dir zu Runs.

    Gruppenschluessel ist (category, study_id) -- category kommt vom
    Elternordner (final/ablation/devnoise/qwell/alt_20260911_2355 o.ae.),
    study_id ist der Dateiname ohne '_sN'. Jede Gruppe muss genau die Seeds
    1, 2, 3 enthalten (sonst Fehler statt stillem Ignorieren -- ein
    fehlender Seed soll auffallen, nicht in einer Study mit 2 Kurven enden).

    Rueckgabe: dict {(category, study_id): Run}, Run hat zusaetzlich .meta
    mit den gemeinsamen Metadaten des Runs (aus Seed 1, ohne 'acc'/'seed').
    """
    groups = {}
    for p in sorted(Path(csv_dir).glob(pattern)):
        rec = parse_csv(p)
        key = (rec["category"], rec["study_id"])
        groups.setdefault(key, {})[rec["seed"]] = rec

    runs = {}
    for key, seeds in groups.items():
        if set(seeds) != {1, 2, 3}:
            raise ValueError(f"{key}: erwarte Seeds 1,2,3, habe {sorted(seeds)}")
        run = Run(seeds[1]["acc"], seeds[2]["acc"], seeds[3]["acc"])
        run.meta = {k: v for k, v in seeds[1].items() if k not in ("acc", "seed", "path")}
        runs[key] = run
    return runs


# Wo welche Kategorie liegt, und ein optionaler Namensfilter je Quelle -- noetig
# weil CNN/Results/final eine vollstaendige Kopie (byte-identisch geprueft) von
# MLP/Results/final ist, PLUS die echten cnn_*-Studies obendrauf. Ohne den
# Filter wuerden dieselben MLP-Studies doppelt eingelesen und load_runs() liefe
# in den "erwarte Seeds 1,2,3, habe [1,1,2,2,3,3]"-Fehler.
#_SOURCES = [
#    ("MLP/Results/final", None),
 #   ("CNN/Results/final", lambda name: name.startswith("cnn_")),
  #  ("CNN/Results/ablation", None),
   # ("CNN/Results/devnoise", None),
    #("CNN/Results/qwell", None),
    #("CNN/Results/alt_20260911_2355", None),
#]
_SOURCES = [("Results/final", None),("Results/ablation", None),("Results/devnoise", None),("Results/qwell", None),("Results/alt_20260911_2355", None)]

def load_all_runs(root):
    """Laedt alle Runs aus allen bekannten Ergebnisordnern unter root.

    root ist das HWKitUse-Verzeichnis (Elternordner von CNN/ und MLP/).
    Rueckgabe: dict {(category, study_id): Run}, wie load_runs, nur ueber
    alle Quellen zusammengefuehrt. 'category' ist dabei nicht der rohe
    Ordnername, sondern das letzte Pfadsegment (z.B. "alt_20260911_2355"),
    damit alt/final/ablation/devnoise/qwell klar auseinanderbleiben.
    """
    root = Path(root)
    all_runs = {}
    for rel_dir, name_filter in _SOURCES:
        d = root / rel_dir
        if name_filter is None:
            paths = sorted(d.glob("*_s[123].csv"))
        else:
            paths = sorted(p for p in d.glob("*_s[123].csv") if name_filter(p.stem))

        groups = {}
        for p in paths:
            rec = parse_csv(p)
            key = (rec["category"], rec["study_id"])
            groups.setdefault(key, {})[rec["seed"]] = rec

        for key, seeds in groups.items():
            if key in all_runs:
                raise ValueError(f"{key}: doppelt vergeben (aus {rel_dir})")
            if set(seeds) != {1, 2, 3}:
                raise ValueError(f"{key}: erwarte Seeds 1,2,3, habe {sorted(seeds)}")
            run = Run(seeds[1]["acc"], seeds[2]["acc"], seeds[3]["acc"])
            run.meta = {k: v for k, v in seeds[1].items() if k not in ("acc", "seed", "path")}
            all_runs[key] = run

    return all_runs


# Reihenfolge/Namen der sim-config-Felder, die je Lauf variieren koennen und
# fuer die Origin-Sortierung interessant sind. Epochs/Dataset/Crossbar-Groesse
# etc. sind ueber alle Laeufe konstant, werden hier trotzdem mitgenommen --
# schadet nicht und Origin kann sie einfach ausblenden.
_SIM_CONFIG_COLS = [
    "Device Mode", "Optimizer", "LR", "Fast LR", "Transfer Every",
    "Chopper Prob", "Refresh Every", "DAC Resolution (bits)", "DAC Noise Std",
    "ADC Resolution (bits)", "ADC Noise Std", "Circuit Read Noise Std",
    "Noisy Backward Pass", "Epochs", "Train Batch Size", "Test Batch Size",
    "Max Crossbar Size", "Dataset",
]


def write_table(runs, out_path):
    """Schreibt eine Zeile je Run (Study) -- Identifikations- + Kennzahl-Spalten.

    Semikolon-getrennt, wie die HWKit-Rohdaten, damit direkt nach Origin
    importierbar. Eine Zeile je Simulation/Study (nicht je Seed) -- Seed 1/2/3
    unterscheiden sich nur im Zufalls-Seed, alle anderen Parameter sind je
    Study konstant (siehe _SOURCES-Docstring/Pruefung oben).
    """
    id_cols = ["Category", "StudyBase", "NetworkType", "PulseScheme", "DevNoise"]
    metric_cols = [
        "Mstar_mean", "Mstar_min", "Mstar_max", "Mstar_span", "Mstar_meancurve",
        "E_conv", "E_conv_nSeed", "S_ripple", "nLearned", "nLost", "SeedDependent",
    ]
    fieldnames = id_cols + _SIM_CONFIG_COLS + ["Network"] + metric_cols

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(fieldnames)
        for (category, study_id), run in sorted(runs.items()):
            meta = run.meta
            sim_config = meta["sim_config"]
            row = [
                category,
                meta["study_base"],
                meta["network_type"],
                meta["pulse_scheme"] or "",
                meta["devnoise"],
            ]
            row += [sim_config.get(col, "") for col in _SIM_CONFIG_COLS]
            row.append(json.dumps(meta["network"], ensure_ascii=False))
            row += [
                run.mstar_mean, run.mstar_min, run.mstar_max, run.mstar_span,
                run.mstar_meancurve, run.e_conv, run.e_conv_n_seed, run.ripple,
                run.n_learned, run.n_lost, run.seed_dependent,
            ]
            w.writerow(row)


if __name__ == "__main__":
    root = Path(__file__).parent
    runs = load_all_runs(root)
    out_path = root / "auswertung_tabelle.csv"
    write_table(runs, out_path)
    print(f"{len(runs)} Runs -> {out_path}")
