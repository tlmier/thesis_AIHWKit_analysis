"""Headless batch driver: runs many simulations from a manifest, N at a time.

Same code path as the GUI. The GUI's "Simulate" button loads a `.fit`, a `.sim` and
a `.net`, hands them to SimulationRunner and collects the progress tuples into a
results CSV -- this module does exactly that, without a window, once per manifest row
and several rows in parallel.

Nothing in the simulator is changed or reimplemented here: device loading, config
parsing, the run itself and the CSV formatting all call the same functions
gui_frontend.App calls (load_device, load_sim_config, load_network,
SimulationRunner, format_results_metadata / format_results_row).

Manifest (`;` or `,` separated, one row per run):

    RunID;fit;sim;net;seed;out
    A1;devices/svs.fit;configs/A1.sim;nets/mlp100.net;1;results/A1_s1.csv
    A1;devices/svs.fit;configs/A1.sim;nets/mlp100.net;2;results/A1_s2.csv

  RunID  free label, only used for logging
  fit    .fit file, as saved by "Save fit"
  sim    .sim file, as saved by "Save simulation config"
  net    .net file, as saved by "Save network"
  seed   overrides the "Seed" entry of the .sim; leave empty to keep the file's value.
         NOTE: 0 means *unseeded* -- simulate() does `if seed: torch.manual_seed(seed)`,
         and 0 is falsy in Python. Use 1, 2, 3 ... for repetitions.
  out    where the results CSV goes; relative paths are relative to the manifest

Because `seed` overrides the file, one .sim per configuration is enough for all of its
repetitions -- no need to duplicate configs just to vary the seed.

Run it from the HWKit directory with the aihwkit environment's interpreter -- the
training data is loaded from "./data" relative to the working directory (framework.py),
so starting elsewhere downloads MNIST again:

    conda activate aihwkit
    cd HWKit
    python batch_runner.py manifest.csv --workers 4
    python batch_runner.py manifest.csv --workers 2 --log batch.log
    python batch_runner.py manifest.csv --dry-run      # only check the manifest

SECOND MODE -- `--optuna`: search for configurations instead of running given ones.

The manifest mode above needs a .sim per run. The Optuna mode does not: it builds each
trial's config in code from DEFAULT_SIM_CONFIG and writes it out, so .sim files are an
*output* here, never an input. Only .fit and .net are supplied.

Study manifest (`;` or `,` separated, one row per study):

    StudyID;fit;net;mode;optimizer;noise;search;trials;out
    ideal_sgd;;Nets/MLP_100.net;Ideal;AnalogSGD;full;LR;30;Studies/ideal_sgd
    tt_CVS;Devices/CVS.fit;Nets/MLP_100.net;TikiTaka;AnalogSGD;full;LR+FastLR+TransferEvery;60;Studies/tt_CVS

  StudyID    label; also names the output files and the Optuna study
  fit        .fit file. Empty ONLY for mode "Ideal", which ignores the device entirely.
  net        .net file
  mode       one of simulation.DEVICE_MODES
  optimizer  AnalogSGD or AnalogAdam -- fixed per study, never searched
  noise      one of NOISE_PRESETS: "full" (config as-is), "off" (all ADC/DAC/circuit
             noise std to 0), "no_adc" (only the ADC read noise), "exact_bw" (only the
             backward pass exact). The DAC/ADC *resolution* stays active in every
             preset, so quantization always applies. The device's own non-ideality is
             not settable here -- it lives in the .fit, so select it via the fit column.
  search     "+"-joined tokens from SEARCH_SPACE, e.g. LR+FastLR+TransferEvery
  trials     how many trials this study gets
  out        directory for this study's per-trial .sim/.net/.csv

Every trial writes its files whether it finished or was pruned, so nothing is lost.
After the search, a manifest for the final runs is written automatically:

    python batch_runner.py --optuna studies.csv --workers 6 --epochs 30 --seeds 2
    python batch_runner.py final_manifest.csv --workers 36      # the numbers to report

Set HWKIT_NUM_THREADS (see framework.py) when running several workers -- without it every
worker asks torch for the whole node.

Filling a large allocation. Two measured facts shape this:

  * Threads buy nothing. Measured on one fixed config, 6 epochs: 1 thread 10.0s,
    2 threads 11.0s, 4 threads 11.0s, 8 threads 10.9s. A 784x100 layer is too small
    for OpenMP to pay for itself. So HWKIT_NUM_THREADS=1 and one core per worker.
  * Parallelising whole studies only does NOT scale: there are just 20 of them, and
    the wall time is then the longest single study (ttv2c is ~4h sequential).
    Measured against a 72-core budget: studies-only 4.0h on 20 cores, 12 shards x
    6 workers 0.7h on 72.

Do not simply raise --workers to the trial count either -- TPE learns only from trials
that have reported back, and Optuna's first ~10 trials are random anyway, so one wave
per study degenerates into a random search. 12 shards x 6 workers is the balance:
all 72 cores busy, and every study still gets several informed waves.

    for i in $(seq 1 12); do
        HWKIT_NUM_THREADS=1 python batch_runner.py --optuna studies.csv \
            --shard $i/12 --workers 6 --epochs 30 --seeds 2 \
            --export-best final_$i.csv &
    done
    wait
    head -1 final_1.csv > final_manifest.csv                # one header
    tail -q -n +2 final_*.csv >> final_manifest.csv         # all the rows

Keep the study manifest sorted most-expensive-first: --shard hands out rows round
robin, so that spreads the heavy studies evenly across the shards.
"""

import os

# aihwkit and PyTorch each bundle their own OpenMP runtime; on macOS loading both
# crashes the process unless this flag is set. Same line as main.py:9 -- it has to run
# before anything imports torch, which is why the heavy imports live inside run_one()
# and this sits at module import time (every spawned worker re-imports this module).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import statistics
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path

MANIFEST_COLUMNS = ["RunID", "fit", "sim", "net", "seed", "out"]
POLL_SECONDS = 0.2      # how often a worker drains its runner's queue


def read_manifest(path):
    '''Read the manifest into a list of dicts, delimiter auto-detected.

    Args:
        path (Path): manifest file

    Returns:
        list[dict]: one dict per run, keys as in MANIFEST_COLUMNS
    '''
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.DictReader(fh, dialect=dialect)]

    if not rows:
        sys.exit(f"{path}: no rows")
    missing = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing:
        sys.exit(f"{path}: missing columns {missing}. Expected {MANIFEST_COLUMNS}")

    cleaned = []
    for i, row in enumerate(rows, start=2):
        if not (row.get("RunID") or "").strip():
            continue                      # blank line
        cleaned.append({k: (row.get(k) or "").strip() for k in MANIFEST_COLUMNS} | {"_line": i})
    return cleaned


def run_one(row, base_dir, dump_sim):
    '''Run a single manifest row. Executed in its own process.

    Mirrors gui_frontend.App.do_simulate()/_poll_runner(): build the results header,
    start a SimulationRunner, drain it until it finishes, append one row per epoch.

    Args:
        row (dict): manifest row
        base_dir (str): directory the manifest lives in; relative paths resolve against it
        dump_sim (bool): also write the effective .sim (seed applied) next to the results

    Returns:
        dict: RunID, seed, status ("ok" | "error"), epochs, seconds, message
    '''
    base = Path(base_dir)
    started = time.time()
    run_id = row["RunID"]
    seed_txt = row["seed"]

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else base / p

    try:
        # Imported here, not at module level: keeps the parent process free of torch,
        # makes sure every spawned worker imports them after KMP_DUPLICATE_LIB_OK is set,
        # and -- being inside the try -- turns a broken environment into one failed run
        # instead of a dead batch.
        from config_loading import dump_sim_config, load_network, load_sim_config
        from device_fitting import load_device
        from gui_helpers import RESULTS_CSV_HEADER, format_results_metadata, format_results_row
        from simulation import SimulationRunner

        # The "Ideal" device mode is pure floating point -- it ignores the device and
        # every periphery non-ideality (see simulation.py's DEVICE_MODES), so its rows
        # may leave the fit column empty. Every other mode still requires one.
        device = offset = scale = None
        if row["fit"]:
            loaded = load_device(resolve(row["fit"]))
            device, offset, scale = loaded["device"], loaded["offset"], loaded["scale"]
        config = load_sim_config(resolve(row["sim"]))
        network = load_network(resolve(row["net"]))

        # Seed override -- the whole point of the manifest column. Applied before the
        # metadata block is built, so the saved CSV documents the seed actually used.
        if seed_txt:
            config = dict(config)
            config["Seed"] = int(seed_txt)

        out_path = resolve(row["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = format_results_metadata(device, offset, scale, config, network) + RESULTS_CSV_HEADER

        runner = SimulationRunner(device, config, network).start()
        epochs = 0
        while True:
            for item in runner.drain():
                if item[0] == "run":
                    _, test_acc, train_acc, epoch, loss = item
                    results += format_results_row(epoch, test_acc, train_acc, loss)
                    epochs += 1
            if runner.finished:
                break
            time.sleep(POLL_SECONDS)
        for item in runner.drain():          # whatever arrived after the last poll
            if item[0] == "run":
                _, test_acc, train_acc, epoch, loss = item
                results += format_results_row(epoch, test_acc, train_acc, loss)
                epochs += 1

        if runner.error is not None:
            raise runner.error

        out_path.write_text(results, encoding="utf-8")
        if dump_sim:
            out_path.with_suffix(".sim").write_text(dump_sim_config(config), encoding="utf-8")

        return {"RunID": run_id, "seed": seed_txt, "status": "ok", "epochs": epochs,
                "seconds": round(time.time() - started, 1), "message": str(out_path)}

    except Exception as exc:
        return {"RunID": run_id, "seed": seed_txt, "status": "error", "epochs": 0,
                "seconds": round(time.time() - started, 1),
                "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}


class Progress:
    """Single-line progress bar with a wall-clock ETA, written to stdout.

    The estimate is elapsed * remaining / done -- based on wall time, so it needs no
    knowledge of how many workers are running: if four runs finish in parallel, the
    elapsed time per completed run already reflects that.

    Runs differ a lot in length (a CNN config takes far longer than a small MLP), so
    the first estimates are rough and settle as the batch progresses.

    Falls back to one plain line per completed run when stdout is not a terminal, so
    redirecting into a file stays readable.
    """

    def __init__(self, total, width=28):
        self.total = total
        self.width = width
        self.done = 0
        self.failed = 0
        self.started = time.time()
        self.tty = sys.stdout.isatty()
        self._last_len = 0
        self._label = "queued"

    @staticmethod
    def _hms(seconds):
        seconds = int(max(0, seconds))
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def _eta(self):
        '''Remaining wall time, or None while nothing has finished yet.'''
        if not self.done:
            return None
        return (time.time() - self.started) * (self.total - self.done) / self.done

    def _draw(self):
        '''Redraw the single line. Same bar whether or not a run just finished.'''
        if not self.tty:
            return
        eta = self._eta()
        filled = round(self.width * self.done / self.total)
        bar = "\u2588" * filled + "\u2591" * (self.width - filled)
        line = (f"[{bar}] {self.done}/{self.total}  "
                + (f"failed {self.failed}  " if self.failed else "")
                + f"elapsed {self._hms(time.time() - self.started)}  "
                + f"ETA {self._hms(eta) if eta is not None else '--:--'}  {self._label}")
        sys.stdout.write("\r" + line.ljust(self._last_len))
        sys.stdout.flush()
        self._last_len = len(line)

    def start(self):
        '''Draw the empty bar before the first run finishes.

        update() only fires on completion, so without this the terminal shows nothing
        at all for as long as the slowest of the first `workers` runs takes -- with a
        small "Max Crossbar Size" that is many minutes, and looks like a hang.
        '''
        self.started = time.time()
        self._label = f"{self.total} queued"
        if self.tty:
            self._draw()
        else:
            print(f"  [0/{self.total}] started", flush=True)

    def tick(self):
        '''Redraw without a completion, so the elapsed clock keeps moving.'''
        self._draw()

    def update(self, res):
        self.done += 1
        if res["status"] != "ok":
            self.failed += 1
        self._label = f"{res['RunID']} seed={res['seed']}"

        if not self.tty:
            print(f"  [{self.done}/{self.total}] {self._label} {res['status']} "
                  f"{res['seconds']}s", flush=True)
            return

        # A failure scrolls away as its own line -- the bar would overwrite it.
        if res["status"] != "ok":
            self._clear()
            print(f"  FAILED {self._label}: {res['message'].splitlines()[0]}", flush=True)

        self._draw()

    def _clear(self):
        if self.tty and self._last_len:
            sys.stdout.write("\r" + " " * self._last_len + "\r")
            self._last_len = 0

    def finish(self):
        self._clear()


# --------------------------------------------------------------------------- #
# Optuna mode                                                                  #
#                                                                              #
# Second mode of this module. The manifest mode above runs configurations that #
# already exist as .sim files; this one searches for them. Nothing above is    #
# reused-by-copy: run_trial() drives the same SimulationRunner through the     #
# same drain loop as run_one(), and writes the same CSV format, so a winning   #
# trial's output triple can be fed straight back into a manifest.              #
# --------------------------------------------------------------------------- #

STUDY_COLUMNS = ["StudyID", "fit", "net", "mode", "optimizer", "noise", "search", "trials", "out"]

# Fixed for every study. The DAC/ADC resolution and the noise levels are hardware
# givens, not things to tune -- searching them would tune away the very
# non-ideality the comparison is about. 8/8 bits in BOTH noise arms on purpose:
# otherwise the "full" and "off" rows would differ in two things at once
# (noise AND quantization) and the difference could not be attributed to either.
STUDY_FIXED = {
    "Test Epoch Size": 10000,          # the complete MNIST test set
    "DAC Resolution (bits)": 8,
    "ADC Resolution (bits)": 8,
    # Array dimension, i.e. mapping.max_input_size/max_output_size -- 32x32 is the
    # largest FeFET crossbar actually demonstrated in hardware (Soliman et al.,
    # Nat. Commun. 14, 6348, 2023: 28nm HKMG, doped HfO2), so it is citable rather
    # than assumed. For reference: DNN+NeuroSim's numRowSubArray/numColSubArray is
    # 128 -- an architectural target, not a fabricated FeFET array. NeuroSim's
    # numColMuxed = 16 is a peripheral sharing factor (columns per read circuit),
    # NOT an array size, and does not belong here.
    "Max Crossbar Size": 32,
}

# What the manifest's noise column selects. The DAC/ADC *resolution* is deliberately
# not part of any preset -- quantization stays active everywhere, so the arms differ
# in one thing at a time.
#
# The device's own non-ideality is NOT reachable from here -- it lives in the .fit
# (dw_min_std, dw_min_dtod), so that arm is selected by pointing the fit column at
# a different device instead.
#
# Only three noise sources are ever active: the ADC read noise (0.06 by default),
# the noisy backward pass, and the device's own variation. DAC and circuit noise
# sit at 0.0 in DEFAULT_SIM_CONFIG, which is why "off" and "no_adc" currently
# produce identical configs -- kept apart anyway so the manifest says what it means.
#
# The presets are written so each source can be switched on ALONE, starting from
# "none". That is what answers "which one hurts most"; removing one source from
# the full set answers the different question of marginal contribution.
#
#   arm        preset      fit      ADC    backward   device
#   none       none        plain    0      exact      -
#   adc        exact_bw    plain    0.06   exact      -
#   bw         off         plain    0      noisy      -
#   dev        none        noisy    0      exact      yes
#   full       full        plain    0.06   noisy      -
#   all        full        noisy    0.06   noisy      yes
NOISE_PRESETS = {
    "full": {},                                          # config as-is
    "off": {"ADC Noise Std": 0.0, "DAC Noise Std": 0.0,  # all periphery noise
            "Circuit Read Noise Std": 0.0},
    "no_adc": {"ADC Noise Std": 0.0},                    # only the ADC read noise
    "exact_bw": {"Noisy Backward Pass": "No"},           # only the backward pass
    "none": {"ADC Noise Std": 0.0, "DAC Noise Std": 0.0, # periphery off AND
             "Circuit Read Noise Std": 0.0,              # backward exact: the
             "Noisy Backward Pass": "No"},               # clean starting point
}

# Tokens accepted in the manifest's `search` column, mapped to the config key they
# set and how the value is drawn. Ranges: LR spans the useful band for both
# AnalogSGD and AnalogAdam; Fast LR is widened around aihwkit's own presets
# (0.1-1.0) because this project's ExpStep fit is far more asymmetric than the
# devices those were tuned on; Refresh Every excludes 0 (= never refresh), which
# saturates both cells and is a demonstration, not a candidate.
# Von main_optuna() aus --lr-min/--lr-max gesetzt, bevor die Suche startet.
LR_RANGE = [1e-4, 1.0]

SEARCH_SPACE = {
    # Upper bound is 1.0, not 0.1: in the first full run 7 of 20 studies put their
    # winner in the top 10% of a 1e-4..1e-1 range and three sat essentially on the
    # 0.1 ceiling -- every CVS study among them. A winner at the boundary means the
    # box was too small, not that the optimum was found. High learning rates are
    # survivable here because the analog weights are hardware-bounded by w_min/w_max
    # (only the digital bias is unbounded, which BIAS_LR_SCALE already handles).
    # Grenzen kommen aus --lr-min/--lr-max, Default 1e-4..1.0 (der MLP-Lauf).
    # Faltungsnetze brauchen deutlich kleinere Werte: in AnalogConv2d wird dieselbe
    # Kachel fuer jede Bildposition wiederverwendet, die erste Faltung von CNN_D
    # bekommt also 784 analoge Updates je Bild statt einem. Die Asymmetrie des
    # Devices akkumuliert entsprechend schneller, und das Optimum liegt um etwa
    # diesen Faktor tiefer -- beim ersten CNN-Lauf klebten zwei von sechs
    # Gewinnern auf der 1e-4-Untergrenze.
    "LR": ("LR", lambda t: t.suggest_float("LR", LR_RANGE[0], LR_RANGE[1], log=True)),
    "FastLR": ("Fast LR", lambda t: t.suggest_float("Fast LR", 1e-3, 10.0, log=True)),
    "TransferEvery": ("Transfer Every", lambda t: t.suggest_float("Transfer Every", 1.0, 100.0, log=True)),
    "ChopperProb": ("Chopper Prob", lambda t: t.suggest_float("Chopper Prob", 1e-3, 0.5, log=True)),
    "RefreshEvery": ("Refresh Every", lambda t: t.suggest_categorical("Refresh Every", [1, 2, 5, 10, 50, 100])),
}

# Epochs whose accuracy is medianed into a trial's score. The median of a window
# rather than max() over the whole curve: aihwkit's device noise is not seedable
# (its C++ backend exposes no seed at all), so the curve is genuinely noisy and
# its maximum is an optimistically biased estimator.
SCORE_WINDOW = 5


def read_study_manifest(path):
    '''Read the study manifest -- one row per Optuna study.

    Args:
        path (Path): study manifest CSV, columns as in STUDY_COLUMNS

    Returns:
        list[dict]: one dict per study
    '''
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(fh, dialect=dialect))

    if not rows:
        sys.exit(f"{path}: no rows")
    missing = [c for c in STUDY_COLUMNS if c not in rows[0]]
    if missing:
        sys.exit(f"{path}: missing columns {missing}. Expected {STUDY_COLUMNS}")

    cleaned = []
    for i, row in enumerate(rows, start=2):
        if not (row.get("StudyID") or "").strip():
            continue
        cleaned.append({k: (row.get(k) or "").strip() for k in STUDY_COLUMNS} | {"_line": i})
    return cleaned


def build_study_config(row, epochs, params):
    '''Assemble a full .sim-style config for one trial.

    Built from DEFAULT_SIM_CONFIG rather than read from a file, so the search
    needs no hand-written .sim input -- the .sim is an *output* here, written per
    trial and directly reusable as manifest input.

    Args:
        row (dict): study manifest row
        epochs (int): epochs for this trial
        params (dict): the parameters Optuna drew, already keyed by config name

    Returns:
        dict: the config to hand to SimulationRunner
    '''
    from simulation import DEFAULT_SIM_CONFIG

    config = dict(DEFAULT_SIM_CONFIG)
    config.update(STUDY_FIXED)
    config.update({
        "Epochs": epochs,
        "Device Mode": row["mode"],
        "Optimizer": row["optimizer"],
    })
    config.update(NOISE_PRESETS[row["noise"]])
    config.update(params)
    return config


def run_trial(fit_path, config, network, out_base, seeds, epoch_queue, cancel_event):
    '''Run one trial -- every seed of one configuration -- in its own process.

    Mirrors run_one()'s drain loop, with two additions the search needs: each
    finished epoch of the FIRST seed is pushed to `epoch_queue` so the parent can
    prune on it, and a set `cancel_event` stops the run via
    SimulationRunner.cancel() (which finishes the current epoch and keeps
    everything already produced, see simulation.py:471).

    Writes `<out_base>.sim`, `<out_base>.net` and one `<out_base>_s<seed>.csv`
    per seed -- always, including for a pruned trial, so every trial leaves a
    complete, manifest-ready record behind.

    Args:
        fit_path (str): .fit file, or "" for the device-less "Ideal" mode
        config (dict): full config from build_study_config()
        network (dict): loaded .net
        out_base (str): path prefix for this trial's output files
        seeds (list[int]): seeds to average over
        epoch_queue: Manager().Queue() the parent drains
        cancel_event: Manager().Event() the parent sets to prune

    Returns:
        dict: scores, pruned flag, status, epochs, seconds, message
    '''
    started = time.time()
    out_base = Path(out_base)

    try:
        from config_loading import dump_network, dump_sim_config
        from device_fitting import load_device
        from gui_helpers import RESULTS_CSV_HEADER, format_results_metadata, format_results_row
        from simulation import SimulationRunner

        # "Ideal" is pure floating point -- it ignores the device and every
        # periphery non-ideality, so no .fit is loaded (or needed) for it.
        device = offset = scale = None
        if fit_path:
            loaded = load_device(fit_path)
            device, offset, scale = loaded["device"], loaded["offset"], loaded["scale"]

        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_base.with_suffix(".net").write_text(dump_network(network), encoding="utf-8")

        scores, pruned, total_epochs = [], False, 0
        for index, seed in enumerate(seeds):
            cfg = dict(config)
            cfg["Seed"] = seed

            results = format_results_metadata(device, offset, scale, cfg, network) + RESULTS_CSV_HEADER
            accuracies = []

            runner = SimulationRunner(device, cfg, network).start()
            while True:
                for item in runner.drain():
                    if item[0] == "run":
                        _, test_acc, train_acc, epoch, loss = item
                        results += format_results_row(epoch, test_acc, train_acc, loss)
                        accuracies.append(test_acc)
                        total_epochs += 1
                        # Only the first seed feeds the pruner: the parent needs one
                        # curve to judge, and the later seeds exist to average away
                        # the device noise, not to be judged again.
                        if index == 0:
                            epoch_queue.put((epoch, test_acc))
                if index == 0 and cancel_event.is_set() and not runner.cancelled:
                    runner.cancel()
                if runner.finished:
                    break
                time.sleep(POLL_SECONDS)
            for item in runner.drain():
                if item[0] == "run":
                    _, test_acc, train_acc, epoch, loss = item
                    results += format_results_row(epoch, test_acc, train_acc, loss)
                    accuracies.append(test_acc)
                    total_epochs += 1

            (out_base.parent / f"{out_base.name}_s{seed}.csv").write_text(results, encoding="utf-8")
            if runner.error is not None:
                raise runner.error
            if not accuracies:
                raise RuntimeError("no epoch completed")

            window = accuracies[-SCORE_WINDOW:]
            scores.append(statistics.median(window))

            # Pruned on the first seed -- the remaining seeds would only spend time
            # on a configuration the pruner has already rejected.
            if index == 0 and cancel_event.is_set():
                pruned = True
                break

        out_base.with_suffix(".sim").write_text(dump_sim_config(config), encoding="utf-8")

        return {"scores": scores, "pruned": pruned, "status": "ok", "epochs": total_epochs,
                "seconds": round(time.time() - started, 1), "message": str(out_base)}

    except Exception as exc:
        return {"scores": [], "pruned": False, "status": "error", "epochs": 0,
                "seconds": round(time.time() - started, 1),
                "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}


def run_study(row, base_dir, args, progress):
    '''Run one study: ask/tell over `trials` trials, `--workers` of them at a time.

    Optuna's ask/tell API rather than study.optimize(): a Trial object cannot be
    pickled into a worker process, so the parent keeps the trials and only the
    plain config dict travels. Per-epoch accuracies come back through a Manager
    queue, which is what makes pruning possible across the process boundary.

    Args:
        row (dict): study manifest row
        base_dir (Path): directory the study manifest lives in
        args (Namespace): parsed CLI arguments
        progress (Progress): shared progress bar, ticked per finished trial

    Returns:
        optuna.Study: the finished study
    '''
    import optuna
    from optuna.trial import TrialState

    from config_loading import load_network

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else base_dir / p

    network = load_network(resolve(row["net"]))
    fit = str(resolve(row["fit"])) if row["fit"] else ""
    out_dir = resolve(row["out"])
    seeds = list(range(1, args.seeds + 1))
    tokens = [t for t in row["search"].split("+") if t]
    n_trials = int(row["trials"])

    study = optuna.create_study(
        study_name=row["StudyID"],
        direction="maximize",
        # Percentile rather than median: a MedianPruner stops everything below the
        # median, i.e. about half of all trials including ones only marginally
        # behind -- and with a run-to-run spread of ~1.5pp (measured over the
        # existing Seed1/2/3 results) that means pruning on noise. At 25% only the
        # clearly bad quarter is stopped, which still catches the diverged runs
        # that actually cost time. --prune-percentile 50 restores median behaviour.
        # n_warmup_steps keeps the pruner off the first epochs, where an analog run
        # legitimately looks bad before the tiles settle.
        pruner=optuna.pruners.PercentilePruner(
            args.prune_percentile, n_startup_trials=5, n_warmup_steps=args.prune_warmup),
    )

    with Manager() as manager, ProcessPoolExecutor(max_workers=args.workers) as pool:
        inflight, asked = {}, 0
        while asked < n_trials or inflight:
            while asked < n_trials and len(inflight) < args.workers:
                trial = study.ask()
                params = {}
                for token in tokens:
                    key, draw = SEARCH_SPACE[token]
                    params[key] = draw(trial)
                config = build_study_config(row, args.epochs, params)
                queue_, event = manager.Queue(), manager.Event()
                out_base = out_dir / f"{row['StudyID']}_t{trial.number:04d}"
                future = pool.submit(run_trial, fit, config, network, str(out_base),
                                     seeds, queue_, event)
                inflight[future] = {"trial": trial, "queue": queue_, "event": event}
                asked += 1

            done, _ = wait(set(inflight), timeout=0.5, return_when=FIRST_COMPLETED)

            # Drain BEFORE retiring finished futures, otherwise the last epochs of a
            # run that just completed are dropped on the floor.
            for state in inflight.values():
                while True:
                    try:
                        epoch, acc = state["queue"].get_nowait()
                    except Exception:
                        break
                    state["trial"].report(acc, epoch)
                    if state["trial"].should_prune():
                        state["event"].set()

            for future in done:
                state = inflight.pop(future)
                try:
                    res = future.result()
                except Exception as exc:
                    res = {"scores": [], "pruned": False, "status": "error", "epochs": 0,
                           "seconds": 0.0, "message": f"worker died: {type(exc).__name__}: {exc}"}
                if res["status"] != "ok":
                    study.tell(state["trial"], state=TrialState.FAIL)
                elif res["pruned"]:
                    study.tell(state["trial"], state=TrialState.PRUNED)
                else:
                    study.tell(state["trial"], sum(res["scores"]) / len(res["scores"]))
                progress.update({"RunID": row["StudyID"], "seed": f"t{state['trial'].number}",
                                 "status": res["status"], "seconds": res["seconds"],
                                 "message": res["message"]})
            if not done:
                progress.tick()

    return study


def export_best(studies, out_path, base_dir, seeds=3, epochs=100):
    '''Write a normal manifest for the final runs from the finished studies.

    A fresh `<StudyID>_best.sim` is written per study rather than reusing the
    winning trial's file: that one carries the *search* epoch count (short on
    purpose), and the final runs need the full one. Everything else is the
    winning configuration verbatim.

    The final numbers come from these re-runs, not from the trial values -- the
    noise draw that pushed a trial to the top does not repeat with fresh seeds,
    so a re-run is the honest estimate of the chosen configuration.

    Args:
        studies (list[tuple]): (study manifest row, optuna.Study) pairs
        out_path (Path): manifest to write
        base_dir (Path): directory the study manifest lives in
        seeds (int): repetitions per winning configuration
        epochs (int): epochs for the final runs

    Returns:
        Path: out_path
    '''
    from config_loading import dump_sim_config

    lines = [";".join(MANIFEST_COLUMNS)]
    for row, study in studies:
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if not completed:
            print(f"  {row['StudyID']}: no completed trial -- skipped", file=sys.stderr)
            continue
        best = max(completed, key=lambda t: t.value)

        params = {SEARCH_SPACE[tok][0]: best.params[SEARCH_SPACE[tok][0]]
                  for tok in row["search"].split("+") if tok}
        config = build_study_config(row, epochs, params)

        out_dir = Path(row["out"])
        resolved = out_dir if out_dir.is_absolute() else base_dir / out_dir
        (resolved / f"{row['StudyID']}_best.sim").write_text(
            dump_sim_config(config), encoding="utf-8")

        for seed in range(1, seeds + 1):
            lines.append(";".join([
                row["StudyID"], row["fit"], str(out_dir / f"{row['StudyID']}_best.sim"),
                str(out_dir / f"{row['StudyID']}_t{best.number:04d}.net"), str(seed),
                str(Path("Results/final") / f"{row['StudyID']}_s{seed}.csv"),
            ]))
        print(f"  {row['StudyID']:<28} best t{best.number:04d}  {best.value:.2f}%  {best.params}")

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nfinal manifest -> {out_path}  ({len(lines) - 1} runs)")
    return out_path


def main_optuna(args):
    '''Entry point for `--optuna` -- run every study in the study manifest.'''
    manifest_path = Path(args.optuna).resolve()
    base_dir = manifest_path.parent
    rows = read_study_manifest(manifest_path)

    problems = []
    for row in rows:
        for col in ("fit", "net"):
            if col == "fit" and not row["fit"]:
                continue                      # only "Ideal" may leave it empty
            p = Path(row[col])
            p = p if p.is_absolute() else base_dir / p
            if not p.is_file():
                problems.append(f"line {row['_line']} ({row['StudyID']}): {col} not found: {p}")
        if not row["fit"] and row["mode"] != "Ideal":
            problems.append(f"line {row['_line']} ({row['StudyID']}): only mode 'Ideal' may "
                            f"have an empty fit, got {row['mode']!r}")
        if row["noise"] not in NOISE_PRESETS:
            problems.append(f"line {row['_line']} ({row['StudyID']}): unknown noise "
                            f"{row['noise']!r}. Known: {sorted(NOISE_PRESETS)}")
        for token in row["search"].split("+"):
            if token and token not in SEARCH_SPACE:
                problems.append(f"line {row['_line']} ({row['StudyID']}): unknown search token "
                                f"{token!r}. Known: {sorted(SEARCH_SPACE)}")
        try:
            int(row["trials"])
        except ValueError:
            problems.append(f"line {row['_line']} ({row['StudyID']}): trials is not an integer")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(f"\n{len(problems)} problem(s) in the study manifest -- nothing was run.")

    # Sharding: studies are independent, so several processes can each take every
    # n-th row. This is the way to use a big allocation WITHOUT starving the
    # sampler -- TPE only learns from trials that have already reported back, so
    # asking 30 trials at once (one worker per trial of a 30-trial study) makes it
    # a random search. Few workers per study, many studies at once, is better.
    if args.shard:
        try:
            index, count = (int(x) for x in args.shard.split("/"))
        except ValueError:
            sys.exit(f"--shard expects i/n, got {args.shard!r}")
        if not 1 <= index <= count:
            sys.exit(f"--shard {args.shard}: i must be between 1 and n")
        rows = rows[index - 1::count]
        print(f"shard {index}/{count}: {len(rows)} of the studies")
        if not rows:
            return

    LR_RANGE[0], LR_RANGE[1] = args.lr_min, args.lr_max
    if args.lr_min >= args.lr_max:
        sys.exit(f"--lr-min {args.lr_min} muss kleiner als --lr-max {args.lr_max} sein")

    total = sum(int(r["trials"]) for r in rows)
    print(f"LR-Suchraum {args.lr_min:g} .. {args.lr_max:g}")
    print(f"{len(rows)} studies, {total} trials from {manifest_path.name}, "
          f"{args.workers} in parallel, {args.epochs} epochs, {args.seeds} seeds")
    if args.dry_run:
        print("dry run -- study manifest and input files are fine, nothing executed")
        return

    progress = Progress(total)
    progress.start()
    started = time.time()
    studies = []
    for row in rows:
        studies.append((row, run_study(row, base_dir, args, progress)))
    progress.finish()
    print(f"\n{len(rows)} studies in {round(time.time() - started, 1)}s\n")

    export_best(studies, base_dir / args.export_best, base_dir,
                seeds=args.final_seeds, epochs=args.final_epochs)



def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", help="manifest CSV, see module docstring")
    ap.add_argument("--workers", type=int, default=1,
                    help="how many runs in parallel (default 1)")
    ap.add_argument("--log", default=None, help="write a per-run log here")
    ap.add_argument("--no-dump-sim", action="store_true",
                    help="do not write the effective .sim next to each results file")
    ap.add_argument("--dry-run", action="store_true",
                    help="only check the manifest and that all input files exist")
    # Second mode. Without --optuna every argument above behaves exactly as before.
    ap.add_argument("--optuna", metavar="STUDY_CSV", default=None,
                    help="run the Optuna studies described by this study manifest")
    ap.add_argument("--epochs", type=int, default=30,
                    help="epochs per trial, shorter than the final runs (--optuna only)")
    ap.add_argument("--seeds", type=int, default=2,
                    help="seeds averaged per trial (--optuna only)")
    ap.add_argument("--final-seeds", type=int, default=3,
                    help="seed rows per winner in the exported manifest (--optuna only)")
    ap.add_argument("--lr-min", type=float, default=1e-4,
                    help="untere Grenze des LR-Suchraums (--optuna only)")
    ap.add_argument("--lr-max", type=float, default=1.0,
                    help="obere Grenze des LR-Suchraums (--optuna only)")
    ap.add_argument("--prune-percentile", type=float, default=25.0,
                    help="stop a trial below this percentile of the finished trials at the "
                         "same epoch (50 = the usual median pruning; --optuna only)")
    ap.add_argument("--prune-warmup", type=int, default=10,
                    help="never prune before this epoch (--optuna only)")
    ap.add_argument("--shard", metavar="i/n", default=None,
                    help="process only every n-th study, starting at i -- run n of these "
                         "in parallel to fill a large allocation (--optuna only)")
    ap.add_argument("--final-epochs", type=int, default=100,
                    help="epochs for the final runs written by --export-best")
    ap.add_argument("--export-best", metavar="CSV", default="final_manifest.csv",
                    help="where --optuna writes the manifest for the final runs")
    args = ap.parse_args()

    if args.optuna:
        return main_optuna(args)
    if not args.manifest:
        ap.error("give a manifest CSV, or --optuna with a study manifest")

    manifest_path = Path(args.manifest).resolve()
    base_dir = manifest_path.parent
    rows = read_manifest(manifest_path)

    # Fail before starting hours of compute rather than on row 47.
    problems = []
    for row in rows:
        for col in ("fit", "sim", "net"):
            if col == "fit" and not row["fit"]:
                continue          # allowed for "Ideal" rows, which use no device
            p = Path(row[col])
            p = p if p.is_absolute() else base_dir / p
            if not p.is_file():
                problems.append(f"line {row['_line']} ({row['RunID']}): {col} not found: {p}")
        if row["seed"]:
            try:
                if int(row["seed"]) == 0:
                    problems.append(f"line {row['_line']} ({row['RunID']}): seed 0 means "
                                    f"*unseeded* (simulate() does `if seed:`). Use 1, 2, 3 ...")
            except ValueError:
                problems.append(f"line {row['_line']} ({row['RunID']}): seed is not an integer")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(f"\n{len(problems)} problem(s) in the manifest -- nothing was run.")

    print(f"{len(rows)} runs from {manifest_path.name}, {args.workers} in parallel")
    if args.dry_run:
        print("dry run -- manifest and input files are fine, nothing executed")
        return

    results = []
    started = time.time()
    payload = [(row, str(base_dir), not args.no_dump_sim) for row in rows]

    progress = Progress(len(rows))
    progress.start()

    if args.workers <= 1:
        for row, bd, ds in payload:
            res = run_one(row, bd, ds)
            results.append(res)
            progress.update(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, row, bd, ds): row for row, bd, ds in payload}
            pending = set(futures)
            while pending:
                # wait(timeout=) rather than as_completed(): the loop comes back once a
                # second even when nothing finished, so the bar can keep ticking. A run
                # can take many minutes, and a frozen line is indistinguishable from a
                # dead batch.
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        res = fut.result()
                    except Exception as exc:
                        # Worker died outside run_one's own error handling (OOM, segfault,
                        # broken interpreter). Record it and keep the remaining runs going.
                        row = futures[fut]
                        res = {"RunID": row["RunID"], "seed": row["seed"], "status": "error",
                               "epochs": 0, "seconds": 0.0,
                               "message": f"worker died: {type(exc).__name__}: {exc}"}
                    results.append(res)
                    progress.update(res)
                if not done:
                    progress.tick()

    progress.finish()
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    total = round(time.time() - started, 1)
    print(f"\n{len(ok)} ok, {len(bad)} failed, {total}s wall time")
    for r in bad:
        print(f"  FAILED {r['RunID']} seed={r['seed']}: {r['message'].splitlines()[0]}",
              file=sys.stderr)

    if args.log:
        with open(args.log, "w", encoding="utf-8") as fh:
            fh.write(f"# {manifest_path}  workers={args.workers}  wall={total}s\n")
            for r in results:
                fh.write(f"{r['RunID']};{r['seed']};{r['status']};{r['epochs']};"
                         f"{r['seconds']};{r['message']}\n")
        print(f"log -> {args.log}")

    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
