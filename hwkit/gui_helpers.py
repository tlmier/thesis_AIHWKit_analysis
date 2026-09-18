"""Helper functions used by gui_frontend.App that aren't themselves widget
construction file dialogs, ttk styling, and text/data formatting. 
Keptseparate so the frontend module stays limited to layout and thin event
handlers."""

import json
from datetime import datetime

# tkinter is deliberately NOT imported at module level. batch_runner.py pulls the
# formatting helpers (RESULTS_CSV_HEADER, format_results_row,
# format_results_metadata) out of this module, and a headless Python -- a cluster
# compute node, say -- has no _tkinter, which would make every batch run fail on
# import. The three functions that actually need Tk import it themselves.

SECTION_WIDTH = 34

# Explicit color palette so the GUI looks the same regardless of the OS's
BG = "#f3f4f6"
CARD_BG = "#ffffff"
BORDER = "#d7d9de"
TEXT = "#1f2430"
MUTED_TEXT = "#6b7280"
ACCENT = "#2f6fed"
ACCENT_ACTIVE = "#2557c1"
ACCENT_TEXT = "#ffffff"
SECONDARY_BG = "#e7e9ee"
SECONDARY_ACTIVE = "#d7dae2"
STATUS_OK = "#1a8a4a"
STATUS_ERROR = "#c13c3c"

RESULTS_CSV_HEADER = "epoch;test_acc;train_acc;loss\n"


def mono_font(size=11):
    '''Returns the first monospace font family installed on this platform
    (Menlo on macOS, Consolas on Windows, DejaVu on Linux), as a tkinter
    font tuple. Must be called after the Tk root exists.

    Args:
        size (int, optional): point size. Defaults to 11.

    Returns:
        tuple: (family, size) for tkinter font options
    '''
    import tkinter.font as tkfont
    available = set(tkfont.families())
    for family in ("Menlo", "Consolas", "DejaVu Sans Mono", "Courier New"):
        if family in available:
            return (family, size)
    return ("Courier", size)


def apply_style(widget):
    """Configure the ttk styles the frontend's widgets use, scoped to
    `widget`'s Tk interpreter. Called once from App.__init__."""
    import tkinter as tk
    from tkinter import ttk

    style = ttk.Style(widget)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("App.TFrame", background=BG)
    style.configure("TFrame", background=BG)

    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("", 20, "bold"))
    style.configure("Status.TLabel", background=BG, foreground=MUTED_TEXT, font=("", 10))

    style.configure(
        "Card.TLabelframe", background=CARD_BG, bordercolor=BORDER,
        relief="solid", borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label", background=CARD_BG, foreground=TEXT, font=("", 11, "bold"),
    )
    # Labels/frames placed inside a white Card.TLabelframe -- the defaults carry the
    # window-grey background and would show as grey boxes on the white card.
    style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT)
    style.configure("Card.TFrame", background=CARD_BG)

    style.configure(
        "Accent.TButton", background=ACCENT, foreground=ACCENT_TEXT,
        font=("", 10, "bold"), borderwidth=0, focuscolor=ACCENT, padding=8,
    )
    style.map("Accent.TButton", background=[("active", ACCENT_ACTIVE), ("disabled", "#a8bdf0")])

    style.configure(
        "Secondary.TButton", background=SECONDARY_BG, foreground=TEXT,
        font=("", 10), borderwidth=0, focuscolor=SECONDARY_BG, padding=7,
    )
    style.map("Secondary.TButton", background=[("active", SECONDARY_ACTIVE), ("disabled", "#f0f1f3")])


def missing_config_message(device, config, network):
    '''Return a warning message listing which of device/config/network are
    still unset, or None if all three are ready to simulate. All three are
    required -- there is deliberately no silent default fallback: whatever a
    simulation runs with must have been configured or loaded explicitly.

    Args:
        device (_type_): the current device configuration
        config (_type_): the loaded simulation configuration
        network (_type_): the current network configuration

    Returns:
        string: warning message
    '''
    from simulation import device_mode_from_config

    missing = []
    # The "Ideal" (software baseline) mode is the one mode that runs without a device.
    if device is None and device_mode_from_config(config) != "Ideal":
        missing.append("Device")
    if config is None:
        missing.append("Simulation")
    if network is None:
        missing.append("Network")
    if not missing:
        return None
    return "Please load " + ", ".join(missing) + " (File menu) before simulating."


def format_results_row(epoch, test_acc, train_acc, loss):
    """One line of the results CSV matches RESULTS_CSV_HEADER's columns."""
    return f"{epoch};{test_acc};{train_acc};{loss}\n"


def open_file(file_format):
    '''saves the path to a file from a file dialog

    Args:
        file_format (string): file type

    Returns:
        "string": path to the file
    '''
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
        filetypes=[(f"{file_format.upper()} files", f"*{file_format}")]
    )
    if not path:
        return None

    if not path.lower().endswith(file_format.lower()):
        messagebox.showerror("Wrong file type", f"Expected a {file_format} file, got: {path}")
        return None

    return path


def save_file(file_format, data):
    '''write the data to a specific file in format

    Args:
        file_format ("string"): file type
        data (any): data
    '''
    from tkinter import filedialog, messagebox

    path = filedialog.asksaveasfilename(
        defaultextension=file_format,
        filetypes=[(f"{file_format.upper()} files", f"*{file_format}")]
    )
    if not path:
        return

    if not path.lower().endswith(file_format.lower()):
        messagebox.showerror("Wrong file type", f"Expected a {file_format} file, got: {path}")
        return

    with open(path, "w") as f:
        f.write(data)


def section(title):
    '''A short divider line used to separate blocks in the text panels,
    e.g. '── Fitted Device '.ljust(.... , '─').Ä

    Args:
        title (string): title of the divider line

    Returns:
        _type_: divider line
    '''
    return f"── {title} ".ljust(SECTION_WIDTH, "─")


def describe(value):
    '''Render a possibly-None plain value (e.g. offset) for display.

    Args:
        value (any): value to transform

    Returns:
        string: value as string or None if value == None
    '''
    return "None" if value is None else str(value)


def describe_dict(value):
    '''Pretty-print a dict (e.g. a .sim/.net config) as indented JSON
    instead of Python's single-line dict repr.
        Args:
        value (dict): dict to transform

    Returns:
        string: dict as string or None if value == None
    '''
    return "-- none --" if value is None else json.dumps(value, indent=2)


def describe_results(results):
    '''Summarize the results (epoch count + last row) instead of
    dumping the whole (potentially long) results into a small panel.
    Args:
        results (any): results to shorten

    Returns:
        string: values of the last epoch, the number of the epochs, the last test acc, the last train acc, the last loss
    '''
    
    if results is None:
        return "-- none --"
    # drop the "#" metadata lines (see format_results_metadata) and the header row
    rows = [line for line in results.strip().splitlines() if line and not line.startswith("#")][1:]
    if not rows:
        return "-- no epochs recorded yet --"
    epoch, test_acc, train_acc, loss = rows[-1].split(";")
    return (
        f"{len(rows)} epoch(s) recorded\n"
        f"last epoch:  {epoch}\n"
        f"  test acc:  {float(test_acc):.2f}%\n"
        f"  train acc: {float(train_acc):.2f}%\n"
        f"  loss:      {float(loss):.4f}"
    )


def format_results_metadata(device, offset, scale, config, network):
    '''Comment block written at the top of every results CSV so a saved result
    always carries the full run configuration it came from. Every line starts
    with "#", so CSV tools and describe_results() skip them.

    Args:
        device (_type_): the fitted device the run used
        offset (float | None): conductance offset (uS)
        scale (float | None): conductance scale (uS per weight unit)
        config (dict): the simulation config
        network (dict): the network config

    Returns:
        string: "#"-prefixed metadata lines, newline-terminated
    '''
    return (
        f"# HWKit results -- {datetime.now().isoformat(timespec='seconds')}\n"
        f"# device: {device!r}\n"
        f"# offset: {offset!r}; scale: {scale!r}\n"
        f"# sim config: {json.dumps(config)}\n"
        f"# network: {json.dumps(network)}\n"
    )


def format_state_text(fit_device_config, offset, scale, config, network, results):
    '''Build the text shown in the GUI's Current Data panel: everything
    currently held in memory and ready to be saved.

    Args:
        fit_device_config (_type_): current device
        offset (_type_): offset value (uS) -- None for legacy .fit files
        scale (_type_): scale value (uS per normalized weight unit) -- None for legacy .fit files
        config (_type_): current config dict
        network (_type_): current network dict
        results (_type_): last results

    Returns:
        string: formatted output
    '''
    return (
        f"{section('Fitted Device')}\n{describe(fit_device_config)}\n"
        f"offset: {describe(offset)} uS\n"
        f"scale:  {describe(scale)} uS/weight unit\n\n"
        f"{section('Simulation Config (.sim)')}\n{describe_dict(config)}\n\n"
        f"{section('Network (.net)')}\n{describe_dict(network)}\n\n"
        f"{section('Results (.csv)')}\n{describe_results(results)}\n"
    )


def format_analog_summary(info):
    """Compact, narrow-panel rendering of an aihwkit AnalogInfo report.
    Its own str(info) renders a table hundreds of characters wide 

    Uses info.layer_summary/total_tile_number/total_nb_analog -- the
    values AnalogInfo.__init__ already computed -- rather than calling
    info.create_layer_summary()/calculate_num_tiles()/calculate_num_analog()
    again, which would re-register forward hooks and re-run the model,
    double-counting everything.
    
    Args:
     info(string): output of the Analog Summary
    Returns:
     string: formated text 
    """
    lines = [section("Analog Summary")]
    for layer in info.layer_summary:
        d = layer.layer_summary_dict()
        kind = "analog" if d["isanalog"] == "analog" else "digital"
        lines.append(f"{d['name']} ({kind})")
        lines.append(f"  in:  {d['input_size']}")
        lines.append(f"  out: {d['output_size']}")
        if d["isanalog"] == "analog":
            lines.append(f"  tiles: {d['num_tiles']}")
        lines.append("")
    lines.append(section("Totals"))
    lines.append(f"tiles:  {info.total_tile_number}")
    lines.append(f"analog layers: {info.total_nb_analog}")
    return "\n".join(lines)
