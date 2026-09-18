'''Frontend of the GUI launches Simulation / read write so on'''

import os
import re
import webbrowser
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from aihwkit.utils.analog_info import analog_summary

from config_loading import dump_network, dump_sim_config, load_network, load_sim_config
from device_fitting import fitting, load_device, probe_device_response, UNIT_MULTIPLIERS, VAR_DEF
from gui_helpers import (
    BG, CARD_BG, MUTED_TEXT, RESULTS_CSV_HEADER, STATUS_ERROR, STATUS_OK, TEXT,
    apply_style, format_analog_summary, format_results_metadata, format_results_row, format_state_text,
    missing_config_message, mono_font, open_file, save_file,
)
from plot import Plot
from simulation import (
    DEFAULT_SIM_CONFIG, DEVICE_MODES, SimulationRunner, build_rpu_config_and_model,
    device_mode_from_config,
)


class App(tk.Frame):
    ''' Class of the GUI, makes the main gui object and saves the current loaded configurations
         Attributes:
            fit_device_config: Properties of the fitted or loaded device and the Device used in Simulation (.fit)
            offset: offset of the relative weight to the physical conductance value 
            config: config of the simulation generated or loaded (.sim)
            network: config of the neural network generated or loaded (.net)
            results: results after simulation (Epoch, Test Acc, Train Acc, Loss)
            runnter: thread of the simulation (set after pressing simulate)
            plot: mpl object to plot data 
            canvas: object to plot data
            canvas_device: object to plot device data
    '''
    def __init__(self, root):
        super().__init__(root, padx=16, pady=16, bg=BG)
        self.pack(fill="both", expand=True)
        root.title("HWKit")
        root.configure(bg=BG)
        root.geometry("1400x800")
        apply_style(self)
        self._build_menu(root)

        self.fit_device_config = None
        self.offset = None
        self.scale = None
        self.config = None
        self.network = None
        self.results = None
        self.runner = None
        ttk.Label(self, text="HWKit", style="Title.TLabel").pack(anchor="w", pady=(0, 12))

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, style="App.TFrame", width=260)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        right = ttk.Frame(body, style="App.TFrame")
        right.pack(side="left", fill="both", expand=True)

        actions = ttk.LabelFrame(left, text="Actions", style="Card.TLabelframe", padding=10)
        actions.pack(fill="x", pady=(0, 10))
        self.fit_button = ttk.Button(
            actions, text="Fit device from file...", style="Accent.TButton", command=self.do_fit,
        )
        self.fit_button.pack(fill="x", pady=3)
        self.simulate_button = ttk.Button(
            actions, text="Simulate (train MLP)", style="Accent.TButton", command=self.do_simulate,
        )
        self.simulate_button.pack(fill="x", pady=3)
        self.cancel_button = ttk.Button(
            actions, text="Cancel simulation", style="Secondary.TButton", command=self.do_cancel, state="disabled",
        )
        self.cancel_button.pack(fill="x", pady=3)

        configure = ttk.LabelFrame(left, text="Configure", style="Card.TLabelframe", padding=10)
        configure.pack(fill="x", pady=(0, 10))
        self.config_sim_button = ttk.Button(
            configure, text="Configure simulation...", style="Secondary.TButton", command=self.do_config_sim,
        )
        self.config_sim_button.pack(fill="x", pady=3)
        self.config_network_button = ttk.Button(
            configure, text="Configure network...", style="Secondary.TButton", command=self.do_config_network,
        )
        self.config_network_button.pack(fill="x", pady=3)

        state_frame = ttk.LabelFrame(left, text="Current Data", style="Card.TLabelframe", padding=10)
        state_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.state_view = tk.Text(
            state_frame, state="disabled", wrap="word", bg=CARD_BG, fg=TEXT,
            relief="flat", highlightthickness=0, font=mono_font(11), padx=2, pady=2,
        )
        self.state_view.pack(fill="both", expand=True)
        self.refresh_state_view()
        notes_frame = ttk.LabelFrame(left, text="Notes", style="Card.TLabelframe", padding=10)
        notes_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.notes_view = tk.Text(
            notes_frame, state="disabled", wrap="word", bg=CARD_BG, fg=TEXT,
            relief="flat", highlightthickness=0, font=mono_font(11), padx=2, pady=2,
        )
        self.notes_view.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(left, textvariable=self.status, style="Status.TLabel")
        self.status_label.pack(fill="x", pady=(0, 8))

        ttk.Button(left, text="Exit", style="Secondary.TButton", command=root.destroy).pack(fill="x")

        plot_frame = ttk.LabelFrame(right, text="Live Training Plot", style="Card.TLabelframe", padding=8)
        plot_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.plot_frame = plot_frame

        device_plot_frame = ttk.LabelFrame(right, text="Fitted Device Response", style="Card.TLabelframe", padding=8)
        device_plot_frame.pack(fill="both", expand=True)
        self.device_plot_frame = device_plot_frame

        self.plot = None
        self.canvas = None
        self.device_canvas = None
        self._embed_plot(Plot())

    def _build_menu(self, root):
        """Native top menu bar (macOS: the system menu bar; Windows: in-window bar)
        with Load/Save grouped into submenus and a Help menu opening the docs."""
        menubar = tk.Menu(root)

        file_menu = tk.Menu(menubar, tearoff=False)
        load_menu = tk.Menu(file_menu, tearoff=False)
        load_menu.add_command(label="Fitted Device (.fit)...", command=self.do_load)
        load_menu.add_command(label="Simulation Config (.sim)...", command=self.do_load_sim_config)
        load_menu.add_command(label="Network (.net)...", command=self.do_load_network)
        file_menu.add_cascade(label="Load", menu=load_menu)
        save_menu = tk.Menu(file_menu, tearoff=False)
        save_menu.add_command(label="Fitted Device (.fit)...", command=self.save_fit)
        save_menu.add_command(label="Simulation Config (.sim)...", command=self.save_sim_config)
        save_menu.add_command(label="Network (.net)...", command=self.save_network)
        save_menu.add_command(
            label="Results (.csv)...", command=lambda: self.save_generic(".csv", self.results, "results"),
        )
        file_menu.add_cascade(label="Save", menu=save_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Manual", command=lambda: self._open_doc("docs.html", "manual"))
        help_menu.add_command(label="API Reference", command=lambda: self._open_doc("docs.html", "reference"))
        menubar.add_cascade(label="Help", menu=help_menu)

        root.config(menu=menubar)

    def _open_doc(self, filename, tab=None):
        """Open a documentation file shipped next to the source in the default
        browser -- works the same on macOS and Windows.

        Args:
            filename (str): file next to the sources, e.g. "docs.html"
            tab (str, optional): fragment appended as #tab -- docs.html uses it to start
                on a predefined tab ("manual" or "reference"), switchable in the page.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(path):
            messagebox.showwarning("Not found", f"{filename} was not found next to the application.")
            return
        # as_uri() builds a valid file:// URL on every platform (a hand-built
        # "file://" + path breaks on Windows drive letters).
        url = Path(path).as_uri()
        if tab:
            url += f"#{tab}"
        webbrowser.open(url)

    def _embed_plot(self, plot):
        """(Re)embed a Plot instance's two figures (training_fig into
        plot_frame, device_fig into device_plot_frame), replacing any
        previously embedded canvases. only used once, at startup."""
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.device_canvas is not None:
            self.device_canvas.get_tk_widget().destroy()
        self.plot = plot
        self.canvas = FigureCanvasTkAgg(self.plot.training_fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()
        self.device_canvas = FigureCanvasTkAgg(self.plot.device_fig, master=self.device_plot_frame)
        self.device_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.device_canvas.draw()

    def refresh_state_view(self):
        """Show what's currently held in memory and ready to be saved, called if new data is loaded/created"""
        text = format_state_text(self.fit_device_config, self.offset, self.scale, self.config, self.network, self.results)
        self.state_view.config(state="normal")
        self.state_view.delete("1.0", "end")
        self.state_view.insert("1.0", text)
        self.state_view.config(state="disabled")
        self.config_sim_button.config(text="Modify Simulation" if self.config is not None else "Configure simulation...")
        self.config_network_button.config(text="Modify Network" if self.network is not None else "Configure network...")

    def set_notes_text(self, text):
        """Replace the Notes panel's content. Called by
        _show_analog_summary() on every Simulate press"""
        self.notes_view.config(state="normal")
        self.notes_view.delete("1.0", "end")
        self.notes_view.insert("1.0", text)
        self.notes_view.config(state="disabled")

    def _set_status(self, text, kind="muted"):
        '''Set the status bar to a text and color of the text

        Args:
            text (_type_): Text to display at status bar
            kind (str, optional): Color of the Text, defined in the gui_helpers.py . Defaults to "muted".
        '''
        color = {"muted": MUTED_TEXT, "ok": STATUS_OK, "error": STATUS_ERROR}[kind]
        self.status.set(text)
        self.status_label.configure(foreground=color)

    def do_fit(self):
        '''do fit if button is pressed, loads conductance values handles the gui response. if currently Simulation or Fitting ongoing button is disabled.
        Opens ONE combined settings window (conductance unit + the four variation
        parameters), fits every supported device type against the data and lets the
        user pick which fitted device to keep.
        '''

        if str(self.fit_button["state"]) == "disabled":
            return
        path = open_file(".txt")
        if not path:
            return
        settings = self._open_fit_settings_dialog()
        if settings is None:
            # Closing the settings window with its ✕ aborts the whole fit -- only
            # "Fit" continues (all-zero variations are a valid choice, not a cancel).
            self._set_status("Fit cancelled.")
            return
        unit, deviation = settings
        self.fit_button.config(state="disabled")
        self.simulate_button.config(state="disabled")
        self._set_status("Fitting...")
        self.update_idletasks()
        try:
            candidates, failures = fitting(path, noise=deviation, unit=unit, plot=False, embed_in=None)
            chosen = self._open_device_choice_dialog(candidates, failures)
            if chosen is None:
                self._set_status("Fit cancelled.")
                return
            self.fit_device_config = chosen["device_config"]
            self.offset = chosen["offset"]
            self.scale = chosen["scale"]
            self.plot.update_fit(
                chosen["pulses_per_direction"], chosen["ltp_fit"], chosen["ltd_fit"], chosen["title"],
                ltp_weights=chosen["ltp_weights"], ltd_weights=chosen["ltd_weights"],
            )
            self.device_canvas.draw_idle()
            self._set_status(f"Fit complete ({chosen['name']}).", "ok")
            self.refresh_state_view()
        except Exception as exc:
            messagebox.showerror("Fitting failed", str(exc))
            self._set_status("Fitting failed.", "error")
        finally:
            self.fit_button.config(state="normal")
            self.simulate_button.config(state="normal")

    def _open_device_choice_dialog(self, candidates, failures=None):
        '''Modal dialog listing every device type device_fitting.fitting() managed to fit
        against the measurement file, best (lowest rmse) first. Selecting a row
        previews that candidate's simulated LTP/LTD response against the measured
        points in the device response plot. "Select" commits the highlighted
        candidate and returns it; closing the window or "Cancel" returns None.
        Device types that could not be fit are listed below the choices with their
        failure reason, so a missing candidate is never a silent mystery.

        Args:
            candidates (list[dict]): candidates as returned by device_fitting.fitting()
            failures (list[tuple], optional): (name, reason) for device types that could
                not be fit, as returned by device_fitting.fitting(). Defaults to None.

        Returns:
            dict | None: the chosen candidate, or None if cancelled
        '''
        window = tk.Toplevel(self)
        window.title("Choose fitted device")
        window.configure(bg=BG)
        window.transient(self)
        window.grab_set()

        frame = ttk.Frame(window, style="App.TFrame", padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Multiple device models were fit to the data. Pick one:").pack(
            anchor="w", pady=(0, 6),
        )

        listbox = tk.Listbox(frame, height=min(8, len(candidates)), width=48, exportselection=False)
        for c in candidates:
            listbox.insert("end", f"{c['name']}  (rmse={c['rmse']:.4g})")
        listbox.pack(fill="both", expand=True)
        listbox.selection_set(0)

        if failures:
            lines = "\n".join(f"{name}: {reason[:70]}" for name, reason in failures)
            ttk.Label(
                frame, text=f"Could not fit:\n{lines}", style="Status.TLabel",
                foreground=MUTED_TEXT, justify="left", wraplength=380,
            ).pack(anchor="w", pady=(6, 0))

        def on_select(event=None):
            index = listbox.curselection()
            if not index:
                return
            c = candidates[index[0]]
            self.plot.update_fit(
                c["pulses_per_direction"], c["ltp_fit"], c["ltd_fit"], c["title"],
                ltp_weights=c["ltp_weights"], ltd_weights=c["ltd_weights"],
            )
            self.device_canvas.draw_idle()

        listbox.bind("<<ListboxSelect>>", on_select)
        on_select()

        result = {"value": None}

        def on_ok():
            index = listbox.curselection()
            result["value"] = candidates[index[0]] if index else candidates[0]
            window.destroy()

        buttons = ttk.Frame(frame, style="App.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Select", style="Accent.TButton", command=on_ok).pack(
            side="left", expand=True, fill="x", padx=(0, 4),
        )
        ttk.Button(buttons, text="Cancel", style="Secondary.TButton", command=window.destroy).pack(
            side="left", expand=True, fill="x", padx=(4, 0),
        )

        self.wait_window(window)
        return result["value"]

    def do_load(self):
        '''open .fit file in dialog'''
        path = open_file(".fit")
        if not path:
            return
        self._load_and_display_device(path)

    def _load_and_display_device(self, path):
        '''Load a device from a .fit file and show its LTP/LTD response (only the Simulation of the Device).
           Shared by the explicit Load button. Also set the .fit_device_config. 

        Args:
            path : path to the .fit file

        Returns:
            device: device loaded from the file
        '''
        try:
            loaded = load_device(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not parse {path}:\n{exc}")
            return None
        device = loaded["device"]
        self.fit_device_config = device
        # None for legacy .fit files, which never stored the physical mapping.
        self.offset = loaded["offset"]
        self.scale = loaded["scale"]
        try:
            ltp_curve, ltd_curve = probe_device_response(device)
            pulses_per_direction = np.arange(1, len(ltp_curve) + 1)
            self.plot.update_fit(
                pulses_per_direction, ltp_curve, ltd_curve,
                f"Device loaded from {path} (simulated response)",
            )
        except Exception as exc:
            self.plot.set_fit_placeholder(f"Device loaded from {path}\n(could not simulate response: {exc})")
        self.device_canvas.draw_idle()
        self.refresh_state_view()
        self._set_status(f"Loaded device from {path}.", "ok")
        return device

    def do_simulate(self):
        ''' starts simulation in a seperate thread if button is pressed, if button is not available (simulation or fitting running) retunrns, checks if all data for simulation is loaded)
            forwards the information to the results as starting the live plot '''
        if str(self.simulate_button["state"]) == "disabled":
            return

        warning = missing_config_message(self.fit_device_config, self.config, self.network)
        if warning is not None:
            messagebox.showwarning("Nothing to simulate", warning)
            return

        self.fit_button.config(state="disabled")
        self.simulate_button.config(state="disabled")
        self._set_status("Simulating")
        self.plot.reset_training()
        self.canvas.draw_idle()
        # Results start with a "#" metadata block: device, offset/scale, sim config and
        # network of THIS run, so every saved .csv is self-documenting.
        self.results = format_results_metadata(
            self.fit_device_config, self.offset, self.scale, self.config, self.network,
        ) + RESULTS_CSV_HEADER

        self.runner = SimulationRunner(self.fit_device_config, self.config, self.network)
        self._show_analog_summary()
        self.runner.start()
        self.cancel_button.config(state="normal")
        self.after(100, self._poll_runner)

    def do_cancel(self):
        '''Signal the running simulation to stop after its current epoch. Whatever
        epochs/results already accumulated (live plot, Current Data panel, Save Results)
        are kept -- this only stops further training, it does not discard anything.'''
        if self.runner is None or self.runner.finished:
            return
        self.runner.cancel()
        self.cancel_button.config(state="disabled")
        self._set_status("Cancelling after the current epoch...")

    def _show_analog_summary(self):
        '''Build the (not-yet-trained) model the same way self.runner is
        about to, and show aihwkit.utils.analog_info.analog_summary()'s
        report of it in the Notes panel.'''
        if self.runner.device_mode == "Ideal":
            self.set_notes_text("Ideal (software) mode -- exact floating-point weights,\nno analog tiles, no periphery non-idealities.")
            return
        try:
            model, rpu_config, spec = build_rpu_config_and_model(
                self.runner.device, self.runner.network, self.runner.dataset,
                self.runner.max_crossbar_size, self.runner.device_mode,
                fast_lr=self.runner.fast_lr, transfer_every=self.runner.transfer_every,
                chopper_prob=self.runner.chopper_prob, refresh_every=self.runner.refresh_every,
                dac_bits=self.runner.dac_bits, dac_noise=self.runner.dac_noise,
                adc_bits=self.runner.adc_bits, adc_noise=self.runner.adc_noise,
                circuit_noise=self.runner.circuit_noise, noisy_backward=self.runner.noisy_backward,
            )
            input_size = (1, spec["channels"], spec["size"], spec["size"])
            summary = analog_summary(model, input_size=input_size, rpu_config=rpu_config)
            self.set_notes_text(format_analog_summary(summary))
        except Exception as exc:
            self.set_notes_text(f"analog_summary failed:\n{exc}")

    def _poll_runner(self):
        '''grab the live data to the plot and save it for the results later, calls itself until simulation thread is finished
        '''
        for data in self.runner.drain():
            if data[0] == 'init':
                self.plot.startup(data)
            elif data[0] == 'run':
                self.plot.update(data)
                _, test_acc, train_acc, epoch, loss = data
                self.results += format_results_row(epoch, test_acc, train_acc, loss)
        self.canvas.draw_idle()

        if self.runner.finished:
            self._on_simulate_done(self.runner.error)
        else:
            self.after(100, self._poll_runner)

    def _on_simulate_done(self, exc):
        ''' resets the button for the simulation and show a error window if occured'''
        self.fit_button.config(state="normal")
        self.simulate_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        self.refresh_state_view()
        if exc is not None:
            messagebox.showerror("Simulation failed", str(exc))
            self._set_status("Simulation failed.", "error")
        elif self.runner is not None and self.runner.cancelled:
            self._set_status("Simulation cancelled -- partial results kept.", "ok")
        else:
            self._set_status("Simulation complete.", "ok")

    def do_load_sim_config(self):
        '''loads the .sim file and throws a error if .sim file could not be loaded, refresh the gui to show the loaded data
        '''
        path = open_file(".sim")
        if not path:
            return
        try:
            self.config = load_sim_config(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load simulation config from {path}:\n{exc}")
            return
        self.refresh_state_view()
        self._set_status(f"Loaded simulation config from {path}.", "ok")

    def do_config_sim(self):
        '''
            Ask for the specified parameters in a seperate window if do_config_sim button is pressed.
            Prefills with the currently loaded/configured self.config if there is one, otherwise
            with DEFAULT_SIM_CONFIG (AIHWKit's own standard values, for the DAC/ADC/circuit noise fields).
        '''
        current = {**DEFAULT_SIM_CONFIG, **(self.config or {})}
        # A .sim may store the mode under an older spelling, so prefill the dropdown with
        # the mode SimulationRunner would actually run it as.
        current["Device Mode"] = device_mode_from_config(self.config)
        fields = [
            ("Training", [
                ("Epochs", "int_pos", str(current["Epochs"])),
                ("LR", "lr", str(current["LR"])),
                ("Optimizer", ("choice", ["AnalogAdam", "AnalogSGD"]), current["Optimizer"]),
                ("Train Batch Size", "int_pos", str(current["Train Batch Size"])),
                ("Test Batch Size", "int_pos", str(current["Test Batch Size"])),
                ("Seed", "int_nonneg", str(current["Seed"])),
            ]),
            ("Data", [
                ("Dataset", ("choice", ["MNIST", "FashionMNIST", "CIFAR10"]), current["Dataset"]),
                ("Train Epoch Size", "int_pos", str(current["Train Epoch Size"])),
                ("Test Epoch Size", "int_pos", str(current["Test Epoch Size"])),
            ]),
            ("Synapse & Device Mode", [
                ("Max Crossbar Size", "int_pos", str(current["Max Crossbar Size"])),
                ("Device Mode", ("choice", list(DEVICE_MODES)), current["Device Mode"]),
                ("Fast LR", "float_nonneg", str(current["Fast LR"])),
                ("Transfer Every", "float_nonneg", str(current["Transfer Every"])),
                ("Chopper Prob", "unit_interval", str(current["Chopper Prob"])),
                ("Refresh Every", "int_nonneg", str(current["Refresh Every"])),
            ]),
            ("Converters & Noise", [
                ("DAC Resolution (bits)", "int_pos", str(current["DAC Resolution (bits)"])),
                ("DAC Noise Std", "unit_interval", str(current["DAC Noise Std"])),
                ("ADC Resolution (bits)", "int_pos", str(current["ADC Resolution (bits)"])),
                ("ADC Noise Std", "unit_interval", str(current["ADC Noise Std"])),
                ("Circuit Read Noise Std", "unit_interval", str(current["Circuit Read Noise Std"])),
                ("Noisy Backward Pass", ("choice", ["Yes", "No"]), current["Noisy Backward Pass"]),
            ]),
        ]
        self._open_key_value_dialog("Simulation Config", fields, target_attr="config")

    def _open_key_value_dialog(self, title, fields, target_attr):
        '''
        opens seperate dialog window to enter the requestet data
        allowed values in list:
          "int_pos"          -- positive integer (> 0)
          "lr"                -- decimal with 0 < value < 1
          "unit_interval"     -- decimal with 0 <= value <= 1 (e.g. noise std deviations)
          "float_nonneg"      -- decimal with value >= 0 (e.g. learning rates or transfer
                                 periods, which may well exceed 1)
          "int_nonneg"        -- integer >= 0 (e.g. refresh periods where 0 means "never")
          ("choice", options) -- dropdown restricted to options
        Args:
            title (string): title of the window
            fields (list): either a flat list of (name, allowed values, default) rows, or a
                list of (section title, rows) groups -- grouped fields render as one labeled
                card per section with two field columns, so long dialogs stay compact
            target_attr (string): where answers should be stored: self.target_attr
        '''

        int_vcmd = (self.register(lambda s: s == "" or s.isdigit()), "%P")
        lr_vcmd = (self.register(lambda s: s == "" or re.fullmatch(r"\d*\.?\d*", s) is not None), "%P")

        window = tk.Toplevel(self)
        window.title(title)
        window.configure(bg=BG)
        window.transient(self)

        rows = ttk.Frame(window, style="App.TFrame", padding=14)
        rows.pack(fill="both", expand=True)

        # A group's second element is a list of rows; a flat row's second element is the
        # kind (a string or a ("choice", ...) tuple), so this test tells the two apart.
        groups = fields if fields and isinstance(fields[0][1], list) else [(None, fields)]

        entries = []
        for section_title, group_fields in groups:
            if section_title is not None:
                holder = ttk.Labelframe(rows, text=section_title, style="Card.TLabelframe", padding=10)
                label_style = "Card.TLabel"
            else:
                holder = ttk.Frame(rows, style="App.TFrame")
                label_style = "TLabel"
            holder.pack(fill="x", pady=(0, 8))

            for index, (key, kind, default) in enumerate(group_fields):
                grid_row, column_pair = divmod(index, 2)
                ttk.Label(holder, text=key, width=20, style=label_style, anchor="e").grid(
                    row=grid_row, column=column_pair * 2, sticky="e", padx=(0, 6), pady=3,
                )

                if isinstance(kind, tuple) and kind[0] == "choice":
                    widget = ttk.Combobox(holder, width=15, state="readonly", values=kind[1])
                    widget.set(default if default in kind[1] else kind[1][0])
                else:
                    vcmd = int_vcmd if kind in ("int_pos", "int_nonneg") else lr_vcmd if kind in ("lr", "unit_interval", "float_nonneg") else None
                    widget = ttk.Entry(holder, width=17)
                    if vcmd is not None:
                        widget.configure(validate="key", validatecommand=vcmd)
                    widget.insert(0, default)
                widget.grid(row=grid_row, column=column_pair * 2 + 1, sticky="w", padx=(0, 14), pady=3)

                entries.append((key, kind, default, widget))

        error_label = ttk.Label(rows, text="", style="Status.TLabel", foreground=STATUS_ERROR)
        error_label.pack(fill="x", pady=(6, 0))

        def on_save():
            '''does a type check if values are allowed, write results back if button save is pressed and closes window.
            A blank entry falls back to that field's default (AIHWKit's own standard value, for the
            DAC/ADC/circuit noise fields) instead of being flagged invalid.
            '''
            result = {}
            invalid = []
            for key, kind, default, widget in entries:
                value = widget.get().strip()
                if value == "" and not (isinstance(kind, tuple) and kind[0] == "choice"):
                    value = default
                if kind == "int_pos":
                    if not str(value).isdigit() or int(value) <= 0:
                        invalid.append(key)
                        continue
                    value = int(value)
                elif kind == "int_nonneg":
                    if not str(value).isdigit():
                        invalid.append(key)
                        continue
                    value = int(value)
                elif kind == "lr":
                    try:
                        parsed = float(value)
                    except ValueError:
                        invalid.append(key)
                        continue
                    if not (0 < parsed < 1):
                        invalid.append(key)
                        continue
                    value = parsed
                elif kind == "unit_interval":
                    try:
                        parsed = float(value)
                    except ValueError:
                        invalid.append(key)
                        continue
                    if not (0 <= parsed <= 1):
                        invalid.append(key)
                        continue
                    value = parsed
                elif kind == "float_nonneg":
                    try:
                        parsed = float(value)
                    except ValueError:
                        invalid.append(key)
                        continue
                    if parsed < 0:
                        invalid.append(key)
                        continue
                    value = parsed
                result[key] = value
            if invalid:
                error_label.configure(text=f"Invalid: {', '.join(invalid)}")
                return
            setattr(self, target_attr, result)
            self.refresh_state_view()
            window.destroy()

        ttk.Button(rows, text="Save", style="Accent.TButton", command=on_save).pack(pady=(10, 0), fill="x")

    def do_load_network(self):
        '''opens dialog to load network and check if loading was succsessful, refreshs gui'''
        path = open_file(".net")
        if not path:
            return
        try:
            self.network = load_network(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load network from {path}:\n{exc}")
            return
        self.refresh_state_view()
        self._set_status(f"Loaded network from {path}.", "ok")




    def do_config_network(self):
        '''One window for the whole network configuration -- replaces the old chain of
        type/count prompts: a type dropdown (MLP/CNN), dynamic layer rows with
        add/remove buttons for the conv and hidden layers, and the activation.
        The conv section only shows while CNN is selected. Prefills from the
        currently loaded/configured self.network if there is one. Save builds the
        same dict network.mlp()/network.cnn() consume and stores it on self.network.
        '''
        existing = self.network or {}
        int_vcmd = (self.register(lambda s: s == "" or s.isdigit()), "%P")
        activations = ["ReLU", "Tanh", "Sigmoid", "LeakyReLU"]
        default_channels = [16, 32, 64, 128]
        default_hidden = [256, 128, 64, 32]
        max_conv_layers = 4  # a 28x28 input survives at most 4 halvings

        window = tk.Toplevel(self)
        window.title("Network Config")
        window.configure(bg=BG)
        window.transient(self)

        rows = ttk.Frame(window, style="App.TFrame", padding=14)
        rows.pack(fill="both", expand=True)

        type_row = ttk.Frame(rows, style="App.TFrame")
        type_row.pack(fill="x", pady=(0, 8))
        ttk.Label(type_row, text="Network type", width=20, anchor="e").pack(side="left", padx=(0, 6))
        type_box = ttk.Combobox(type_row, width=15, state="readonly", values=["MLP", "CNN"])
        type_box.set(existing.get("type", "MLP"))
        type_box.pack(side="left")

        conv_card = ttk.Labelframe(rows, text="Convolutional layers", style="Card.TLabelframe", padding=10)
        conv_rows_frame = ttk.Frame(conv_card, style="Card.TFrame")
        conv_rows_frame.pack(fill="x")
        conv_rows = []

        fc_card = ttk.Labelframe(rows, text="Fully-connected hidden layers", style="Card.TLabelframe", padding=10)
        fc_card.pack(fill="x", pady=(0, 8))
        fc_rows_frame = ttk.Frame(fc_card, style="Card.TFrame")
        fc_rows_frame.pack(fill="x")
        fc_rows = []

        def add_conv_row(channels=None, kernel=None):
            '''appends one (channels, kernel) row, prefilled from args or the defaults'''
            if len(conv_rows) >= max_conv_layers:
                return
            index = len(conv_rows)
            row = ttk.Frame(conv_rows_frame, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"Conv {index + 1}: channels / kernel", width=24,
                      style="Card.TLabel", anchor="e").pack(side="left", padx=(0, 6))
            channels_entry = ttk.Entry(row, width=8, validate="key", validatecommand=int_vcmd)
            channels_entry.insert(0, str(channels if channels is not None else default_channels[index % len(default_channels)]))
            channels_entry.pack(side="left", padx=(0, 4))
            kernel_entry = ttk.Entry(row, width=8, validate="key", validatecommand=int_vcmd)
            kernel_entry.insert(0, str(kernel if kernel is not None else 3))
            kernel_entry.pack(side="left")
            conv_rows.append((row, channels_entry, kernel_entry))

        def remove_conv_row():
            '''removes the last conv row (at least one stays)'''
            if len(conv_rows) > 1:
                conv_rows.pop()[0].destroy()

        def add_fc_row(size=None):
            '''appends one hidden-layer size row, prefilled from arg or the defaults'''
            index = len(fc_rows)
            row = ttk.Frame(fc_rows_frame, style="Card.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"Hidden layer {index + 1} size", width=24,
                      style="Card.TLabel", anchor="e").pack(side="left", padx=(0, 6))
            entry = ttk.Entry(row, width=18, validate="key", validatecommand=int_vcmd)
            entry.insert(0, str(size if size is not None else default_hidden[index % len(default_hidden)]))
            entry.pack(side="left")
            fc_rows.append((row, entry))

        def remove_fc_row():
            '''removes the last hidden-layer row (zero hidden layers is allowed)'''
            if fc_rows:
                fc_rows.pop()[0].destroy()

        def row_buttons(parent, add_command, remove_command):
            '''the +/- button pair below a layer card'''
            buttons = ttk.Frame(parent, style="Card.TFrame")
            buttons.pack(fill="x", pady=(6, 0))
            ttk.Button(buttons, text="+ Add layer", style="Secondary.TButton",
                       command=add_command).pack(side="left", expand=True, fill="x", padx=(0, 4))
            ttk.Button(buttons, text="− Remove last", style="Secondary.TButton",
                       command=remove_command).pack(side="left", expand=True, fill="x", padx=(4, 0))

        row_buttons(conv_card, add_conv_row, remove_conv_row)
        row_buttons(fc_card, add_fc_row, remove_fc_row)

        activation_row = ttk.Frame(rows, style="App.TFrame")
        activation_row.pack(fill="x", pady=(0, 8))
        ttk.Label(activation_row, text="Activation", width=20, anchor="e").pack(side="left", padx=(0, 6))
        activation_box = ttk.Combobox(activation_row, width=15, state="readonly", values=activations)
        activation_box.set(existing.get("activation") if existing.get("activation") in activations else activations[0])
        activation_box.pack(side="left")

        error_label = ttk.Label(rows, text="", style="Status.TLabel", foreground=STATUS_ERROR)
        error_label.pack(fill="x", pady=(6, 0))

        def sync_type(*_):
            '''shows the conv card only while CNN is selected'''
            if type_box.get() == "CNN":
                conv_card.pack(fill="x", pady=(0, 8), before=fc_card)
            else:
                conv_card.pack_forget()

        type_box.bind("<<ComboboxSelected>>", sync_type)

        # prefill rows from the existing network (or sensible starters)
        for spec in existing.get("conv_layers") or [{"out_channels": 16, "kernel_size": 3},
                                                    {"out_channels": 32, "kernel_size": 3}]:
            add_conv_row(spec["out_channels"], spec["kernel_size"])
        for size in existing.get("hidden_sizes") if existing.get("hidden_sizes") is not None else [256, 128]:
            add_fc_row(size)
        sync_type()

        def on_save():
            '''validates all rows, builds the network dict and closes the window'''
            net_type = type_box.get()
            sizes = []
            for _, entry in fc_rows:
                value = entry.get().strip()
                if not value.isdigit() or int(value) <= 0:
                    error_label.configure(text="All hidden layer sizes must be positive integers.")
                    return
                sizes.append(int(value))

            if net_type == "MLP":
                self.network = {"type": "MLP", "hidden_sizes": sizes, "activation": activation_box.get()}
            else:
                conv_layers = []
                spatial = 28
                for _, channels_entry, kernel_entry in conv_rows:
                    channels_value = channels_entry.get().strip()
                    kernel_value = kernel_entry.get().strip()
                    if not channels_value.isdigit() or int(channels_value) <= 0 \
                            or not kernel_value.isdigit() or int(kernel_value) <= 0:
                        error_label.configure(text="Conv channels/kernel sizes must be positive integers.")
                        return
                    # network.cnn() uses padding = kernel // 2, which only preserves the image
                    # size for odd kernels -- an even kernel grows the image by one pixel and
                    # desynchronizes the tracked spatial size, crashing at the flatten.
                    if int(kernel_value) % 2 == 0:
                        error_label.configure(text="Kernel sizes must be odd (padding is kernel // 2).")
                        return
                    conv_layers.append({"out_channels": int(channels_value), "kernel_size": int(kernel_value)})
                    spatial //= 2
                if spatial < 1:
                    error_label.configure(text="Too many conv layers for a 28x28 input.")
                    return
                self.network = {
                    "type": "CNN", "conv_layers": conv_layers, "hidden_sizes": sizes,
                    "activation": activation_box.get(),
                }
            self.refresh_state_view()
            window.destroy()

        ttk.Button(rows, text="Save", style="Accent.TButton", command=on_save).pack(pady=(10, 0), fill="x")

    def save_fit(self):
        '''checks if data is currently in the programm, otherwise warining, opening then safe dialog for .fit data
        '''
        if self.fit_device_config is None:
            messagebox.showwarning("Nothing to save", "No fitted device yet -- run Fit first.")
            return
        # Dict format so the physical mapping travels with the device: offset/scale (uS)
        # map weights back to conductance via G = w * scale + offset. load_device() also
        # still reads the legacy bare-repr format (offset/scale lost there).
        save_file(".fit", repr({
            "device": self.fit_device_config, "offset": self.offset, "scale": self.scale,
        }))

    def save_sim_config(self):
        '''checks if data is currently in the programm, otherwise warining, opening then safe dialog for .sim data
        '''
        if self.config is None:
            messagebox.showwarning("Nothing to save", "No simulation config yet -- use Configure simulation first.")
            return
        save_file(".sim", dump_sim_config(self.config))

    def save_network(self):
        '''checks if data is currently in the programm, otherwise warining, opening then safe dialog for .net data
        '''
        if self.network is None:
            messagebox.showwarning("Nothing to save", "No network config yet -- use Configure network first.")
            return
        save_file(".net", dump_network(self.network))

    def save_generic(self, file_format, data, label):
        '''checks if data is currently in the programm, otherwise warining, opening safe dialog for .file_format data

        Args:
            file_format (string): file format to save
            data (any): data to save
            label (string): label for warning if, none
        '''
        if data is None:
            messagebox.showwarning("Nothing to save", f"No {label} available yet.")
            return
        save_file(file_format, data)

    def _open_fit_settings_dialog(self):
        '''One combined, modal window for everything the fit needs to know:
        the conductance unit of the measurement file and the four device
        variation parameters (0 <= x <= 1, see VAR_DEF). Blocks until closed.

        Returns:
            tuple | None: (unit, {aihwkit field: value, ...}) if "Fit" was
                pressed, None if the window was closed -- the caller treats
                that as cancelling the fit
        '''
        window = tk.Toplevel(self)
        window.title("Fit Settings")
        window.configure(bg=BG)
        window.transient(self)
        window.grab_set()

        rows = ttk.Frame(window, style="App.TFrame", padding=14)
        rows.pack(fill="both", expand=True)

        unit_card = ttk.Labelframe(rows, text="Measurement file", style="Card.TLabelframe", padding=10)
        unit_card.pack(fill="x", pady=(0, 8))
        ttk.Label(unit_card, text="Conductance unit", width=28, style="Card.TLabel", anchor="e").grid(
            row=0, column=0, sticky="e", padx=(0, 6), pady=3,
        )
        unit_box = ttk.Combobox(unit_card, width=15, state="readonly", values=list(UNIT_MULTIPLIERS.keys()))
        unit_box.set("µS")
        unit_box.grid(row=0, column=1, sticky="w", pady=3)

        var_card = ttk.Labelframe(rows, text="Device variation", style="Card.TLabelframe", padding=10)
        var_card.pack(fill="x", pady=(0, 8))
        entries = []
        for index, (label, key, default) in enumerate(VAR_DEF):
            ttk.Label(var_card, text=label, width=28, style="Card.TLabel", anchor="e", wraplength=220).grid(
                row=index, column=0, sticky="e", padx=(0, 6), pady=3,
            )
            widget = ttk.Entry(var_card, width=17)
            widget.insert(0, str(default))
            widget.grid(row=index, column=1, sticky="w", pady=3)
            entries.append((key, widget))

        error_label = ttk.Label(rows, text="", style="Status.TLabel", foreground=STATUS_ERROR)
        error_label.pack(fill="x", pady=(6, 0))

        result = {"value": None}

        def on_fit():
            '''validates the variation values (0 <= x <= 1), stores the result and closes'''
            values = {}
            invalid = []
            for key, widget in entries:
                try:
                    value = float(widget.get().strip())
                except ValueError:
                    invalid.append(key)
                    continue
                if not (0 <= value <= 1):
                    invalid.append(key)
                    continue
                values[key] = value
            if invalid:
                error_label.configure(text=f"Invalid: {', '.join(invalid)}")
                return
            result["value"] = (unit_box.get(), values)
            window.destroy()

        ttk.Button(rows, text="Fit", style="Accent.TButton", command=on_fit).pack(pady=(10, 0), fill="x")

        self.wait_window(window)
        return result["value"]