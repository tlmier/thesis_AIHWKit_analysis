from matplotlib.figure import Figure


class Plot:
    ''' Class owing two seperate figures, a training fig accuracy + loss with live sync during the training
        and device fig for the currently loaded LTP / LTD config.
        Can be implemeanted in a Host UI.
        FigureCanvasTkAgg(plot.training_fig, master=...) /
        FigureCanvasTkAgg(plot.device_fig, master=...) -- doesn't manage its own
        window or process
    '''

    def __init__(self):
        self.epochs = []
        self.accs = []
        self.acctr = []
        self.loss = []
        self.training_fig = Figure(figsize=(9, 4))
        self.ax1, self.ax2 = self.training_fig.subplots(1, 2)
        self.device_fig = Figure(figsize=(4.5, 4))
        self.ax3 = self.device_fig.subplots(1, 1)
        self.line_test, = self.ax1.plot([], [], marker='o', label='Test', color='red', linewidth=1)
        self.line_train, = self.ax1.plot([], [], marker='o', label='Train', color='blue', linewidth=1)
        self.line_loss, = self.ax2.plot([], [], marker='o', color='green')
        self.ax1.legend()
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Accuracy %')
        self.ax1.set_title('Training Progress')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_title('Loss')
        self.ax2.set_yscale('log')
        # Dashed band marking min/max of the last 10 test accuracies -- a quick visual
        # stability indicator; the exact numbers live in the legend label.
        self.min = self.ax1.axhline(0, color='r', linestyle='--', linewidth=0.5)
        self.max = self.ax1.axhline(0, color='r', linestyle='--', linewidth=0.5)
        self.last10 = []
        self.ax1_placeholder = self.ax1.text(
            0.5, 0.5, 'No simulation run yet', ha='center', va='center',
            transform=self.ax1.transAxes, color='gray',
        )
        self.ax2_placeholder = self.ax2.text(
            0.5, 0.5, 'No simulation run yet', ha='center', va='center',
            transform=self.ax2.transAxes, color='gray',
        )
        self.set_fit_placeholder('No device fitted/loaded yet')

    def reset_training(self):
        '''Clear only the accuracy/loss panels for a fresh simulation run --
        leaves ax3 (fitted device response) alone, since that reflects
        whatever was last fit/loaded, independent of any particular
        simulation run.'''
        self.epochs = []
        self.accs = []
        self.acctr = []
        self.loss = []
        self.last10 = []
        self.line_test.set_data([], [])
        self.line_train.set_data([], [])
        self.line_loss.set_data([], [])
        self.min.set_ydata([0, 0])
        self.max.set_ydata([0, 0])
        self.ax1.set_title('Training Progress')
        self.ax1_placeholder.set_text('No simulation run yet')
        self.ax2_placeholder.set_text('No simulation run yet')
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

    def set_fit_placeholder(self, message):
        '''Show a placeholder on ax3 for when there's no measurement data to
        plot yet (nothing fitted, or a device loaded from a .fit file with
        no raw pulse-response data attached).
        
        Attr: 
            message (str): message to display
        '''
        self.ax3.clear()
        self.ax3.set_title('Fitted Device Response')
        self.ax3.set_xlabel('Pulse number')
        self.ax3.set_ylabel('Weight (normalized conductance units)')
        self.ax3.text(0.5, 0.5, message, ha='center', va='center', transform=self.ax3.transAxes, color='gray')

    def update_fit(self, pulses_per_direction, ltp_fit, ltd_fit, title, ltp_weights = None, ltd_weights= None):
        '''Draw the LTP/LTD measurement data (already shifted by the fit's
        offset) against the fitted device's simulated response on ax3 --
        called once fitting() finishes.

        Args:
            pulses_per_direction (ing): pulses per LTD LTP courve
            ltp_fit (ArrayLike): fittet Weights of LTP curve
            ltd_fit (ArrayLike): fittet Weights of LTD curve
            title (str): name of window
            ltp_weights (ArrayLike, optional): Real measurment points LTP. Defaults to None.
            ltd_weights (ArrayLike, optional): Real measurment points LTD. Defaults to None.
        '''
        self.ax3.clear()
        if (ltd_weights is not None and ltp_weights is not None):
            self.ax3.plot(pulses_per_direction, ltp_weights, "o", color="blue", label="LTP (data - offset)")
            self.ax3.plot(pulses_per_direction, ltd_weights, "o", color="orange", label="LTD (data - offset)")
            self.ax3.plot(pulses_per_direction, ltp_fit, "-", color="navy", label="LTP (fit)")
            self.ax3.plot(pulses_per_direction, ltd_fit, "-", color="darkorange", label="LTD (fit)")
            self.ax3.set_ylabel("Weight (normalized conductance units)")
        elif (ltd_weights is None and ltp_weights is None):
            self.ax3.plot(pulses_per_direction, ltp_fit, "-", color="navy", label="LTP (fit + noise)")
            self.ax3.plot(pulses_per_direction, ltd_fit, "-", color="darkorange", label="LTD (fit + noise)")
            self.ax3.set_ylabel("Weight")

        else:
            raise ValueError("Only one real measurment 2 or 0 neccesarry")
        self.ax3.set_title(title)
        self.ax3.set_xlabel("Pulse number")
        self.ax3.legend()

    def startup(self, data):
        '''starttup the plot

        Args:
            data (ArrayLike): data to show
        '''
        _, epochs, batch_size, lr = data
        self.ax1.set_title(f'Training Progress (epochs={epochs}, batch={batch_size}, lr={lr})')

    def update(self, data):
        '''called if the ggraph should be 

        Args:
            data (ArrayLike): data wich should be displayed for the next eppoch
        '''
        self.ax1_placeholder.set_text('')
        self.ax2_placeholder.set_text('')
        _, acctes, acctrai, epoch, total_loss = data
        self.accs.append(acctes)
        self.acctr.append(acctrai)
        self.epochs.append(epoch)
        self.loss.append(total_loss)
        self.last10.append(acctes)
        if len(self.last10) == 11:
            self.last10 = self.last10[1:]
        mini = min(self.last10)
        maxi = max(self.last10)
        self.min.set_ydata([mini] * 2)
        self.max.set_ydata([maxi] * 2)
        best_acc = max(self.accs)
        idx = self.accs.index(best_acc)
        delta = maxi - mini
        self.line_test.set_data(self.epochs, self.accs)
        self.line_train.set_data(self.epochs, self.acctr)
        self.line_loss.set_data(self.epochs, self.loss)
        self.line_test.set_label(f'Test Acc, Best:{best_acc:.1f}%, E:{self.epochs[idx]}, delta 10:{delta:.1f}%')
        self.ax1.legend()
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
