import sys
import numpy as np
from obspy import read, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
from DWT import wavedec, wrcoef
import pywt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QGridLayout, QFileDialog, QTextEdit,
    QMenuBar, QAction, QComboBox
)
from PyQt5.QtCore import Qt
import csv
import json


class SeismicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DWT Algorithm: Auto & Manual Picking")
        self.setGeometry(100, 100, 1200, 800)
        self.file_path = None
        self.manual_pick_time = None
        self.auto_pick_index = None
        self.default_pick_index = None  # Store default STA/LTA pick

        # Create menu bar
        self.create_menu_bar()

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        main_layout = QHBoxLayout()

        # Left panel: Controls
        control_panel = QWidget()
        self.control_layout = QVBoxLayout()

        input_grid = QGridLayout()

        input_grid.addWidget(QLabel("Wavelet Type"), 0, 0)
        self.wavelet_family_combo = QComboBox()
        self.wavelet_family_combo.addItems(["db", "sym", "coif"])
        input_grid.addWidget(self.wavelet_family_combo, 0, 1)

        input_grid.addWidget(QLabel("Degree"), 1, 0)
        self.degree_input = QLineEdit("2")
        input_grid.addWidget(self.degree_input, 1, 1)

        input_grid.addWidget(QLabel("Threshold"), 2, 0)
        self.threshold_input = QLineEdit("3.5")
        input_grid.addWidget(self.threshold_input, 2, 1)

        input_grid.addWidget(QLabel("STA Length (s)"), 3, 0)
        self.sta_input = QLineEdit("1")
        input_grid.addWidget(self.sta_input, 3, 1)

        input_grid.addWidget(QLabel("LTA Length (s)"), 4, 0)
        self.lta_input = QLineEdit("10")
        input_grid.addWidget(self.lta_input, 4, 1)

        self.control_layout.addLayout(input_grid)

        # Buttons
        button_layout = QHBoxLayout()
        self.process_button = QPushButton("Process", clicked=self.process_data)
        button_layout.addWidget(self.process_button)

        self.save_csv_button = QPushButton("Save to CSV", clicked=self.save_to_csv)
        button_layout.addWidget(self.save_csv_button)

        self.save_json_button = QPushButton("Save to JSON", clicked=self.save_to_json)
        button_layout.addWidget(self.save_json_button)

        self.reset_button = QPushButton("Reset", clicked=self.reset)
        button_layout.addWidget(self.reset_button)

        self.control_layout.addLayout(button_layout)

        # Result Field
        result_layout = QGridLayout()
        result_layout.addWidget(QLabel("Picking Time:"), 0, 0)
        self.pick_result = QTextEdit(readOnly=True)
        result_layout.addWidget(self.pick_result, 1, 0)

        self.control_layout.addLayout(result_layout)
        control_panel.setLayout(self.control_layout)

        # Right panel: Plots + Toolbar
        plot_panel = QWidget()
        plot_layout = QVBoxLayout()

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6))
        self.ax1 = self.fig.add_subplot(311)  # Original signal
        self.ax2 = self.fig.add_subplot(312)  # DWT Detail Level 1
        self.ax3 = self.fig.add_subplot(313)  # STA/LTA comparison

        self.canvas = FigureCanvas(self.fig)

        # Add zoom/pan toolbar
        self.toolbar = NavigationToolbar(self.canvas, plot_panel)

        # Connect mouse click
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click_ax2)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_panel.setLayout(plot_layout)

        # Assemble layout
        main_layout.addWidget(control_panel, 30)
        main_layout.addWidget(plot_panel, 70)
        self.central_widget.setLayout(main_layout)

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load File", self)
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open MiniSEED File", "", "MiniSEED Files (*.mseed)"
        )
        if not file_name:
            return
        self.file_path = file_name
        print(f"Selected file: {file_name}")
        self.reset()
        self.process_data()

    def process_data(self):
        if not self.file_path:
            print("No file selected.")
            return

        try:
            st = read(self.file_path)
            tr = st[0]

            sampling_rate = tr.stats.sampling_rate
            delta = tr.stats.delta
            npts = tr.stats.npts
            time = np.arange(0, npts / sampling_rate, delta)

            threshold = float(self.threshold_input.text())
            sta_len = int(float(self.sta_input.text()) * sampling_rate)
            lta_len = int(float(self.lta_input.text()) * sampling_rate)
            family = self.wavelet_family_combo.currentText()
            level = int(self.degree_input.text())
            w_type = f"{family}{level}"

            # Default STA/LTA (on raw data)
            cft_default = recursive_sta_lta(tr, sta_len, lta_len)
            on_of_default = trigger_onset(cft_default, threshold, 0.2)
            trigger_idx_default = on_of_default[0][0] if len(on_of_default) > 0 else None

            # DWT Processing
            try:
                w = pywt.Wavelet(w_type)
            except ValueError:
                print(f"Invalid wavelet: {w_type}")
                self.pick_result.setText(f"Error: Invalid wavelet '{w_type}'")
                return

            C, L = wavedec(tr, wavelet=w, level=4)
            D1 = wrcoef('d', C, L, wavelet=w, level=1)
            cft_dwt = recursive_sta_lta(D1, sta_len, lta_len)
            on_of_dwt = trigger_onset(cft_dwt, threshold, 0.2)
            trigger_idx_dwt = on_of_dwt[0][0] if len(on_of_dwt) > 0 else None

            # Store picks
            self.default_pick_index = trigger_idx_default
            self.auto_pick_index = trigger_idx_dwt

            # Plotting
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # 1. Original signal
            self.ax1.plot(time, tr.data, color='black', label="Original Signal")
            if trigger_idx_default is not None:
                self.ax1.axvline(x=time[trigger_idx_default], color='blue', linestyle='--', linewidth=2, label="Default STA/LTA")
            if trigger_idx_dwt is not None:
                self.ax1.axvline(x=time[trigger_idx_dwt], color='red', linestyle='--', linewidth=2, label="DWT + STA/LTA")
            if self.manual_pick_time is not None:
                self.ax1.axvline(x=time[self.manual_pick_time], color='green', linestyle='--', linewidth=2, label="Manual Pick")

            self.ax1.set_title(f"Original Signal | Start Time: {tr.stats.starttime}")
            self.ax1.legend(loc="upper right")

            # 2. DWT Detail Level 1
            self.ax2.plot(time, D1, color='green')
            self.ax2.set_title("DWT Detail Level 1")

            # 3. STA/LTA Comparison
            self.ax3.plot(time, cft_default, label="Default STA/LTA", color='blue')
            self.ax3.plot(time, cft_dwt, label="DWT + STA/LTA", color='red')

            # Draw vertical lines for all picks
            if trigger_idx_default is not None:
                self.ax3.axvline(x=time[trigger_idx_default], color='blue', linestyle='--', linewidth=1.5, label="Default Pick")
            if trigger_idx_dwt is not None:
                self.ax3.axvline(x=time[trigger_idx_dwt], color='red', linestyle='--', linewidth=1.5, label="Auto Pick (DWT)")
            if self.manual_pick_time is not None:
                self.ax3.axvline(x=time[self.manual_pick_time], color='green', linestyle='--', linewidth=1.5, label="Manual Pick")

            # Highlight difference between auto and manual picks
            if self.auto_pick_index is not None and self.manual_pick_time is not None:
                min_pick = min(time[self.auto_pick_index], time[self.manual_pick_time])
                max_pick = max(time[self.auto_pick_index], time[self.manual_pick_time])
                self.ax3.axvspan(min_pick, max_pick, color='yellow', alpha=0.3, label="Pick Difference")
                diff = abs(time[self.auto_pick_index] - time[self.manual_pick_time])
                self.ax3.text((min_pick + max_pick) / 2, max(cft_dwt) * 0.9,
                              f"Δ={diff:.2f}s", fontsize=10, ha='center', va='center',
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

            self.ax3.set_title("STA/LTA Comparison")
            self.ax3.set_xlabel("Time (s)")
            self.ax3.set_ylabel("STA/LTA Ratio")
            handles, labels = self.ax3.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax3.legend(by_label.values(), by_label.keys(), loc="upper right")

            self.fig.tight_layout()
            self.canvas.draw()

            # Update result box
            start_time = tr.stats.starttime
            auto_utc = start_time + time[self.auto_pick_index] if self.auto_pick_index is not None else "Not Found"
            manual_utc = start_time + time[self.manual_pick_time] if self.manual_pick_time is not None else "Not Set"
            default_utc = start_time + time[trigger_idx_default] if trigger_idx_default is not None else "Not Detected"

            diff_text = "N/A"
            if self.auto_pick_index is not None and self.manual_pick_time is not None:
                diff = abs(time[self.auto_pick_index] - time[self.manual_pick_time])
                diff_text = f"{diff:.2f} s"

            result_text = f"Start Time: {start_time}\n"
            result_text += f"Default Pick: {time[trigger_idx_default]:.2f}s → {default_utc}\n" if trigger_idx_default is not None else "Default Pick: Not Detected\n"
            result_text += f"Auto Pick (DWT): {time[self.auto_pick_index]:.2f}s → {auto_utc}\n" if self.auto_pick_index is not None else "Auto Pick: Not Detected\n"
            result_text += f"Manual Pick: {time[self.manual_pick_time]:.2f}s → {manual_utc}\n" if self.manual_pick_time is not None else "Manual Pick: Not Set\n"
            result_text += f"Difference: {diff_text}"

            self.pick_result.setText(result_text)

        except Exception as e:
            print(f"Error processing data: {e}")
            self.pick_result.setText(f"Error in processing: {e}")

    def on_click_ax2(self, event):
        """Handle mouse double-click on DWT Detail plot"""
        if event.inaxes != self.ax2:
            return

        if event.dblclick:
            x_click = event.xdata
            if x_click is None:
                return

            tr = read(self.file_path)[0]
            sampling_rate = tr.stats.sampling_rate
            delta = tr.stats.delta
            npts = tr.stats.npts
            time = np.arange(0, npts / sampling_rate, delta)

            idx = np.abs(time - x_click).argmin()
            self.manual_pick_time = idx
            self.process_data()  # Refresh display

    def save_to_csv(self):
        if not self.file_path:
            print("No file loaded.")
            return

        tr = read(self.file_path)[0]
        time = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)

        start_time = tr.stats.starttime
        auto_utc = start_time + time[self.auto_pick_index] if self.auto_pick_index is not None else "Not Found"
        manual_utc = start_time + time[self.manual_pick_time] if self.manual_pick_time is not None else "Not Set"
        default_utc = start_time + time[self.default_pick_index] if self.default_pick_index is not None else "Not Detected"

        out_file = self.file_path.replace(".mseed", "_picks.csv")

        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            writer.writerow(["Start Time", str(start_time)])
            writer.writerow(["Default Pick (s)", time[self.default_pick_index] if self.default_pick_index is not None else "Not Detected"])
            writer.writerow(["Default Pick (UTC)", str(default_utc)])
            writer.writerow(["Auto Pick (s)", time[self.auto_pick_index] if self.auto_pick_index is not None else "Not Found"])
            writer.writerow(["Auto Pick (UTC)", str(auto_utc)])
            writer.writerow(["Manual Pick (s)", time[self.manual_pick_time] if self.manual_pick_time is not None else "Not Set"])
            writer.writerow(["Manual Pick (UTC)", str(manual_utc)])
            if self.auto_pick_index is not None and self.manual_pick_time is not None:
                diff = abs(time[self.auto_pick_index] - time[self.manual_pick_time])
                writer.writerow(["Difference (s)", f"{diff:.2f}"])

        print(f"Picks saved to: {out_file}")
        self.pick_result.append(f"\nPicks saved to: {out_file}")

    def save_to_json(self):
        if not self.file_path:
            print("No file loaded.")
            return

        tr = read(self.file_path)[0]
        time = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)

        start_time = tr.stats.starttime
        auto_utc = start_time + time[self.auto_pick_index] if self.auto_pick_index is not None else "Not Found"
        manual_utc = start_time + time[self.manual_pick_time] if self.manual_pick_time is not None else "Not Set"
        default_utc = start_time + time[self.default_pick_index] if self.default_pick_index is not None else "Not Detected"

        result = {
            "filename": self.file_path,
            "start_time": str(start_time),
            "default_pick_sec": float(time[self.default_pick_index]) if self.default_pick_index is not None else "Not Detected",
            "default_pick_utc": str(default_utc),
            "auto_pick_sec": float(time[self.auto_pick_index]) if self.auto_pick_index is not None else "Not Found",
            "auto_pick_utc": str(auto_utc),
            "manual_pick_sec": float(time[self.manual_pick_time]) if self.manual_pick_time is not None else "Not Set",
            "manual_pick_utc": str(manual_utc),
            "difference_sec": abs(time[self.auto_pick_index] - time[self.manual_pick_time]) if self.auto_pick_index is not None and self.manual_pick_time is not None else "N/A"
        }

        out_file = self.file_path.replace(".mseed", "_picks.json")
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Picks saved to: {out_file}")
        self.pick_result.append(f"\nPicks saved to: {out_file}")

    def reset(self):
        self.manual_pick_time = None
        self.auto_pick_index = None
        self.default_pick_index = None
        self.pick_result.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas.draw()


# Run the app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeismicApp()
    window.show()
    sys.exit(app.exec_())