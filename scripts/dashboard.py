import time
import re
import os
import sys
import psutil
from collections import deque
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.ansi import AnsiDecoder

# Configuration
LOG_FILE = os.path.expanduser("~/PhaseGPT/training.log")
PID_FILE = os.path.expanduser("~/PhaseGPT/training.pid")
HISTORY_LEN = 60

class TrainingMonitor:
    def __init__(self):
        self.loss_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.current_step = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.start_time = time.time()
        self.status = "Initializing"
        self.log_buffer = deque(maxlen=15)

    def get_process_stats(self):
        """Fetch CPU/Memory info from the PID file."""
        try:
            if not os.path.exists(PID_FILE):
                return {"status": "STOPPED", "cpu": 0.0, "mem": 0.0, "uptime": "0:00", "pid": "N/A"}
            
            with open(PID_FILE, 'r') as f:
                pid_str = f.read().strip()
                if not pid_str:
                    return {"status": "STOPPED", "cpu": 0.0, "mem": 0.0, "uptime": "0:00", "pid": "N/A"}
                pid = int(pid_str)
            
            proc = psutil.Process(pid)
            with proc.oneshot():
                cpu_percent = proc.cpu_percent()
                mem_info = proc.memory_info()
                create_time = datetime.fromtimestamp(proc.create_time())
                uptime = datetime.now() - create_time
            
            uptime_str = str(uptime).split('.')[0]
            mem_gb = mem_info.rss / (1024 * 1024 * 1024)
            
            return {
                "status": "RUNNING",
                "cpu": cpu_percent,
                "mem": mem_gb,
                "uptime": uptime_str,
                "pid": pid
            }
        except (psutil.NoSuchProcess, ValueError):
            return {"status": "DEAD", "cpu": 0.0, "mem": 0.0, "uptime": "0:00", "pid": "N/A"}

    def parse_logs(self):
        """Read the log file and extract metrics."""
        if not os.path.exists(LOG_FILE):
            return

        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            
            self.log_buffer.clear()
            for line in lines[-15:]:
                self.log_buffer.append(line.strip())

            for line in reversed(lines):
                if "Loss:" in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        try:
                            # "  Epoch 1 "
                            self.current_epoch = int(parts[0].strip().split()[1])
                            # " Step 20 "
                            self.current_step = int(parts[1].strip().split()[1])
                            # " Loss: 1.5778"
                            loss_val = float(parts[2].strip().split()[1])
                            
                            if loss_val != self.current_loss:
                                self.current_loss = loss_val
                                self.loss_history.append(loss_val)
                                self.status = "Training Loop Active"
                            break
                        except:
                            continue
                
                if "Generating Dataset" in line:
                    self.status = "Building Dataset"
                elif "Loading Model" in line:
                    self.status = "Loading Model (FP16)"

    def generate_sparkline(self):
        """Generate a text-based sparkline graph."""
        if not self.loss_history:
            return ""
        
        data = list(self.loss_history)
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            return "─" * len(data)
            
        chars = "  ▂▃▄▅▆▇█"
        spark = ""
        for val in data:
            if val == 0:
                spark += " "
                continue
            normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
            idx = int(normalized * 8)
            idx = min(idx, 8)
            spark += chars[idx]
        
        return spark

def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10)
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout

def main():
    monitor = TrainingMonitor()
    layout = make_layout()

    with Live(layout, refresh_per_second=2, screen=True) as live:
        while True:
            monitor.parse_logs()
            sys_stats = monitor.get_process_stats()
            
            # HEADER
            status_color = "green" if sys_stats['status'] == "RUNNING" else "red"
            header_text = f"[b white]PHASEGPT v1.4[/] [cyan]VOLITIONAL ORACLE TRAINING[/]\n[{status_color}]● {sys_stats['status']}[/] | PID: {sys_stats['pid']} | Uptime: {sys_stats['uptime']}"
            layout["header"].update(Panel(Align.center(header_text), style="bold white on black"))

            # LEFT PANEL (Metrics)
            loss_color = "red" if monitor.current_loss > 5.0 else "yellow" if monitor.current_loss > 1.0 else "green"
            phase_color = "blue" if "Training" in monitor.status else "yellow"
            
            metrics_table = Table.grid(padding=1)
            metrics_table.add_column(style="bold cyan", justify="right")
            metrics_table.add_column(style="bold white")
            metrics_table.add_row("Phase:", f"[{phase_color}]{monitor.status}")
            metrics_table.add_row("Epoch:", str(monitor.current_epoch))
            metrics_table.add_row("Global Step:", str(monitor.current_step))
            metrics_table.add_row("Current Loss:", f"[{loss_color}]{monitor.current_loss:.4f}")
            
            sparkline = monitor.generate_sparkline()
            graph_text = f"[{loss_color}]{sparkline}[/]\n[grey50]History (60 ticks)[/]]"
            
            left_layout = Layout()
            left_layout.split_column(
                Layout(Panel(Align.center(metrics_table), title="Training Metrics", border_style="cyan"), ratio=2),
                Layout(Panel(graph_text, title="Loss Convergence", border_style="white"), ratio=1)
            )
            layout["left"].update(left_layout)

            # RIGHT PANEL (System)
            sys_table = Table.grid(padding=1)
            sys_table.add_column(style="bold magenta", justify="right")
            sys_table.add_column(style="bold white")
            
            cpu_str = f"{sys_stats['cpu']}%"
            mem_str = f"{sys_stats['mem']:.2f} GB"
            
            sys_table.add_row("Device:", "Mac Studio (M4 Max)")
            sys_table.add_row("Backend:", "MPS (Metal)")
            sys_table.add_row("Precision:", "FP16")
            sys_table.add_row("CPU Usage:", cpu_str)
            sys_table.add_row("Memory:", mem_str)
            
            layout["right"].update(Panel(Align.center(sys_table), title="System Telemetry", border_style="magenta"))

            # FOOTER (Logs)
            log_text = Text()
            for line in monitor.log_buffer:
                if "Warning" in line or "warn" in line:
                    log_text.append(line + "\n", style="yellow")
                elif "Error" in line or "CRITICAL" in line:
                    log_text.append(line + "\n", style="bold red")
                elif "Loss" in line:
                    log_text.append(line + "\n", style="green")
                else:
                    log_text.append(line + "\n", style="dim white")
            
            layout["footer"].update(Panel(log_text, title="Live Logs", border_style="grey50"))

            time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
