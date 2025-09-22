import psutil
import os
import csv
from datetime import datetime

LOG_DIR = "/home/edicoba/Documents/predictive_maintenance/data/monitor_script/"

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def get_log_filename():
    now = datetime.now()
    filename = f"usage_{now.strftime('%Y-%m-%d_%H')}.csv"
    return os.path.join(LOG_DIR, filename)

def get_python_processes():
    """Ambil data proses python beserta CPU, RAM absolut, dan RAM percent."""
    result = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                cpu = proc.cpu_percent(interval=0.1)
                ram_mb = proc.info['memory_info'].rss / 1024 / 1024
                ram_percent = proc.memory_percent()
                result.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'pid': proc.pid,
                    'cpu_percent': round(cpu, 2),
                    'memory_mb': round(ram_mb, 2),
                    'memory_percent': round(ram_percent, 2),
                    'cmdline': cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def write_to_csv(data):
    filename = get_log_filename()
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'pid', 'cpu_percent', 'memory_mb', 'memory_percent', 'cmdline'
        ])
        if not file_exists:
            writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    ensure_log_dir()
    processes = get_python_processes()
    write_to_csv(processes)
