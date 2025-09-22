import psutil
import datetime
import csv
import os

# Folder tujuan
BASE_DIR = "/home/eddy2025/Documents/data/stats_pi"

def get_stats():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    temp = None

    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
    except:
        temp = -1

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_used_mb": round(mem.used / (1024 * 1024), 2),
        "disk_percent": disk.percent,
        "cpu_temp_c": temp
    }

def get_csv_path():
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H")  # Format: 2025-05-10_14
    filename = f"stats_pi_{date_str}.csv"
    return os.path.join(BASE_DIR, filename)

def write_log(data):
    os.makedirs(BASE_DIR, exist_ok=True)
    file_path = get_csv_path()
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

if __name__ == "__main__":
    stats = get_stats()
    write_log(stats)
    print(f"[{stats['timestamp']}] CPU: {stats['cpu_percent']}%, MEM: {stats['memory_percent']}%, TEMP: {stats['cpu_temp_c']}°C")
