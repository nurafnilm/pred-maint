import collections
import time
import datetime
import math
import threading
import paho.mqtt.client as mqtt
import json
import tensorflow.lite as tflite
import numpy as np
import requests
import logging
from logging.handlers import RotatingFileHandler
import os
import tempfile
import csv
import gc

# ========== KONFIGURASI LOGGING ========== #
LOG_FILE = "/home/edicoba/Documents/predictive_maintenance/pm_gas_2model/pm_gas_2model_log.log"
LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 10  # Simpan hingga 10 file backup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ubah level sesuai kebutuhan (INFO, WARNING, ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler = RotatingFileHandler(LOG_FILE, maxBytes=LOG_SIZE, backupCount=LOG_BACKUP_COUNT, encoding='utf-8')
handler.setFormatter(formatter)
logger.addHandler(handler)

# ========== NILAI MIN-MAX UNTUK NORMALISASI MANUAL ========== #
min1_vals = np.array([79.02655792, 22.41605568, -0.720020115, 50.20676422, 22.72220802, 0, 0, 1827.950323, -1.0, -1.0])
max1_vals = np.array([101.333786, 35.29693985, 65.12341309, 99.14396667, 34.02838135, 714, 0, 2600.609292, 1.0, 1.0])

min2_vals = np.array([79.02655792, 22.41605568, -0.720020115, 50.20676422, 22.72220802, 0, 0, 1827.950323])
max2_vals = np.array([101.333786, 35.29693985, 65.12341309, 99.14396667, 34.02838135, 714, 0, 2600.609292])

# ========== KONFIGURASI SIMPAN CSV ========== #
def write_csv_fallback(timestamp, feature, error, threshold):
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
        filename = f"/home/edicoba/Documents/predictive_maintenance/pm_gas_2model/fallback/pm_gas_2model_fallback_{dt.strftime('%Y%m%d_%H')}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "feature", "error", "threshold"])
            writer.writerow([timestamp, feature, error, threshold])
        logger.info(f"[Fallback CSV] Disimpan: {feature} = {error:.4f} ke {filename}")
    except Exception as e:
        logger.error(f"[Fallback CSV] Gagal tulis CSV: {e}")

# ========== BUFFER SENSOR ========== #
class SensorBuffer:
    def __init__(self, capacity=60):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.capacity = capacity

    def _calculate_hour_features(self):
        now = datetime.datetime.now()
        total_seconds = now.hour * 3600 + now.minute * 60 + now.second
        seconds_in_day = 24 * 3600
        return math.sin(2 * math.pi * total_seconds / seconds_in_day), math.cos(2 * math.pi * total_seconds / seconds_in_day)

    def add_data(self, processed_sensor_data):
        hour_sin, hour_cos = self._calculate_hour_features()
        data_with_time = processed_sensor_data.copy()
        data_with_time["hour_sin"] = hour_sin
        data_with_time["hour_cos"] = hour_cos
        with self.lock:
            self.buffer.append(data_with_time)  # Menambah data baru ke buffer

    def get_buffer_contents(self):
        with self.lock:
            return list(self.buffer)

    def size(self):
        with self.lock:
            return len(self.buffer)

    def clear(self):
        with self.lock:
            self.buffer.clear()

# ========== BUFFER PREDIKSI ========== #
class PredictionBuffer:
    def __init__(self, capacity=60):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()

    def add_prediction(self, prediction):
        with self.lock:
            self.buffer.append(prediction)

    def get_all(self):
        with self.lock:
            return list(self.buffer)

    def size(self):
        with self.lock:
            return len(self.buffer)

# ========== FUNGSI TAMBAHAN ========== #
def replace_null(value):
    return -9999 if value is None else value

def clear_unused_memory():
    gc.collect()  # Memaksa garbage collection

def periodic_memory_cleanup():
    while True:
        clear_unused_memory()  # Hanya membersihkan objek yang tidak terpakai
        logger.info("[Memory] Pembersihan memori dilakukan.")
        time.sleep(600)  # Pembersihan memori setiap 10 menit

# ========== KONFIGURASI MQTT DAN MODEL ========== #
MQTT_BROKER = "c-greenproject.org"
MQTT_PORT = 1883
MQTT_TOPIC = "testing_pm/gas_analyzer"
MQTT_USERNAME = "eddystation"
MQTT_PASSWORD = "pwdMQTT@123"

MODEL_CNNLSTM_PATH = "/home/edicoba/Documents/predictive_maintenance/model/predicted_gas_pls_32_32_batch1.tflite"
MODEL_AUTOENCODER_PATH = "/home/edicoba/Documents/predictive_maintenance/model/anomali_gas_pls_64_batch1.tflite"
# ========== LOAD MODEL CNN-LSTM ========== #
interpreter = tflite.Interpreter(model_path=MODEL_CNNLSTM_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========== LOAD MODEL LSTM AUTOENCODER ========== #
ae_interpreter = tflite.Interpreter(model_path=MODEL_AUTOENCODER_PATH)
ae_interpreter.allocate_tensors()
ae_input_details = ae_interpreter.get_input_details()
ae_output_details = ae_interpreter.get_output_details()

# ========== INISIALISASI ========== #
sensor_buffer = SensorBuffer(capacity=60)
prediction_buffer = PredictionBuffer(capacity=60)
buffer_penuh = threading.Event()
anomaly_queue = collections.deque(maxlen=100) 
telegram_notification_event = threading.Event()

pengambilan_offset = 0
pengambilan_size = 60
last_telegram_sent_second = -1  # Inisialisasi detik terakhir pengiriman

# Telegram Bot
TELEGRAM_TOKEN = "7923876453:AAHSC_lUCpI7xnZVTlnziMbQbzLCIawsO9M"
TELEGRAM_CHAT_ID = "1770743884"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

# ========== KONFIGURASI CSV UNTUK PENGIRIMAN TELEGRAM ========== #
def write_telegram_metrics_csv(timestamp, num_messages, execution_time, status):
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
        filename = f"/home/edicoba/Documents/predictive_maintenance/pm_gas_2model/telegram_metrics/pm_gas_2model_telegram_{dt.strftime('%Y%m%d_%H')}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "num_messages", "execution_time", "status"])  # Menambahkan kolom status
            writer.writerow([timestamp, num_messages, execution_time, status])  # Menyimpan data dengan jumlah pesan dan status
        logger.info(f"[Telegram Metrics CSV] Disimpan: {num_messages} pesan, Waktu Eksekusi: {execution_time:.4f} detik, Status: {status} ke {filename}")
    except Exception as e:
        logger.error(f"[Telegram Metrics CSV] Gagal tulis CSV: {e}")

# ========== FUNGSI UNTUK MENGIRIM PESAN TELEGRAM DENGAN METRIK WAKTU ========== #
def send_telegram_message(message_queue, event):
    while True:
        event.wait()
        event.clear()
        anomalies_per_second = {}
        while message_queue:
            timestamp, feature, error, threshold = message_queue.popleft()
            if timestamp not in anomalies_per_second:
                anomalies_per_second[timestamp] = {}
            if feature not in anomalies_per_second[timestamp] or error > anomalies_per_second[timestamp][feature][0]:
                anomalies_per_second[timestamp][feature] = (error, threshold)

        num_messages_sent = 0
        start_time = time.time()  # Mulai hitung waktu pengiriman
        for timestamp, anomaly_data in anomalies_per_second.items():
            anomaly_message = f"[{timestamp}] Anomali Terdeteksi (Tertinggi per Atribut):\n"
            for feature, (error_val, threshold_val) in anomaly_data.items():
                anomaly_message += f"- {feature}: {error_val:.4f} > {threshold_val:.4f}\n"

            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": anomaly_message
            }

            while True:
                try:
                    response = requests.post(TELEGRAM_API_URL, data=payload)
                    if response.status_code == 200:
                        logger.info(f"[Telegram] Pesan berhasil dikirim untuk {timestamp}")
                        num_messages_sent += 1
                        break
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 30))
                        logger.warning(f"[Telegram] Terlalu banyak permintaan, menunggu {retry_after} detik...")
                        time.sleep(retry_after)
                    else:
                        logger.error(f"[Telegram] Gagal kirim pesan ({response.status_code}): {response.text}")
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(f"[Telegram] Error koneksi saat mengirim pesan: {e}")
                    break

        execution_time = time.time() - start_time  # Hitung waktu eksekusi pengiriman pesan
        status = "Sukses" if num_messages_sent > 0 else "Gagal"
        # Menulis metrik pengiriman Telegram ke CSV
        timestamp_now = datetime.datetime.now().isoformat()
        write_telegram_metrics_csv(timestamp_now, num_messages_sent, execution_time, status)
        time.sleep(1)  # Beri sedikit jeda sebelum mengecek event lagi

# ========== CALLBACK MQTT ========== #
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"[MQTT] Terhubung ke broker: {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"[MQTT] Gagal terhubung (kode {rc}). Mencoba terhubung kembali dalam 5 detik...")
        time.sleep(5)
        client.reconnect()

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        processed = {
            "bmp388Pressure": replace_null(data.get("bmp388Pressure")),
            "bmp388Temp": replace_null(data.get("bmp388Temp")),
            "bmp388ApprxAltitude": replace_null(data.get("bmp388ApprxAltitude")),
            "sht85Humi": replace_null(data.get("sht85Humi")),
            "sht85Temp": replace_null(data.get("sht85Temp")),
            "co2": replace_null(data.get("co2")),
            "ch4": replace_null(data.get("ch4")),
            "H2OSHT85": replace_null(data.get("H2OSHT85"))
        }
        sensor_buffer.add_data(processed)
        if sensor_buffer.size() == sensor_buffer.capacity:
            buffer_penuh.set()
    except Exception as e:
        logger.error(f"[MQTT] Error memproses pesan: {e}")

# ========== KONFIGURASI CSV UNTUK INTERFERENSI WAKTU (DENGAN MODEL) ========== #
def write_interference_time_csv(timestamp, execution_time, model_name):
    try:
        dt = datetime.datetime.fromisoformat(timestamp)
        filename = f"/home/edicoba/Documents/predictive_maintenance/pm_gas_2model/interference_time/pm_gas_2model_interference_{dt.strftime('%Y%m%d_%H')}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "execution_time", "model"])  # Menambahkan kolom 'model'
            writer.writerow([timestamp, execution_time, model_name])  # Menyimpan data dengan model dan waktu
        logger.info(f"[Interference Time CSV] Disimpan: Waktu Eksekusi: {execution_time:.4f}, Model: {model_name} ke {filename}")
    except Exception as e:
        logger.error(f"[Interference Time CSV] Gagal tulis CSV: {e}")

# ========== THREAD PROSES CNN-LSTM ========== #
def process_buffer_data():
    global pengambilan_offset
    while True:
        buffer_penuh.wait()
        buffer_penuh.clear()

        data = sensor_buffer.get_buffer_contents()
        if len(data) >= pengambilan_size:
            start = pengambilan_offset % len(data)
            end = (start + pengambilan_size) % len(data)
            if start < end:
                chunk = data[start:end]
            else:
                chunk = data[start:] + data[:end]

            if len(chunk) == pengambilan_size:
                raw_data = np.array([[
                    item.get("bmp388Pressure", -9999),
                    item.get("bmp388Temp", -9999),
                    item.get("bmp388ApprxAltitude", -9999),
                    item.get("sht85Humi", -9999),
                    item.get("sht85Temp", -9999),
                    item.get("co2", -9999),
                    item.get("ch4", -9999),
                    item.get("H2OSHT85", -9999),
                    item.get("hour_sin", -9999),
                    item.get("hour_cos", -9999)
                ] for item in chunk], dtype=np.float32)

                with np.errstate(divide='ignore', invalid='ignore'):
                    diff = max1_vals - min1_vals
                    normalized_input_data = (raw_data - min1_vals) / diff
                    normalized_input_data = np.where(np.isfinite(normalized_input_data), normalized_input_data, 0.0)

                input_data = normalized_input_data.reshape(1, pengambilan_size, -1).astype(np.float32)

                # Mulai waktu untuk model CNN-LSTM
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                execution_time = time.time() - start_time  # Hitung waktu eksekusi

                # Simpan waktu eksekusi ke CSV untuk CNN-LSTM
                timestamp_now = datetime.datetime.now().isoformat()
                write_interference_time_csv(timestamp_now, execution_time, "CNN-LSTM")

                logger.info(f"[Prediksi] Hasil CNN-LSTM: {output}")
                prediction_buffer.add_prediction(output[0])
                pengambilan_offset += 1

        time.sleep(0.1)

# ========== THREAD PROSES LSTM AUTOENCODER ========== #
def process_autoencoder_input():
    thresholds = np.array([0.035646, 0.063383, 0.051670, 0.046786, 0.063226, 0.059197, 0.002998, 0.041745], dtype=np.float32)
    fitur_names = ["bmp388Pressure", "bmp388Temp", "bmp388ApprxAltitude", "sht85Humi",
                    "sht85Temp", "co2", "ch4", "H2OSHT85"]
    global last_telegram_sent_second

    while True:
        if prediction_buffer.size() >= prediction_buffer.buffer.maxlen:
            data_pred = prediction_buffer.get_all()
            scaled_output = np.array(data_pred)
            denormalized_output = scaled_output * (max1_vals[:8] - min1_vals[:8]) + min1_vals[:8]
            normalized_for_ae = (denormalized_output - min2_vals) / (max2_vals - min2_vals)
            normalized_for_ae = np.nan_to_num(normalized_for_ae)
            ae_input = normalized_for_ae.reshape(1, 60, -1).astype(np.float32)
            # Mulai waktu untuk model LSTM Autoencoder
            start_time = time.time()
            ae_interpreter.set_tensor(ae_input_details[0]['index'], ae_input)
            ae_interpreter.invoke()
            ae_output = ae_interpreter.get_tensor(ae_output_details[0]['index'])
            execution_time = time.time() - start_time  # Hitung waktu eksekusi

            # Simpan waktu eksekusi ke CSV untuk LSTM Autoencoder
            timestamp_now = datetime.datetime.now().isoformat()
            write_interference_time_csv(timestamp_now, execution_time, "LSTM Autoencoder")
            
            error = np.abs(ae_input - ae_output)[0]
            max_error = np.max(error, axis=0)

            timestamp_now = datetime.datetime.now().isoformat()
            current_second = datetime.datetime.now().second
            anomalous_data_this_batch = False
            anomalies_per_batch = {}

            for i, name in enumerate(fitur_names):
                error_val = float(max_error[i])
                threshold_val = float(thresholds[i])
                if error_val > threshold_val:
                    anomalous_data_this_batch = True
                    write_csv_fallback(timestamp_now, name, error_val, threshold_val)
                    if timestamp_now not in anomalies_per_batch:
                        anomalies_per_batch[timestamp_now] = {}
                    anomalies_per_batch[timestamp_now][name] = (error_val, threshold_val)

            if anomalous_data_this_batch and current_second != last_telegram_sent_second:
                for timestamp, anomaly_data in anomalies_per_batch.items():
                    for feature, (error_val, threshold_val) in anomaly_data.items():
                        anomaly_queue.append((timestamp, feature, error_val, threshold_val))
                telegram_notification_event.set()
                last_telegram_sent_second = current_second

        time.sleep(0.1)  # Sesuaikan sleep time agar mendekati pemrosesan per detik

# ========== MAIN ========== #
def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning(f"[MQTT] Terputus dari broker secara tidak normal (kode {rc}). Mencoba terhubung kembali dalam 5 detik...")
        time.sleep(5)
        client.reconnect()
    else:
        logger.info("[MQTT] Disconnect normal.")

def on_log(client, userdata, level, buf):
    logger.debug(f"[MQTT-LOG] {buf}")

if __name__ == "__main__":
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_log = on_log  # Debug log MQTT

    try:
        logger.info("[MQTT] Menunggu 5 detik sebelum mencoba koneksi awal...")
        time.sleep(5)
        logger.info("[MQTT] Mencoba konek ke broker...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        logger.error(f"[MQTT] Koneksi awal gagal: {e}")
        # Tidak perlu exit() di sini, karena on_connect akan mencoba reconnect jika gagal

    threading.Thread(target=periodic_memory_cleanup, daemon=True).start()
    threading.Thread(target=process_buffer_data, daemon=True).start()
    threading.Thread(target=process_autoencoder_input, daemon=True).start()
    threading.Thread(target=send_telegram_message, args=(anomaly_queue, telegram_notification_event), daemon=True).start()
    client.loop_forever()
