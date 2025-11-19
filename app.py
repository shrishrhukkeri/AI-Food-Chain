from flask import Flask, render_template, request, jsonify, send_file, Response
import pandas as pd
import json
import os
from datetime import datetime
import numpy as np
from datetime import timedelta
import requests
import re
import google.generativeai as genai
from dotenv import load_dotenv
import io
import threading
import time
import tempfile
import torch
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import joblib
from scipy.special import inv_boxcox
import cv2
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# ==================== SECURITY CAMERA DETECTION ====================
# YOLO model for object detection
detection_model = None
detection_thread = None
detection_running = False
frame_lock = threading.Lock()
current_detection_frame = None
detection_stats = {
    "frames_processed": 0,
    "detections": 0,
    "fps": 0,
    "last_detection": "None",
    "last_detection_time": None,
    "detected_objects": []
}

# Allowed objects to detect
ALLOWED_OBJECTS = ["person", "bird", "dog", "horse", "sheep", "cow",
                   "elephant", "bear", "zebra", "giraffe"]

def init_detection_model():
    """Initialize YOLO model for detection"""
    global detection_model
    try:
        from ultralytics import YOLO
        detection_model = YOLO("yolov8x.pt")
        print("YOLO model loaded successfully")
        return True
    except ImportError:
        print("Warning: ultralytics not installed. Detection will be disabled.")
        return False
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
        return False

def fetch_frame_from_camera(ip, port):
    """Fetch frame from ESP32 camera"""
    try:
        stream_url = f"http://{ip}:{port}/stream"
        capture_url = f"http://{ip}:{port}/capture"
        
        # Try stream first
        cap = cv2.VideoCapture(stream_url)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return frame
        
        # Fallback to capture
        resp = requests.get(capture_url, timeout=3)
        if resp.status_code == 200:
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"Error fetching frame: {e}")
    return None

def detect_objects_thread(ip, port):
    """Detection thread that processes frames"""
    global current_detection_frame, detection_stats, detection_running, detection_model
    
    if detection_model is None:
        return
    
    frame_times = []
    last_alert_time = {}
    
    while detection_running:
        start = time.time()
        
        # Fetch frame from camera
        frame = fetch_frame_from_camera(ip, port)
        
        if frame is None:
            time.sleep(0.1)
            continue
        
        # YOLO inference
        try:
            results = detection_model(frame)[0]
            
            keep = []
            detected_now = []
            
            for box in results.boxes:
                cls = detection_model.names[int(box.cls)]
                if cls in ALLOWED_OBJECTS:
                    keep.append(box)
                    detected_now.append(cls)
            
            results.boxes = keep
            annotated = results.plot()
            
            # Update stats
            detection_stats["frames_processed"] += 1
            if detected_now:
                unique_detections = list(set(detected_now))
                detection_stats["detections"] += len(unique_detections)
                detection_stats["last_detection"] = ", ".join(unique_detections)
                detection_stats["last_detection_time"] = datetime.now().isoformat()
                detection_stats["detected_objects"] = unique_detections
                
                # Trigger alerts for new detections (avoid spam - alert once per minute per object type)
                current_time = time.time()
                for obj in unique_detections:
                    if obj not in last_alert_time or (current_time - last_alert_time[obj]) > 60:
                        last_alert_time[obj] = current_time
                        # Alert will be picked up by frontend polling
            
            # FPS calculation
            frame_times.append(time.time() - start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 0:
                detection_stats["fps"] = 1 / (sum(frame_times) / len(frame_times))
            
            # Overlay stats on frame
            #cv2.putText(annotated, f"FPS: {detection_stats['fps']:.1f}",
            #          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if detected_now:
                cv2.putText(annotated, f"Detected: {detection_stats['last_detection']}",
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save frame
            with frame_lock:
                current_detection_frame = annotated.copy()
                
        except Exception as e:
            print(f"Detection error: {e}")
        
        time.sleep(0.02)

def generate_detection_frames():
    """Generate frames for video feed"""
    global current_detection_frame
    while True:
        with frame_lock:
            if current_detection_frame is None:
                time.sleep(0.01)
                continue
            
            ret, buf = cv2.imencode('.jpg', current_detection_frame)
            if not ret:
                continue
            frame = buf.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route('/api/security/video_feed')
def security_video_feed():
    """Video feed with detection"""
    return Response(generate_detection_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/api/security/stats')
def security_stats():
    """Get detection statistics"""
    return jsonify(detection_stats)

@app.route('/api/security/start_detection', methods=['POST'])
def start_detection():
    """Start detection with camera IP and port"""
    global detection_thread, detection_running
    
    data = request.json
    ip = data.get('ip', '')
    port = data.get('port', '81')
    
    if not ip:
        return jsonify({"error": "IP address required"}), 400
    
    # Stop existing detection if running
    if detection_running:
        detection_running = False
        if detection_thread:
            detection_thread.join(timeout=2)
    
    # Initialize model if needed
    if detection_model is None:
        if not init_detection_model():
            return jsonify({"error": "YOLO model not available"}), 500
    
    # Start detection thread
    detection_running = True
    detection_thread = threading.Thread(target=detect_objects_thread, args=(ip, port), daemon=True)
    detection_thread.start()
    
    return jsonify({"status": "started", "ip": ip, "port": port})

@app.route('/api/security/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_running, current_detection_frame
    
    detection_running = False
    with frame_lock:
        current_detection_frame = None
    
    return jsonify({"status": "stopped"})
# ==================== END SECURITY CAMERA DETECTION ====================

FARM_FILES = {
    'FarmA': 'farm_a_data.csv',
    'FarmB': 'farm_b_data.csv',
    'FarmC': 'farm_c_data.csv',
    'FarmD': 'farm_d_data.csv'
}

# Field name mapping from new CSV format to old internal field names
FIELD_NAME_MAPPING = {
    'Fertilizerkgperha': 'Fertilizer_kg_per_ha',
    'SoilMoisture%': 'SoilMoisture_%',
    'TemperatureC': 'Temperature_C',
    'Rainfallmm': 'Rainfall_mm',
    'Yieldtonnesperha': 'Yield_tonnes_per_ha',
    'PestRiskScore': 'PestRiskScore',
    'HarvestRobotUptime%': 'HarvestRobotUptime_%',
    'StorageTemperatureC': 'StorageTemperature_C',
    'Humidity%': 'Humidity_%',
    'SpoilageRate%': 'SpoilageRate_%',
    'GradingScore': 'GradingScore',
    'PredictedShelfLifedays': 'PredictedShelfLife_days',
    'StorageDays': 'StorageDays',
    'ProcessType': 'ProcessType',
    'PackagingType': 'PackagingType',
    'PackagingSpeedunitspermin': 'PackagingSpeed_units_per_min',
    'DefectRate%': 'DefectRate_%',
    'MachineryUptime%': 'MachineryUptime_%',
    'TransportMode': 'TransportMode',
    'TransportDistancekm': 'TransportDistance_km',
    'FuelUsageLper100km': 'FuelUsage_L_per_100km',
    'DeliveryTimehr': 'DeliveryTime_hr',
    'DeliveryDelayFlag': 'DeliveryDelayFlag',
    'SpoilageInTransit%': 'SpoilageInTransit_%',
    'RetailInventoryunits': 'RetailInventory_units',
    'SalesVelocityunitsperday': 'SalesVelocity_units_per_day',
    'DynamicPricingIndex': 'DynamicPricingIndex',
    'WastePercentage%': 'WastePercentage_%',
    'HouseholdWastekg': 'HouseholdWaste_kg',
    'RecipeRecommendationAccuracy%': 'RecipeRecommendationAccuracy_%',
    'SatisfactionScore010': 'SatisfactionScore_0_10',
    'WasteType': 'WasteType',
    'SegregationAccuracy%': 'SegregationAccuracy_%',
    'UpcyclingRate%': 'UpcyclingRate_%',
    'BiogasOutputm3': 'BiogasOutput_m3',
    'minprice': 'minprice',
    'maxprice': 'maxprice',
    'modalprice': 'modalprice',
    'marketname': 'marketname',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'BatchID': 'BatchID',
    'CropType': 'CropType',
    'FarmLocation': 'FarmLocation',
    'HarvestDate': 'HarvestDate'
}

# In-memory cache for farm data (for efficiency)
_farm_data_cache = None
_farm_data_cache_timestamp = None

def normalize_column_names(df):
    """Rename columns from new CSV format to internal field names"""
    rename_dict = {}
    for old_name, new_name in FIELD_NAME_MAPPING.items():
        if old_name in df.columns:
            rename_dict[old_name] = new_name
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df

def load_farm_data(farm_name):
    """Load CSV data for a specific farm"""
    if farm_name in FARM_FILES and os.path.exists(FARM_FILES[farm_name]):
        df = pd.read_csv(FARM_FILES[farm_name])
        df = normalize_column_names(df)
        if 'HarvestDate' in df.columns:
            df['HarvestDate'] = pd.to_datetime(df['HarvestDate'])
        return df
    return pd.DataFrame()

def load_all_farms_data():
    """Load data from all farms for comparison (cached in memory for efficiency)"""
    global _farm_data_cache, _farm_data_cache_timestamp
    
    # Check if cache is still valid (reload if files were modified)
    current_timestamps = {}
    for farm_name, file_path in FARM_FILES.items():
        if os.path.exists(file_path):
            current_timestamps[farm_name] = os.path.getmtime(file_path)
    
    # Use cache if it exists and files haven't changed
    if _farm_data_cache is not None and _farm_data_cache_timestamp == current_timestamps:
        return _farm_data_cache
    
    # Load fresh data
    all_data = {}
    for farm_name, file_path in FARM_FILES.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df = normalize_column_names(df)
            if 'HarvestDate' in df.columns:
                df['HarvestDate'] = pd.to_datetime(df['HarvestDate'])
            all_data[farm_name] = df
    
    # Update cache
    _farm_data_cache = all_data
    _farm_data_cache_timestamp = current_timestamps
    
    return all_data

def clear_farm_data_cache():
    """Clear the cached farm data (useful if CSV files are updated)"""
    global _farm_data_cache, _farm_data_cache_timestamp
    _farm_data_cache = None
    _farm_data_cache_timestamp = None

@app.route('/')
def index():
    return render_template('index.html')

def calculate_carbon_footprint(data):
    """
    Calculate carbon footprint in kg CO2e based on farm operations
    Returns total carbon footprint and breakdown by category
    """
    if data.empty:
        return {'total': 0, 'fertilizer': 0, 'transportation': 0, 'storage': 0, 'processing': 0, 'waste': 0}
    
    # 1. Fertilizer emissions (N2O from nitrogen fertilizers)
    # Average: 1.5 kg CO2e per kg of fertilizer (assuming mixed fertilizer)
    avg_fertilizer = float(data['Fertilizer_kg_per_ha'].mean())
    avg_yield = float(data['Yield_tonnes_per_ha'].mean())
    # Calculate per tonne of yield
    fertilizer_emissions = (avg_fertilizer * 1.5) if avg_yield > 0 else 0
    
    # 2. Transportation emissions (CO2 from fuel consumption)
    # Diesel: 2.31 kg CO2 per liter, Petrol: 2.31 kg CO2 per liter
    avg_distance = float(data['TransportDistance_km'].mean())
    avg_fuel_usage = float(data['FuelUsage_L_per_100km'].mean())
    # Calculate total fuel used per batch
    total_fuel = (avg_distance / 100) * avg_fuel_usage
    transportation_emissions = total_fuel * 2.31
    
    # Transport mode multipliers (Air > Truck > Train > Ship)
    transport_modes = data['TransportMode'].value_counts(normalize=True)
    mode_multiplier = 1.0
    if 'Air' in transport_modes:
        mode_multiplier += transport_modes['Air'] * 0.5  # Air transport is more carbon intensive
    if 'Truck' in transport_modes:
        mode_multiplier += transport_modes['Truck'] * 0.2
    transportation_emissions *= mode_multiplier
    
    # 3. Storage emissions (energy for refrigeration and climate control)
    # Based on storage days and temperature control
    avg_storage_days = float(data['StorageDays'].mean())
    avg_storage_temp = float(data['StorageTemperature_C'].mean())
    # Lower temperatures require more energy
    temp_factor = max(0, (20 - avg_storage_temp) / 20) if avg_storage_temp < 20 else 0.1
    # Energy consumption: approximately 0.5 kWh per day per tonne (rough estimate)
    storage_energy_kwh = avg_storage_days * temp_factor * 0.5 * (avg_yield if avg_yield > 0 else 1)
    # Grid electricity: approximately 0.5 kg CO2e per kWh (varies by region)
    storage_emissions = storage_energy_kwh * 0.5
    
    # 4. Processing emissions (energy for processing and packaging)
    # Based on processing type and machinery uptime
    avg_machinery_uptime = float(data['MachineryUptime_%'].mean())
    avg_packaging_speed = float(data['PackagingSpeed_units_per_min'].mean())
    # Energy consumption based on processing intensity
    processing_energy_kwh = (avg_machinery_uptime / 100) * 2.0 * (avg_yield if avg_yield > 0 else 1)
    # Processing type multipliers
    process_types = data['ProcessType'].value_counts(normalize=True)
    process_multiplier = 1.0
    if 'Freezing' in process_types:
        process_multiplier += process_types['Freezing'] * 0.8  # Freezing is energy intensive
    if 'Canning' in process_types:
        process_multiplier += process_types['Canning'] * 0.5
    processing_emissions = processing_energy_kwh * 0.5 * process_multiplier
    
    # 5. Waste emissions (methane from organic waste decomposition)
    avg_waste_percentage = float(data['WastePercentage_%'].mean())
    avg_household_waste = float(data['HouseholdWaste_kg'].mean())
    # Methane has 25x GWP of CO2, but only a portion of waste decomposes anaerobically
    # Approximately 0.5 kg CO2e per kg of organic waste
    waste_emissions = (avg_waste_percentage / 100 * avg_yield * 1000 * 0.5) if avg_yield > 0 else 0
    waste_emissions += avg_household_waste * 0.5
    
    # Total carbon footprint (per batch/record, averaged)
    total = fertilizer_emissions + transportation_emissions + storage_emissions + processing_emissions + waste_emissions
    
    return {
        'total': round(total, 2),
        'fertilizer': round(fertilizer_emissions, 2),
        'transportation': round(transportation_emissions, 2),
        'storage': round(storage_emissions, 2),
        'processing': round(processing_emissions, 2),
        'waste': round(waste_emissions, 2)
    }

@app.route('/api/farm/<farm_name>/kpis')
def get_farm_kpis(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    carbon_footprint = calculate_carbon_footprint(data)
    kpis = {
        'total_production': float(data['Yield_tonnes_per_ha'].mean()),
        'storage_spoilage': float(data['SpoilageRate_%'].mean()),
        'processing_defects': float(data['DefectRate_%'].mean()),
        'transport_delays': float((data['DeliveryDelayFlag'].sum() / len(data) * 100)),
        'retail_inventory': float(data['RetailInventory_units'].sum()),
        'waste_percentage': float(data['WastePercentage_%'].mean()),
        'satisfaction': float(data['SatisfactionScore_0_10'].mean()),
        'waste_segregation': float(data['SegregationAccuracy_%'].mean()),
        'total_records': len(data),
        'pest_risk': float(data['PestRiskScore'].mean()),
        'machinery_uptime': float(data['MachineryUptime_%'].mean()),
        'harvest_uptime': float(data['HarvestRobotUptime_%'].mean()),
        'carbon_footprint': carbon_footprint
    }
    return jsonify(kpis)

@app.route('/api/farm/<farm_name>/production')
def get_farm_production_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    yield_by_crop = data.groupby('CropType')['Yield_tonnes_per_ha'].mean().to_dict()
    return jsonify({'crop_types': data['CropType'].unique().tolist(), 'yield_by_crop': yield_by_crop})

@app.route('/api/farm/<farm_name>/storage')
def get_farm_storage_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    return jsonify({
        'avg_temp': float(data['StorageTemperature_C'].mean()),
        'avg_humidity': float(data['Humidity_%'].mean()),
        'avg_spoilage': float(data['SpoilageRate_%'].mean()),
        'avg_shelf_life': float(data['PredictedShelfLife_days'].mean())
    })

@app.route('/api/farm/<farm_name>/processing')
def get_farm_processing_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    defect_by_process = data.groupby('ProcessType')['DefectRate_%'].mean().to_dict()
    return jsonify({
        'avg_defect_rate': float(data['DefectRate_%'].mean()),
        'avg_uptime': float(data['MachineryUptime_%'].mean()),
        'avg_packaging_speed': float(data['PackagingSpeed_units_per_min'].mean()),
        'defect_by_process': defect_by_process
    })

@app.route('/api/farm/<farm_name>/transportation')
def get_farm_transportation_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    return jsonify({
        'avg_distance': float(data['TransportDistance_km'].mean()),
        'avg_fuel': float(data['FuelUsage_L_per_100km'].mean()),
        'avg_delivery_time': float(data['DeliveryTime_hr'].mean()),
        'delay_percentage': float((data['DeliveryDelayFlag'].sum() / len(data) * 100))
    })

@app.route('/api/farm/<farm_name>/retail')
def get_farm_retail_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    return jsonify({
        'total_inventory': float(data['RetailInventory_units'].sum()),
        'avg_sales_velocity': float(data['SalesVelocity_units_per_day'].mean()),
        'avg_pricing_index': float(data['DynamicPricingIndex'].mean()),
        'avg_waste': float(data['WastePercentage_%'].mean())
    })

@app.route('/api/farm/<farm_name>/consumption')
def get_farm_consumption_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    return jsonify({
        'avg_household_waste': float(data['HouseholdWaste_kg'].mean()),
        'avg_recipe_accuracy': float(data['RecipeRecommendationAccuracy_%'].mean()),
        'avg_satisfaction': float(data['SatisfactionScore_0_10'].mean())
    })

@app.route('/api/farm/<farm_name>/waste')
def get_farm_waste_data(farm_name):
    data = load_farm_data(farm_name)
    if data.empty:
        return jsonify({'error': 'Farm not found'}), 404
    
    waste_dist = data['WasteType'].value_counts().to_dict()
    return jsonify({
        'avg_segregation': float(data['SegregationAccuracy_%'].mean()),
        'avg_upcycling': float(data['UpcyclingRate_%'].mean()),
        'avg_biogas': float(data['BiogasOutput_m3'].mean()),
        'waste_dist': waste_dist
    })

@app.route('/api/comparison/<section>')
def get_comparison_data(section):
    """Get comparison data for all farms for a specific section"""
    all_farms = load_all_farms_data()
    comparison = {}
    
    for farm_name, data in all_farms.items():
        if section == 'production':
            yield_by_crop = data.groupby('CropType')['Yield_tonnes_per_ha'].mean().to_dict()
            comparison[farm_name] = {
                'yield': float(data['Yield_tonnes_per_ha'].mean()),
                'pest_risk': float(data['PestRiskScore'].mean()),
                'harvest_uptime': float(data['HarvestRobotUptime_%'].mean()),
                'machinery_uptime': float(data['MachineryUptime_%'].mean()),
                'yield_by_crop': yield_by_crop
            }
        elif section == 'storage':
            comparison[farm_name] = {
                'avg_temp': float(data['StorageTemperature_C'].mean()),
                'avg_humidity': float(data['Humidity_%'].mean()),
                'avg_spoilage': float(data['SpoilageRate_%'].mean()),
                'avg_shelf_life': float(data['PredictedShelfLife_days'].mean())
            }
        elif section == 'processing':
            defect_by_process = data.groupby('ProcessType')['DefectRate_%'].mean().to_dict()
            comparison[farm_name] = {
                'avg_defect_rate': float(data['DefectRate_%'].mean()),
                'avg_uptime': float(data['MachineryUptime_%'].mean()),
                'avg_packaging_speed': float(data['PackagingSpeed_units_per_min'].mean()),
                'defect_by_process': defect_by_process
            }
        elif section == 'transportation':
            comparison[farm_name] = {
                'avg_distance': float(data['TransportDistance_km'].mean()),
                'avg_fuel': float(data['FuelUsage_L_per_100km'].mean()),
                'avg_delivery_time': float(data['DeliveryTime_hr'].mean()),
                'delay_percentage': float((data['DeliveryDelayFlag'].sum() / len(data) * 100))
            }
        elif section == 'retail':
            comparison[farm_name] = {
                'total_inventory': float(data['RetailInventory_units'].sum()),
                'avg_sales_velocity': float(data['SalesVelocity_units_per_day'].mean()),
                'avg_pricing_index': float(data['DynamicPricingIndex'].mean()),
                'avg_waste': float(data['WastePercentage_%'].mean())
            }
        elif section == 'consumption':
            comparison[farm_name] = {
                'avg_household_waste': float(data['HouseholdWaste_kg'].mean()),
                'avg_recipe_accuracy': float(data['RecipeRecommendationAccuracy_%'].mean()),
                'avg_satisfaction': float(data['SatisfactionScore_0_10'].mean())
            }
        elif section == 'waste':
            waste_dist = data['WasteType'].value_counts().to_dict()
            comparison[farm_name] = {
                'avg_segregation': float(data['SegregationAccuracy_%'].mean()),
                'avg_upcycling': float(data['UpcyclingRate_%'].mean()),
                'avg_biogas': float(data['BiogasOutput_m3'].mean()),
                'waste_dist': waste_dist
            }
    
    return jsonify(comparison)

@app.route('/api/overview')
def get_overview():
    """Get comparison data for all farms"""
    all_farms = load_all_farms_data()
    overview = {}
    
    for farm_name, data in all_farms.items():
        carbon_footprint = calculate_carbon_footprint(data)
        overview[farm_name] = {
            'yield': float(data['Yield_tonnes_per_ha'].mean()),
            'spoilage': float(data['SpoilageRate_%'].mean()),
            'defects': float(data['DefectRate_%'].mean()),
            'delays': float((data['DeliveryDelayFlag'].sum() / len(data) * 100)),
            'waste': float(data['WastePercentage_%'].mean()),
            'satisfaction': float(data['SatisfactionScore_0_10'].mean()),
            'pest_risk': float(data['PestRiskScore'].mean()),
            'machinery_uptime': float(data['MachineryUptime_%'].mean()),
            'total_records': len(data),
            'performance_score': 0,  # Will calculate below
            'carbon_footprint': carbon_footprint
        }
    
    # Calculate performance scores (0-100)
    if overview:
        max_yield = max(f['yield'] for f in overview.values())
        min_spoilage = min(f['spoilage'] for f in overview.values())
        min_defects = min(f['defects'] for f in overview.values())
        min_delays = min(f['delays'] for f in overview.values())
        min_waste = min(f['waste'] for f in overview.values())
        max_satisfaction = max(f['satisfaction'] for f in overview.values())
        min_pest = min(f['pest_risk'] for f in overview.values())
        max_uptime = max(f['machinery_uptime'] for f in overview.values())
        
        for farm_name in overview:
            f = overview[farm_name]
            score = (
                (f['yield'] / max_yield * 20 if max_yield > 0 else 0) +
                ((1 - f['spoilage'] / 30) * 15 if f['spoilage'] < 30 else 0) +
                ((1 - f['defects'] / 15) * 15 if f['defects'] < 15 else 0) +
                ((1 - f['delays'] / 30) * 10 if f['delays'] < 30 else 0) +
                ((1 - f['waste'] / 25) * 10 if f['waste'] < 25 else 0) +
                (f['satisfaction'] / 10 * 15) +
                ((1 - f['pest_risk'] / 100) * 10 if f['pest_risk'] < 100 else 0) +
                (f['machinery_uptime'] / 100 * 5)
            )
            f['performance_score'] = round(score, 1)
    
    return jsonify(overview)

@app.route('/api/ai-insights/<farm_name>/<section>')
def get_ai_insights(farm_name, section):
    """Generate smart AI insights based on actual data analysis"""
    if farm_name == 'all':
        all_farms = load_all_farms_data()
        return generate_comparison_insights(all_farms, section)
    else:
        data = load_farm_data(farm_name)
        if data.empty:
            return {'insights': [], 'recommendations': []}
        return generate_farm_insights(farm_name, data, section)

def generate_comparison_insights(all_farms, section):
    """Generate concise insights comparing all farms"""
    insights = []
    recommendations = []
    farm_insights = {}  # One-line insights per farm
    
    # Calculate metrics for all farms
    farms_data = {}
    for farm_name, data in all_farms.items():
        farms_data[farm_name] = {
            'yield': float(data['Yield_tonnes_per_ha'].mean()),
            'spoilage': float(data['SpoilageRate_%'].mean()),
            'defects': float(data['DefectRate_%'].mean()),
            'delays': float((data['DeliveryDelayFlag'].sum() / len(data) * 100)),
            'waste': float(data['WastePercentage_%'].mean()),
            'satisfaction': float(data['SatisfactionScore_0_10'].mean()),
            'pest_risk': float(data['PestRiskScore'].mean()),
            'machinery_uptime': float(data['MachineryUptime_%'].mean()),
            'harvest_uptime': float(data['HarvestRobotUptime_%'].mean()),
            'storage_temp': float(data['StorageTemperature_C'].mean()),
            'humidity': float(data['Humidity_%'].mean()),
            'shelf_life': float(data['PredictedShelfLife_days'].mean()),
            'segregation': float(data['SegregationAccuracy_%'].mean()),
            'upcycling': float(data['UpcyclingRate_%'].mean()),
            'biogas': float(data['BiogasOutput_m3'].mean()),
        }
        farm_insights[farm_name] = []
    
    # Production insights
    if section == 'production':
        best_yield = max(farms_data.items(), key=lambda x: x[1]['yield'])
        worst_yield = min(farms_data.items(), key=lambda x: x[1]['yield'])
        worst_pest = max(farms_data.items(), key=lambda x: x[1]['pest_risk'])
        best_pest = min(farms_data.items(), key=lambda x: x[1]['pest_risk'])
        worst_uptime = min(farms_data.items(), key=lambda x: x[1]['harvest_uptime'])
        best_uptime = max(farms_data.items(), key=lambda x: x[1]['harvest_uptime'])
        
        insights.append(f"🏆 {best_yield[0]} leads with highest yield ({best_yield[1]['yield']:.1f}t/ha)")
        if worst_yield[1]['yield'] < 5:
            insights.append(f"⚠️ {worst_yield[0]} needs yield optimization ({worst_yield[1]['yield']:.1f}t/ha)")
            recommendations.append("Optimize soil nutrition and crop rotation")
            recommendations.append("Consider precision agriculture techniques")
        else:
            insights.append(f"📊 {worst_yield[0]} has lower yield ({worst_yield[1]['yield']:.1f}t/ha) - improvement opportunity")
            recommendations.append("Improve farming practices and soil quality")
        
        if worst_pest[1]['pest_risk'] > 50:
            insights.append(f"🐛 {worst_pest[0]} has high pest risk ({worst_pest[1]['pest_risk']:.0f}) - requires attention")
            recommendations.append("Implement integrated pest management (IPM)")
        if best_pest[1]['pest_risk'] < 20:
            insights.append(f"✅ {best_pest[0]} has excellent pest control (risk: {best_pest[1]['pest_risk']:.0f})")
        
        if worst_uptime[1]['harvest_uptime'] < 85:
            insights.append(f"⚙️ {worst_uptime[0]} has low harvest uptime ({worst_uptime[1]['harvest_uptime']:.0f}%) - maintenance needed")
            recommendations.append("Schedule preventive equipment maintenance")
        if best_uptime[1]['harvest_uptime'] > 95:
            insights.append(f"✅ {best_uptime[0]} has excellent machinery uptime ({best_uptime[1]['harvest_uptime']:.0f}%)")
        
        # Yield gap analysis
        yield_gap = best_yield[1]['yield'] - worst_yield[1]['yield']
        if yield_gap > 2:
            insights.append(f"📈 Yield gap of {yield_gap:.1f}t/ha between best and worst performers")
            recommendations.append("Share best practices across farms")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['yield'] == worst_yield[1]['yield'] and metrics['yield'] < 5:
                farm_insights[farm] = f"Low yield ({metrics['yield']:.1f}t/ha)"
            elif metrics['pest_risk'] == worst_pest[1]['pest_risk'] and metrics['pest_risk'] > 50:
                farm_insights[farm] = f"High pest risk ({metrics['pest_risk']:.0f})"
            elif metrics['harvest_uptime'] > 95:
                farm_insights[farm] = f"Excellent uptime ({metrics['harvest_uptime']:.0f}%)"
            else:
                farm_insights[farm] = f"Yield: {metrics['yield']:.1f}t/ha"
    
    # Storage insights
    elif section == 'storage':
        best_spoilage = min(farms_data.items(), key=lambda x: x[1]['spoilage'])
        worst_spoilage = max(farms_data.items(), key=lambda x: x[1]['spoilage'])
        best_temp = min(farms_data.items(), key=lambda x: abs(x[1]['storage_temp'] - 3.5))
        # Find worst temp (furthest from optimal range 2-5°C)
        worst_temp_candidates = [(f, m) for f, m in farms_data.items() if m['storage_temp'] < 2 or m['storage_temp'] > 5]
        if worst_temp_candidates:
            worst_temp = max(worst_temp_candidates, key=lambda x: abs(x[1]['storage_temp'] - 3.5))
        else:
            worst_temp = max(farms_data.items(), key=lambda x: abs(x[1]['storage_temp'] - 3.5))
        best_humidity = min(farms_data.items(), key=lambda x: abs(x[1]['humidity'] - 80))
        
        insights.append(f"✅ {best_spoilage[0]} has lowest spoilage ({best_spoilage[1]['spoilage']:.1f}%) - best practice")
        if worst_spoilage[1]['spoilage'] > 15:
            insights.append(f"❌ {worst_spoilage[0]} has critical spoilage ({worst_spoilage[1]['spoilage']:.1f}%) - immediate action needed")
            recommendations.append("Review cold chain integrity and storage protocols")
            recommendations.append("Implement real-time temperature monitoring")
        elif worst_spoilage[1]['spoilage'] > 10:
            insights.append(f"⚠️ {worst_spoilage[0]} needs spoilage reduction ({worst_spoilage[1]['spoilage']:.1f}%)")
            recommendations.append("Optimize storage conditions and temperature control")
        else:
            insights.append(f"📊 {worst_spoilage[0]} has spoilage at {worst_spoilage[1]['spoilage']:.1f}% - within range")
        
        # Temperature optimization
        if abs(best_temp[1]['storage_temp'] - 3.5) < 1:
            insights.append(f"🌡️ {best_temp[0]} maintains optimal temperature ({best_temp[1]['storage_temp']:.1f}°C)")
        
        worst_temp_val = worst_temp[1]['storage_temp']
        if worst_temp_val < 2 or worst_temp_val > 5:
            insights.append(f"🌡️ {worst_temp[0]} has temperature deviation ({worst_temp_val:.1f}°C) - calibration needed")
            recommendations.append("Calibrate refrigeration systems")
        elif abs(worst_temp_val - 3.5) > 1.5:
            insights.append(f"🌡️ {worst_temp[0]} temperature at {worst_temp_val:.1f}°C - can be optimized")
        
        # Humidity optimization
        if abs(best_humidity[1]['humidity'] - 80) < 5:
            insights.append(f"💧 {best_humidity[0]} maintains optimal humidity ({best_humidity[1]['humidity']:.0f}%)")
        
        # Shelf life comparison
        best_shelf = max(farms_data.items(), key=lambda x: x[1]['shelf_life'])
        insights.append(f"📅 {best_shelf[0]} achieves longest shelf life ({best_shelf[1]['shelf_life']:.1f} days)")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['spoilage'] == worst_spoilage[1]['spoilage'] and metrics['spoilage'] > 10:
                farm_insights[farm] = f"High spoilage ({metrics['spoilage']:.1f}%)"
            elif metrics['spoilage'] == best_spoilage[1]['spoilage']:
                farm_insights[farm] = f"Low spoilage ({metrics['spoilage']:.1f}%)"
            else:
                farm_insights[farm] = f"Spoilage: {metrics['spoilage']:.1f}%"
    
    # Processing insights
    elif section == 'processing':
        best_defects = min(farms_data.items(), key=lambda x: x[1]['defects'])
        worst_defects = max(farms_data.items(), key=lambda x: x[1]['defects'])
        best_uptime = max(farms_data.items(), key=lambda x: x[1]['machinery_uptime'])
        worst_uptime = min(farms_data.items(), key=lambda x: x[1]['machinery_uptime'])
        
        insights.append(f"✅ {best_defects[0]} has lowest defect rate ({best_defects[1]['defects']:.1f}%) - quality leader")
        if worst_defects[1]['defects'] > 8:
            insights.append(f"🔧 {worst_defects[0]} has high defects ({worst_defects[1]['defects']:.1f}%) - quality review needed")
            recommendations.append("Implement quality checkpoints and staff training")
            recommendations.append("Review processing procedures")
        elif worst_defects[1]['defects'] > 5:
            insights.append(f"⚠️ {worst_defects[0]} has elevated defects ({worst_defects[1]['defects']:.1f}%) - monitor closely")
            recommendations.append("Improve quality control processes")
        else:
            insights.append(f"📊 {worst_defects[0]} has defects at {worst_defects[1]['defects']:.1f}% - acceptable")
        
        if worst_uptime[1]['machinery_uptime'] < 85:
            insights.append(f"⚙️ {worst_uptime[0]} needs machinery maintenance ({worst_uptime[1]['machinery_uptime']:.0f}% uptime)")
            recommendations.append("Schedule preventive equipment maintenance")
            recommendations.append("Consider upgrading aging machinery")
        if best_uptime[1]['machinery_uptime'] > 95:
            insights.append(f"✅ {best_uptime[0]} has excellent machinery performance ({best_uptime[1]['machinery_uptime']:.0f}%)")
        
        # Defect gap analysis
        defect_gap = worst_defects[1]['defects'] - best_defects[1]['defects']
        if defect_gap > 3:
            insights.append(f"📈 Defect rate gap of {defect_gap:.1f}% - standardization opportunity")
            recommendations.append("Share quality control best practices")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['defects'] == worst_defects[1]['defects'] and metrics['defects'] > 5:
                farm_insights[farm] = f"High defects ({metrics['defects']:.1f}%)"
            elif metrics['machinery_uptime'] == best_uptime[1]['machinery_uptime'] and metrics['machinery_uptime'] > 95:
                farm_insights[farm] = f"Excellent machinery ({metrics['machinery_uptime']:.0f}%)"
            else:
                farm_insights[farm] = f"Defects: {metrics['defects']:.1f}%"
    
    # Transportation insights
    elif section == 'transportation':
        best_delays = min(farms_data.items(), key=lambda x: x[1]['delays'])
        worst_delays = max(farms_data.items(), key=lambda x: x[1]['delays'])
        # Calculate fuel usage from data if available
        fuel_data = {}
        for farm_name, data in all_farms.items():
            fuel_data[farm_name] = float(data['FuelUsage_L_per_100km'].mean())
        best_fuel = min(fuel_data.items(), key=lambda x: x[1]) if fuel_data else None
        worst_fuel = max(fuel_data.items(), key=lambda x: x[1]) if fuel_data else None
        
        insights.append(f"✅ {best_delays[0]} has best on-time delivery ({100-best_delays[1]['delays']:.0f}%) - logistics leader")
        if worst_delays[1]['delays'] > 15:
            insights.append(f"🚨 {worst_delays[0]} has high delays ({worst_delays[1]['delays']:.0f}%) - urgent optimization needed")
            recommendations.append("Optimize delivery routes and schedules")
            recommendations.append("Review carrier performance and contracts")
        elif worst_delays[1]['delays'] > 10:
            insights.append(f"⚠️ {worst_delays[0]} needs delay reduction ({worst_delays[1]['delays']:.0f}%)")
            recommendations.append("Optimize route planning and scheduling")
        else:
            insights.append(f"📊 {worst_delays[0]} has delays at {worst_delays[1]['delays']:.0f}% - acceptable")
        
        if best_fuel and worst_fuel:
            if best_fuel[1] < worst_fuel[1] * 0.8:
                insights.append(f"⛽ {best_fuel[0]} has best fuel efficiency ({best_fuel[1]:.0f}L/100km)")
            if worst_fuel[1] > 30:
                insights.append(f"⛽ {worst_fuel[0]} has high fuel consumption ({worst_fuel[1]:.0f}L/100km) - optimize routes")
                recommendations.append("Consider fuel-efficient vehicles and route optimization")
        
        # Delay gap analysis
        delay_gap = worst_delays[1]['delays'] - best_delays[1]['delays']
        if delay_gap > 5:
            insights.append(f"📈 Delivery delay gap of {delay_gap:.0f}% - logistics improvement opportunity")
            recommendations.append("Share best practices for logistics optimization")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['delays'] == worst_delays[1]['delays'] and metrics['delays'] > 10:
                farm_insights[farm] = f"High delays ({metrics['delays']:.0f}%)"
            elif metrics['delays'] == best_delays[1]['delays']:
                farm_insights[farm] = f"On-time ({100-metrics['delays']:.0f}%)"
            else:
                farm_insights[farm] = f"Delays: {metrics['delays']:.0f}%"
    
    # Retail insights
    elif section == 'retail':
        best_waste = min(farms_data.items(), key=lambda x: x[1]['waste'])
        worst_waste = max(farms_data.items(), key=lambda x: x[1]['waste'])
        
        insights.append(f"✅ {best_waste[0]} has lowest retail waste ({best_waste[1]['waste']:.1f}%)")
        if worst_waste[1]['waste'] > 12:
            insights.append(f"🗑️ {worst_waste[0]} has high waste ({worst_waste[1]['waste']:.1f}%)")
            recommendations.append("Implement dynamic pricing strategy")
        elif worst_waste[1]['waste'] > 10:
            insights.append(f"⚠️ {worst_waste[0]} needs waste reduction ({worst_waste[1]['waste']:.1f}%)")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['waste'] == worst_waste[1]['waste'] and metrics['waste'] > 10:
                farm_insights[farm] = f"High waste ({metrics['waste']:.1f}%)"
            elif metrics['waste'] == best_waste[1]['waste']:
                farm_insights[farm] = f"Low waste ({metrics['waste']:.1f}%)"
            else:
                farm_insights[farm] = f"Waste: {metrics['waste']:.1f}%"
    
    # Consumption insights
    elif section == 'consumption':
        best_satisfaction = max(farms_data.items(), key=lambda x: x[1]['satisfaction'])
        worst_satisfaction = min(farms_data.items(), key=lambda x: x[1]['satisfaction'])
        
        insights.append(f"⭐ {best_satisfaction[0]} has highest satisfaction ({best_satisfaction[1]['satisfaction']:.1f}/10)")
        if worst_satisfaction[1]['satisfaction'] < 7:
            insights.append(f"😞 {worst_satisfaction[0]} needs quality improvement ({worst_satisfaction[1]['satisfaction']:.1f}/10)")
            recommendations.append("Improve product quality and customer service")
        elif worst_satisfaction[1]['satisfaction'] < 8:
            insights.append(f"⚠️ {worst_satisfaction[0]} has moderate satisfaction ({worst_satisfaction[1]['satisfaction']:.1f}/10)")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['satisfaction'] == best_satisfaction[1]['satisfaction']:
                farm_insights[farm] = f"Top satisfaction ({metrics['satisfaction']:.1f}/10)"
            elif metrics['satisfaction'] == worst_satisfaction[1]['satisfaction'] and metrics['satisfaction'] < 7:
                farm_insights[farm] = f"Low satisfaction ({metrics['satisfaction']:.1f}/10)"
            else:
                farm_insights[farm] = f"Satisfaction: {metrics['satisfaction']:.1f}/10"
    
    # Waste insights
    elif section == 'waste':
        best_segregation = max(farms_data.items(), key=lambda x: x[1]['segregation'])
        worst_segregation = min(farms_data.items(), key=lambda x: x[1]['segregation'])
        best_upcycling = max(farms_data.items(), key=lambda x: x[1]['upcycling'])
        worst_upcycling = min(farms_data.items(), key=lambda x: x[1]['upcycling'])
        best_biogas = max(farms_data.items(), key=lambda x: x[1]['biogas'])
        worst_biogas = min(farms_data.items(), key=lambda x: x[1]['biogas'])
        
        # Show meaningful differences even if small
        seg_diff = best_segregation[1]['segregation'] - worst_segregation[1]['segregation']
        upcy_diff = best_upcycling[1]['upcycling'] - worst_upcycling[1]['upcycling']
        bio_diff = best_biogas[1]['biogas'] - worst_biogas[1]['biogas']
        
        insights.append(f"🏆 {best_segregation[0]} leads in waste segregation ({best_segregation[1]['segregation']:.1f}%) - best practice")
        if seg_diff > 0.5:
            insights.append(f"⚠️ {worst_segregation[0]} needs segregation improvement ({worst_segregation[1]['segregation']:.1f}%)")
            recommendations.append("Implement waste sorting training and clear labeling")
        else:
            insights.append(f"📊 All farms have similar segregation accuracy ({worst_segregation[1]['segregation']:.1f}%-{best_segregation[1]['segregation']:.1f}%)")
        
        if upcy_diff > 2:
            insights.append(f"🌱 {best_upcycling[0]} has best upcycling rate ({best_upcycling[1]['upcycling']:.1f}%) - circular economy leader")
            if worst_upcycling[1]['upcycling'] < 45:
                insights.append(f"🔄 {worst_upcycling[0]} needs upcycling boost ({worst_upcycling[1]['upcycling']:.1f}%) - opportunity identified")
                recommendations.append("Develop upcycling partnerships and processes")
                recommendations.append("Explore new upcycling markets")
            else:
                insights.append(f"📊 Upcycling rates range from {worst_upcycling[1]['upcycling']:.1f}% to {best_upcycling[1]['upcycling']:.1f}%")
        
        if bio_diff > 5:
            insights.append(f"⚡ {best_biogas[0]} generates most biogas ({best_biogas[1]['biogas']:.1f}m³) - energy efficiency leader")
            if worst_biogas[1]['biogas'] < 48:
                insights.append(f"⚠️ {worst_biogas[0]} can improve biogas output ({worst_biogas[1]['biogas']:.1f}m³)")
                recommendations.append("Optimize biogas production processes")
                recommendations.append("Review anaerobic digestion efficiency")
            else:
                insights.append(f"📊 Biogas output ranges from {worst_biogas[1]['biogas']:.1f}m³ to {best_biogas[1]['biogas']:.1f}m³")
        
        # Overall waste management performance
        if best_upcycling[1]['upcycling'] > 50 and best_biogas[1]['biogas'] > 50:
            insights.append(f"✅ {best_upcycling[0]} demonstrates excellent circular economy practices")
        
        # Farm-specific insights
        for farm, metrics in farms_data.items():
            if metrics['segregation'] == best_segregation[1]['segregation']:
                farm_insights[farm] = f"Best segregation ({metrics['segregation']:.1f}%)"
            elif metrics['upcycling'] == best_upcycling[1]['upcycling']:
                farm_insights[farm] = f"Best upcycling ({metrics['upcycling']:.1f}%)"
            elif metrics['biogas'] == best_biogas[1]['biogas']:
                farm_insights[farm] = f"Highest biogas ({metrics['biogas']:.1f}m³)"
            elif metrics['segregation'] == worst_segregation[1]['segregation'] and seg_diff > 0.5:
                farm_insights[farm] = f"Segregation: {metrics['segregation']:.1f}%"
            elif metrics['upcycling'] == worst_upcycling[1]['upcycling'] and upcy_diff > 2:
                farm_insights[farm] = f"Upcycling: {metrics['upcycling']:.1f}%"
            else:
                farm_insights[farm] = f"Seg: {metrics['segregation']:.0f}% | Upcy: {metrics['upcycling']:.0f}%"
    
    # Overview insights
    else:  # overview
        best_yield = max(farms_data.items(), key=lambda x: x[1]['yield'])
        worst_spoilage = max(farms_data.items(), key=lambda x: x[1]['spoilage'])
        best_satisfaction = max(farms_data.items(), key=lambda x: x[1]['satisfaction'])
        worst_performer = min(farms_data.items(), key=lambda x: (
            x[1]['yield'] / 10 - x[1]['spoilage'] / 20 + x[1]['satisfaction'] / 10
        ))
        
        insights.append(f"🏆 {best_yield[0]} leads in production ({best_yield[1]['yield']:.1f}t/ha)")
        if worst_spoilage[1]['spoilage'] > 15:
            insights.append(f"❌ {worst_spoilage[0]} needs attention - high spoilage ({worst_spoilage[1]['spoilage']:.1f}%)")
            recommendations.append("Improve storage and cold chain management")
        insights.append(f"⭐ {best_satisfaction[0]} has highest satisfaction ({best_satisfaction[1]['satisfaction']:.1f}/10)")
        
        # Farm-specific insights based on performance score
        for farm, metrics in farms_data.items():
            score = (metrics['yield'] / 10 * 20 + 
                    (1 - metrics['spoilage'] / 30) * 15 +
                    metrics['satisfaction'] / 10 * 15)
            if score > 70:
                farm_insights[farm] = "Excellent performance"
            elif score > 55:
                farm_insights[farm] = "Good performance"
            else:
                farm_insights[farm] = "Needs improvement"
    
    # Ensure all farms have insights
    for farm in farms_data.keys():
        if farm not in farm_insights or not farm_insights[farm]:
            farm_insights[farm] = "Performance within range"
    
    # Ensure we have 5-6 insights for better UI appearance
    if len(insights) < 5:
        # Add more comparison insights if needed
        if section == 'production':
            best_uptime = max(farms_data.items(), key=lambda x: x[1]['harvest_uptime'])
            insights.append(f"⚙️ {best_uptime[0]} has best harvest uptime ({best_uptime[1]['harvest_uptime']:.0f}%)")
        elif section == 'storage':
            best_temp = min(farms_data.items(), key=lambda x: abs(x[1]['storage_temp'] - 3.5))
            insights.append(f"🌡️ {best_temp[0]} has optimal storage temperature ({best_temp[1]['storage_temp']:.1f}°C)")
        elif section == 'processing':
            best_uptime = max(farms_data.items(), key=lambda x: x[1]['machinery_uptime'])
            insights.append(f"⚙️ {best_uptime[0]} has best machinery uptime ({best_uptime[1]['machinery_uptime']:.0f}%)")
    
    # Limit insights to top 5-6 most important
    if len(insights) > 6:
        # Prioritize critical insights
        critical = [i for i in insights if '❌' in i or '🚨' in i]
        warnings = [i for i in insights if '⚠️' in i]
        positives = [i for i in insights if '🏆' in i or '⭐' in i or '✅' in i]
        others = [i for i in insights if i not in critical and i not in warnings and i not in positives]
        insights = (critical[:2] + warnings[:2] + positives[:1] + others[:1])[:6]
    
    # Limit recommendations to top 3-4
    if len(recommendations) > 4:
        recommendations = recommendations[:4]
    
    return jsonify({
        'insights': insights, 
        'recommendations': recommendations,
        'type': 'comparison',
        'farm_insights': farm_insights
    })

def generate_farm_insights(farm_name, data, section):
    """Generate comprehensive insights for a specific farm and section - 5-6 insights with optimization suggestions"""
    insights = []
    recommendations = []
    
    if section == 'overview' or section == 'production':
        yield_avg = float(data['Yield_tonnes_per_ha'].mean())
        yield_std = float(data['Yield_tonnes_per_ha'].std())
        pest_risk = float(data['PestRiskScore'].mean())
        uptime = float(data['HarvestRobotUptime_%'].mean())
        fertilizer = float(data['Fertilizer_kg_per_ha'].mean())
        rainfall = float(data['Rainfall_mm'].mean())
        
        optimal_yield = 7.0
        critical_pest = 50
        optimal_uptime = 90
        
        # Yield insights
        if yield_avg < optimal_yield * 0.8:
            insights.append(f"⚠️ Low yield ({yield_avg:.1f}t/ha) - below optimal threshold")
            recommendations.append("Optimize soil nutrition and crop rotation")
            recommendations.append("Consider precision agriculture techniques")
        elif yield_avg > optimal_yield:
            insights.append(f"✅ Good yield ({yield_avg:.1f}t/ha) - above industry average")
        else:
            insights.append(f"📊 Yield at {yield_avg:.1f}t/ha - within acceptable range")
        
        # Yield variability
        if yield_std > yield_avg * 0.3:
            insights.append(f"📈 High yield variability ({yield_std:.1f}) - inconsistent practices detected")
            recommendations.append("Standardize farming procedures across fields")
        
        # Pest management
        if pest_risk > critical_pest:
            insights.append(f"🐛 High pest risk ({pest_risk:.0f}) - requires immediate attention")
            recommendations.append("Implement integrated pest management (IPM)")
            recommendations.append("Schedule regular field inspections")
        elif pest_risk < 20:
            insights.append(f"✅ Excellent pest control (risk: {pest_risk:.0f})")
        else:
            insights.append(f"🛡️ Pest risk at {pest_risk:.0f} - monitor regularly")
        
        # Equipment uptime
        if uptime < optimal_uptime:
            insights.append(f"⚙️ Low harvest uptime ({uptime:.0f}%) - maintenance needed")
            recommendations.append("Schedule preventive maintenance for harvest equipment")
        elif uptime > 95:
            insights.append(f"✅ Excellent machinery uptime ({uptime:.0f}%)")
        else:
            insights.append(f"⚙️ Harvest uptime at {uptime:.0f}% - can be improved")
        
        # Fertilizer optimization
        if fertilizer > 250:
            insights.append(f"🌾 High fertilizer usage ({fertilizer:.0f}kg/ha) - optimize for cost efficiency")
            recommendations.append("Conduct soil tests to optimize fertilizer application")
    
    if section == 'overview' or section == 'storage':
        spoilage = float(data['SpoilageRate_%'].mean())
        temp = float(data['StorageTemperature_C'].mean())
        humidity = float(data['Humidity_%'].mean())
        shelf_life = float(data['PredictedShelfLife_days'].mean())
        storage_days = float(data['StorageDays'].mean())
        critical_spoilage = 15
        optimal_temp_range = (2, 5)
        optimal_humidity = (75, 85)
        
        # Spoilage insights
        if spoilage > critical_spoilage:
            insights.append(f"❌ Critical spoilage ({spoilage:.1f}%) - immediate intervention required")
            recommendations.append("Review cold chain integrity and storage protocols")
            recommendations.append("Implement real-time monitoring systems")
        elif spoilage > 10:
            insights.append(f"⚠️ Elevated spoilage ({spoilage:.1f}%) - monitor closely")
            recommendations.append("Optimize storage conditions")
        elif spoilage < 5:
            insights.append(f"✅ Low spoilage ({spoilage:.1f}%) - excellent storage management")
        else:
            insights.append(f"📦 Spoilage at {spoilage:.1f}% - within acceptable limits")
        
        # Temperature control
        if temp < optimal_temp_range[0] or temp > optimal_temp_range[1]:
            insights.append(f"🌡️ Temperature out of range ({temp:.1f}°C) - optimal is 2-5°C")
            recommendations.append("Calibrate refrigeration systems")
            recommendations.append("Install temperature monitoring alarms")
        else:
            insights.append(f"✅ Optimal storage temperature ({temp:.1f}°C)")
        
        # Humidity control
        if humidity < optimal_humidity[0] or humidity > optimal_humidity[1]:
            insights.append(f"💧 Humidity at {humidity:.0f}% - optimal range is 75-85%")
            recommendations.append("Adjust humidity control systems")
        else:
            insights.append(f"✅ Optimal humidity levels ({humidity:.0f}%)")
        
        # Shelf life optimization
        if shelf_life < 10:
            insights.append(f"⏱️ Short shelf life ({shelf_life:.1f} days) - quality preservation needs improvement")
            recommendations.append("Review harvest timing and handling practices")
        else:
            insights.append(f"✅ Good shelf life ({shelf_life:.1f} days)")
    
    if section == 'overview' or section == 'processing':
        defect_rate = float(data['DefectRate_%'].mean())
        machinery_uptime = float(data['MachineryUptime_%'].mean())
        packaging_speed = float(data['PackagingSpeed_units_per_min'].mean())
        critical_defects = 5
        optimal_uptime = 95
        
        # Defect rate
        if defect_rate > critical_defects:
            insights.append(f"🔧 High defect rate ({defect_rate:.1f}%) - quality control review needed")
            recommendations.append("Implement quality checkpoints and staff training")
            recommendations.append("Review processing procedures")
        elif defect_rate < 2:
            insights.append(f"✅ Excellent quality control ({defect_rate:.1f}% defects)")
        else:
            insights.append(f"📊 Defect rate at {defect_rate:.1f}% - within acceptable range")
        
        # Machinery uptime
        if machinery_uptime < optimal_uptime:
            downtime = 100 - machinery_uptime
            insights.append(f"⚙️ High machinery downtime ({downtime:.0f}%) - affecting productivity")
            recommendations.append("Schedule equipment maintenance")
            recommendations.append("Consider upgrading aging machinery")
        elif machinery_uptime > 98:
            insights.append(f"✅ Excellent machinery performance ({machinery_uptime:.0f}% uptime)")
        else:
            insights.append(f"⚙️ Machinery uptime at {machinery_uptime:.0f}% - good performance")
        
        # Processing efficiency
        if packaging_speed < 300:
            insights.append(f"📦 Low packaging speed ({packaging_speed:.0f} units/min) - efficiency can be improved")
            recommendations.append("Optimize packaging workflows")
        else:
            insights.append(f"✅ Good packaging efficiency ({packaging_speed:.0f} units/min)")
        
        # Process type analysis
        defect_by_process = data.groupby('ProcessType')['DefectRate_%'].mean()
        if len(defect_by_process) > 0:
            worst_process = defect_by_process.idxmax()
            if defect_by_process.max() > critical_defects:
                insights.append(f"🔍 {worst_process} process has highest defects ({defect_by_process.max():.1f}%)")
                recommendations.append(f"Review and optimize {worst_process} procedures")
    
    if section == 'overview' or section == 'transportation':
        delays = float((data['DeliveryDelayFlag'].sum() / len(data) * 100))
        fuel_usage = float(data['FuelUsage_L_per_100km'].mean())
        transit_spoilage = float(data['SpoilageInTransit_%'].mean())
        distance = float(data['TransportDistance_km'].mean())
        critical_delays = 15
        optimal_fuel = 25
        
        # Delivery delays
        if delays > critical_delays:
            insights.append(f"🚨 High delay rate ({delays:.0f}%) - logistics optimization urgent")
            recommendations.append("Optimize delivery routes and schedules")
            recommendations.append("Review carrier performance and contracts")
        elif delays > 10:
            insights.append(f"⚠️ Moderate delays ({delays:.0f}%) - monitor transportation efficiency")
            recommendations.append("Optimize route planning")
        elif delays < 5:
            insights.append(f"✅ Excellent on-time delivery ({100-delays:.0f}%)")
        else:
            insights.append(f"🚚 Delivery delays at {delays:.0f}% - within acceptable range")
        
        # Fuel efficiency
        if fuel_usage > optimal_fuel * 1.2:
            insights.append(f"⛽ High fuel consumption ({fuel_usage:.0f}L/100km) - route optimization needed")
            recommendations.append("Optimize delivery routes")
            recommendations.append("Consider fuel-efficient vehicles")
        else:
            insights.append(f"✅ Fuel efficiency at {fuel_usage:.0f}L/100km - good performance")
        
        # Transit spoilage
        if transit_spoilage > 5:
            insights.append(f"📦 Transit spoilage ({transit_spoilage:.1f}%) - handling issues detected")
            recommendations.append("Improve packaging and handling during transportation")
        else:
            insights.append(f"✅ Low transit spoilage ({transit_spoilage:.1f}%)")
        
        # Distance optimization
        if distance > 1000:
            insights.append(f"📍 Long average distance ({distance:.0f}km) - consider distribution centers")
            recommendations.append("Evaluate distribution center locations")
    
    if section == 'overview' or section == 'retail':
        waste = float(data['WastePercentage_%'].mean())
        inventory = float(data['RetailInventory_units'].sum())
        sales_velocity = float(data['SalesVelocity_units_per_day'].mean())
        pricing_index = float(data['DynamicPricingIndex'].mean())
        critical_waste = 12
        
        # Waste management
        if waste > critical_waste:
            insights.append(f"🗑️ High retail waste ({waste:.1f}%) - pricing strategy review needed")
            recommendations.append("Implement dynamic pricing and markdown strategies")
            recommendations.append("Optimize inventory levels")
        elif waste < 5:
            insights.append(f"✅ Low waste ({waste:.1f}%) - excellent waste management")
        else:
            insights.append(f"📊 Waste at {waste:.1f}% - within acceptable range")
        
        # Inventory optimization
        if sales_velocity > 0:
            days_of_inventory = inventory / (sales_velocity * len(data)) if len(data) > 0 else 0
            if days_of_inventory > 30:
                insights.append(f"📦 High inventory levels ({days_of_inventory:.0f} days) - risk of overstocking")
                recommendations.append("Adjust procurement to match sales velocity")
            elif days_of_inventory < 7:
                insights.append(f"⚡ Low inventory ({days_of_inventory:.0f} days) - risk of stockouts")
                recommendations.append("Increase safety stock levels")
            else:
                insights.append(f"✅ Optimal inventory levels ({days_of_inventory:.0f} days)")
        
        # Pricing optimization
        if pricing_index < 0.9:
            insights.append(f"💰 Low pricing index ({pricing_index:.2f}) - revenue optimization opportunity")
            recommendations.append("Implement dynamic pricing strategies")
        else:
            insights.append(f"✅ Good pricing strategy (index: {pricing_index:.2f})")
    
    if section == 'overview' or section == 'consumption':
        satisfaction = float(data['SatisfactionScore_0_10'].mean())
        household_waste = float(data['HouseholdWaste_kg'].mean())
        recipe_accuracy = float(data['RecipeRecommendationAccuracy_%'].mean())
        
        # Customer satisfaction
        if satisfaction < 7:
            insights.append(f"😞 Low customer satisfaction ({satisfaction:.1f}/10) - quality improvement needed")
            recommendations.append("Gather customer feedback and improve product quality")
            recommendations.append("Enhance customer service")
        elif satisfaction > 8:
            insights.append(f"⭐ High customer satisfaction ({satisfaction:.1f}/10) - excellent performance")
        else:
            insights.append(f"📊 Customer satisfaction at {satisfaction:.1f}/10 - good performance")
        
        # Household waste
        if household_waste > 3:
            insights.append(f"🍽️ High household waste ({household_waste:.2f}kg) - consumer education needed")
            recommendations.append("Promote meal planning and portion control education")
        else:
            insights.append(f"✅ Low household waste ({household_waste:.2f}kg)")
        
        # Recipe recommendations
        if recipe_accuracy < 80:
            insights.append(f"📱 Recipe accuracy at {recipe_accuracy:.1f}% - improvement needed")
            recommendations.append("Improve recipe recommendation algorithms")
        else:
            insights.append(f"✅ Good recipe accuracy ({recipe_accuracy:.1f}%)")
    
    if section == 'overview' or section == 'waste':
        segregation = float(data['SegregationAccuracy_%'].mean())
        upcycling = float(data['UpcyclingRate_%'].mean())
        biogas = float(data['BiogasOutput_m3'].mean())
        
        # Waste segregation
        if segregation < 85:
            insights.append(f"♻️ Low segregation accuracy ({segregation:.0f}%) - training needed")
            recommendations.append("Implement waste sorting training and clear labeling")
        elif segregation > 90:
            insights.append(f"✅ Excellent waste segregation ({segregation:.0f}%)")
        else:
            insights.append(f"📊 Segregation at {segregation:.0f}% - good performance")
        
        # Upcycling opportunities
        if upcycling < 45:
            insights.append(f"🔄 Low upcycling rate ({upcycling:.0f}%) - circular economy opportunity")
            recommendations.append("Develop upcycling partnerships and processes")
            recommendations.append("Explore new upcycling markets")
        elif upcycling > 70:
            insights.append(f"🌱 Excellent upcycling performance ({upcycling:.0f}%)")
        else:
            insights.append(f"♻️ Upcycling at {upcycling:.0f}% - can be improved")
        
        # Biogas production
        if biogas < 48:
            insights.append(f"⚡ Low biogas output ({biogas:.0f}m³) - optimization opportunity")
            recommendations.append("Optimize biogas production processes")
            recommendations.append("Review anaerobic digestion efficiency")
        elif biogas > 55:
            insights.append(f"✅ High biogas production ({biogas:.0f}m³)")
        else:
            insights.append(f"📊 Biogas output at {biogas:.0f}m³ - within range")
    
    # Ensure we have 5-6 insights for better UI appearance
    if len(insights) < 5:
        # Add some general insights if we don't have enough
        if section == 'overview':
            score = calculate_performance_score_from_data(data)
            if score > 75:
                insights.append(f"🏆 Overall performance score: {score:.0f}/100 - excellent")
            elif score > 60:
                insights.append(f"📊 Overall performance score: {score:.0f}/100 - good")
            else:
                insights.append(f"⚠️ Overall performance score: {score:.0f}/100 - needs improvement")
    
    # Limit to 6 most important insights
    if len(insights) > 6:
        critical = [i for i in insights if '❌' in i or '🚨' in i]
        warnings = [i for i in insights if '⚠️' in i]
        positives = [i for i in insights if '✅' in i or '🏆' in i or '⭐' in i]
        others = [i for i in insights if i not in critical and i not in warnings and i not in positives]
        insights = (critical[:2] + warnings[:2] + positives[:1] + others[:1])[:6]
    
    # Limit recommendations to 4-5 most actionable
    if len(recommendations) > 5:
        recommendations = recommendations[:5]
    
    # If no insights, provide status
    if not insights:
        insights.append("✅ All metrics optimal")
        insights.append("📊 Continue monitoring key performance indicators")
    
    # Generate concise one-line summary for card display
    concise_insight = ""
    if insights:
        critical_insights = [i for i in insights if '❌' in i or '⚠️' in i or '🚨' in i]
        if critical_insights:
            concise_insight = critical_insights[0].replace('❌ ', '').replace('⚠️ ', '').replace('🚨 ', '')
        else:
            concise_insight = insights[0].replace('✅ ', '').replace('🏆 ', '').replace('⭐ ', '')
        if len(concise_insight) > 50:
            concise_insight = concise_insight[:47] + '...'
    
    return jsonify({
        'insights': insights,
        'recommendations': recommendations,
        'type': 'farm_specific',
        'farm': farm_name,
        'section': section,
        'concise_insight': concise_insight
    })

@app.route('/details/all/<stage>')
def view_comparison_details(stage):
    """View comparison details for all farms for a specific stage"""
    all_farms = load_all_farms_data()
    
    stage_fields = {
        'production': ['BatchID', 'CropType', 'FarmLocation', 'SoilMoisture_%', 'Temperature_C', 'Rainfall_mm', 'Fertilizer_kg_per_ha', 'Yield_tonnes_per_ha', 'PestRiskScore', 'HarvestRobotUptime_%'],
        'storage': ['BatchID', 'CropType', 'GradingScore', 'StorageTemperature_C', 'Humidity_%', 'SpoilageRate_%', 'PredictedShelfLife_days', 'StorageDays'],
        'processing': ['BatchID', 'ProcessType', 'PackagingType', 'PackagingSpeed_units_per_min', 'DefectRate_%', 'MachineryUptime_%'],
        'transportation': ['BatchID', 'TransportMode', 'TransportDistance_km', 'FuelUsage_L_per_100km', 'DeliveryTime_hr', 'DeliveryDelayFlag', 'SpoilageInTransit_%'],
        'retail': ['BatchID', 'CropType', 'RetailInventory_units', 'SalesVelocity_units_per_day', 'DynamicPricingIndex', 'WastePercentage_%'],
        'consumption': ['BatchID', 'HouseholdWaste_kg', 'RecipeRecommendationAccuracy_%', 'SatisfactionScore_0_10'],
        'waste': ['BatchID', 'WasteType', 'SegregationAccuracy_%', 'UpcyclingRate_%', 'BiogasOutput_m3'],
        'overview': ['BatchID', 'CropType', 'Yield_tonnes_per_ha', 'SpoilageRate_%', 'DefectRate_%', 'WastePercentage_%']
    }

    fields = stage_fields.get(stage, [])
    if not fields:
        return "Stage not found", 404

    # Aggregate data by farm and create a pivoted table with farms as columns
    farm_stats = {}
    for farm_name, data in all_farms.items():
        farm_data = data[[f for f in fields if f in data.columns]].copy()
        # Calculate mean for numeric columns, mode for categorical
        stats = {}
        for col in farm_data.columns:
            if farm_data[col].dtype in ['int64', 'float64']:
                stats[col] = farm_data[col].mean()
            else:
                # For categorical, get the most common value
                stats[col] = farm_data[col].mode()[0] if len(farm_data[col].mode()) > 0 else farm_data[col].iloc[0]
        farm_stats[farm_name] = stats
    
    # Create a DataFrame with metrics as rows and farms as columns
    metrics_df = pd.DataFrame.from_dict(farm_stats, orient='index').T
    # Transpose so farms are columns and metrics are rows
    comparison_df = metrics_df.T
    comparison_df.index.name = 'Metric'
    comparison_df = comparison_df.reset_index()

    stage_titles = {
        'production': 'Production Stage',
        'storage': 'Storage & Post-Harvest',
        'processing': 'Processing & Packaging',
        'transportation': 'Distribution & Transportation',
        'retail': 'Retail Insights',
        'consumption': 'Consumption & Household',
        'waste': 'Waste Management',
        'overview': 'Dashboard Overview'
    }

    stage_title = stage_titles.get(stage, stage.title())
    
    # Format numeric values
    for col in comparison_df.columns:
        if col != 'Metric':
            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
    
    table_html = comparison_df.to_html(classes='details-table', index=False, escape=False)
    total_records = len(comparison_df)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Farms - {stage_title} Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); padding: 30px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid #1a5f7a; padding-bottom: 15px; }}
        .header h1 {{ color: #1a5f7a; font-size: 28px; }}
        .back-btn {{ background: #1a5f7a; color: white; padding: 10px 20px; border: none; border-radius: 12px; cursor: pointer; font-size: 14px; transition: background 0.3s; text-decoration: none; display: inline-block; }}
        .back-btn:hover {{ background: #0f3a4f; }}
        .record-info {{ background: #ecf0f1; padding: 15px; border-radius: 12px; margin-bottom: 20px; }}
        .table-wrapper {{ overflow-x: auto; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .details-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .details-table thead {{ background-color: #1a5f7a; color: white; }}
        .details-table th {{ padding: 12px; text-align: left; font-weight: 600; position: sticky; top: 0; }}
        .details-table td {{ padding: 10px 12px; border-bottom: 1px solid #ddd; text-align: left; }}
        .details-table tbody tr:hover {{ background-color: #f5f5f5; }}
        .details-table tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .details-table td:first-child {{ font-weight: 600; color: #1a5f7a; background-color: #f0f8ff; }}
        .details-table th:first-child {{ background-color: #0f3a4f; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>All Farms - {stage_title} Comparison</h1>
            </div>
            <a href="/" class="back-btn">← Back to Dashboard</a>
        </div>

        <div class="record-info">
            <strong>Total Metrics:</strong> {total_records} | 
            <strong>View:</strong> Farm Comparison (Farms as Columns)
        </div>

        <div class="table-wrapper">
            {table_html}
        </div>
    </div>
</body>
</html>"""

    return html_content

@app.route('/details/<farm_name>/<stage>')
def view_details(farm_name, stage):
    """View detailed records for each stage with pagination"""
    data = load_farm_data(farm_name)
    if data.empty:
        return f"Farm {farm_name} not found", 404
    
    stage_fields = {
        'production': ['BatchID', 'CropType', 'FarmLocation', 'SoilMoisture_%', 'Temperature_C', 'Rainfall_mm', 'Fertilizer_kg_per_ha', 'Yield_tonnes_per_ha', 'PestRiskScore', 'HarvestRobotUptime_%'],
        'storage': ['BatchID', 'CropType', 'GradingScore', 'StorageTemperature_C', 'Humidity_%', 'SpoilageRate_%', 'PredictedShelfLife_days', 'StorageDays'],
        'processing': ['BatchID', 'ProcessType', 'PackagingType', 'PackagingSpeed_units_per_min', 'DefectRate_%', 'MachineryUptime_%'],
        'transportation': ['BatchID', 'TransportMode', 'TransportDistance_km', 'FuelUsage_L_per_100km', 'DeliveryTime_hr', 'DeliveryDelayFlag', 'SpoilageInTransit_%'],
        'retail': ['BatchID', 'CropType', 'RetailInventory_units', 'SalesVelocity_units_per_day', 'DynamicPricingIndex', 'WastePercentage_%'],
        'consumption': ['BatchID', 'HouseholdWaste_kg', 'RecipeRecommendationAccuracy_%', 'SatisfactionScore_0_10'],
        'waste': ['BatchID', 'WasteType', 'SegregationAccuracy_%', 'UpcyclingRate_%', 'BiogasOutput_m3'],
        'overview': ['BatchID', 'CropType', 'Yield_tonnes_per_ha', 'SpoilageRate_%', 'DefectRate_%', 'WastePercentage_%']
    }

    fields = stage_fields.get(stage, [])
    if not fields:
        return "Stage not found", 404

    stage_data = data[[f for f in fields if f in data.columns]].copy()

    page = request.args.get('page', 1, type=int)
    per_page = 50
    total_records = len(stage_data)
    total_pages = (total_records + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_data = stage_data.iloc[start_idx:end_idx]

    stage_titles = {
        'production': 'Production Stage',
        'storage': 'Storage & Post-Harvest',
        'processing': 'Processing & Packaging',
        'transportation': 'Distribution & Transportation',
        'retail': 'Retail Insights',
        'consumption': 'Consumption & Household',
        'waste': 'Waste Management',
        'overview': 'Dashboard Overview'
    }

    stage_title = stage_titles.get(stage, stage.title())
    table_html = paginated_data.to_html(classes='details-table', index=False)

    pagination_html = ''
    if page > 1:
        pagination_html += f'<a href="?page=1">« First</a> <a href="?page={page-1}">‹ Previous</a> '
    for p in range(max(1, page-2), min(total_pages+1, page+3)):
        if p == page:
            pagination_html += f'<span class="current">{p}</span> '
        else:
            pagination_html += f'<a href="?page={p}">{p}</a> '
    if page < total_pages:
        pagination_html += f'<a href="?page={page+1}">Next ›</a> <a href="?page={total_pages}">Last »</a>'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{farm_name} - {stage_title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); padding: 30px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid #1a5f7a; padding-bottom: 15px; }}
        .header h1 {{ color: #1a5f7a; font-size: 28px; }}
        .back-btn {{ background: #1a5f7a; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; transition: background 0.3s; text-decoration: none; display: inline-block; }}
        .back-btn:hover {{ background: #0f3a4f; }}
        .record-info {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .table-wrapper {{ overflow-x: auto; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .details-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .details-table thead {{ background-color: #1a5f7a; color: white; }}
        .details-table th {{ padding: 12px; text-align: left; font-weight: 600; }}
        .details-table td {{ padding: 10px 12px; border-bottom: 1px solid #ddd; }}
        .details-table tbody tr:hover {{ background-color: #f5f5f5; }}
        .details-table tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .pagination {{ display: flex; justify-content: center; gap: 10px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; }}
        .pagination a, .pagination span {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; text-decoration: none; color: #1a5f7a; transition: all 0.2s; }}
        .pagination a:hover {{ background: #1a5f7a; color: white; }}
        .pagination .current {{ background: #1a5f7a; color: white; border: 1px solid #1a5f7a; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>{farm_name} - {stage_title}</h1>
            </div>
            <a href="/" class="back-btn">← Back to Dashboard</a>
        </div>

        <div class="record-info">
            <strong>Total Records:</strong> {total_records} | 
            <strong>Showing:</strong> {start_idx + 1}-{min(end_idx, total_records)} | 
            <strong>Page:</strong> {page} of {total_pages}
        </div>

        <div class="table-wrapper">
            {table_html}
        </div>

        <div class="pagination">
            {pagination_html}
        </div>
    </div>
</body>
</html>"""

    return html_content

def convert_markdown_to_html(text):
    """Convert markdown formatting to HTML for better display"""
    import re
    
    # First, convert **bold** to <strong>bold</strong> (handle nested cases)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    
    # Convert * list items to <li> items
    lines = text.split('\n')
    html_lines = []
    in_list = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if line starts with * (list item) - but not ** (bold)
        list_match = re.match(r'^\s*\*\s+(.+)$', line)
        if list_match and not line_stripped.startswith('**'):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            # Process the list item content (may contain <strong> tags)
            item_content = list_match.group(1)
            html_lines.append(f'<li>{item_content}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            if line_stripped:
                # Regular paragraph - may contain <strong> tags
                html_lines.append(f'<p>{line}</p>')
            else:
                html_lines.append('<br>')
    
    if in_list:
        html_lines.append('</ul>')
    
    result = '\n'.join(html_lines)
    return result

def is_off_topic(question):
    """Detect if a question is off-topic (not related to farming or project data)"""
    question_lower = question.lower().strip()
    
    # Farming and project-related keywords
    farming_keywords = [
        'farm', 'crop', 'yield', 'harvest', 'soil', 'fertilizer', 'irrigation',
        'livestock', 'cattle', 'chicken', 'pig', 'sheep', 'goat', 'dairy',
        'spoilage', 'waste', 'storage', 'transport', 'delivery', 'processing',
        'retail', 'consumption', 'satisfaction', 'defect', 'pest', 'machinery',
        'uptime', 'delay', 'temperature', 'humidity', 'shelf', 'life',
        'performance', 'score', 'metric', 'data', 'farma', 'farmb', 'farmc',
        'farmd', 'tomato', 'potato', 'wheat', 'corn', 'rice', 'vegetable',
        'fruit', 'grain', 'production', 'supply', 'chain', 'comparison',
        'compare', 'best', 'worst', 'ranking', 'rank', 'top', 'bottom',
        'biogas', 'upcycling', 'segregation', 'waste management', 'packaging'
    ]
    
    # Off-topic keywords (common non-farming topics)
    off_topic_keywords = [
        'weather forecast', 'recipe', 'cooking', 'restaurant', 'movie',
        'music', 'sports', 'politics', 'news', 'stock market', 'cryptocurrency',
        'bitcoin', 'programming', 'code', 'python', 'javascript', 'html',
        'css', 'website', 'app development', 'game', 'video game', 'tv show',
        'celebrity', 'fashion', 'travel', 'vacation', 'hotel', 'flight',
        'mathematics', 'physics', 'chemistry', 'biology', 'history', 'geography',
        'philosophy', 'religion', 'medical', 'health advice', 'diagnosis',
        'legal advice', 'lawyer', 'court', 'investment', 'trading', 'finance',
        'shopping', 'product review', 'amazon', 'netflix', 'youtube'
    ]
    
    # Check if question contains farming keywords
    has_farming_keyword = any(keyword in question_lower for keyword in farming_keywords)
    
    # Check if question is clearly off-topic
    is_clearly_off_topic = any(keyword in question_lower for keyword in off_topic_keywords)
    
    # If it has farming keywords, it's on-topic
    if has_farming_keyword:
        return False
    
    # If it's clearly off-topic, refuse it
    if is_clearly_off_topic:
        return True
    
    # For ambiguous questions (greetings, general questions), allow them
    # but they'll be handled by the prompt to redirect to farming topics
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening))[\s!?.,]*$',
        r'^(what|who|where|when|why|how)\s+(are|is|can|do|does|will|would)',
        r'^(tell\s+me|explain|describe|what\s+is|what\s+are)',
    ]
    
    for pattern in greeting_patterns:
        if re.match(pattern, question_lower):
            return False  # Allow greetings and general questions
    
    # If question is very short and doesn't match anything, consider it potentially off-topic
    if len(question.split()) < 3 and not has_farming_keyword:
        return False  # Let the model handle it, but prompt will guide to farming topics
    
    # Default: if no farming keywords and not clearly off-topic, let the model decide
    # but we'll add strict instructions in the prompt
    return False

def create_farmer_prompt(context, question, language='en'):
    """Create a prompt with expert farmer persona and strict farming-only restrictions"""
    
    # Language instructions
    language_instructions = {
        'en': 'Respond in English. Use natural, conversational English.',
        'hi': 'Respond in Hindi (हिंदी). Use natural, conversational Hindi. Write all text in Devanagari script.',
        'kn': 'Respond in Kannada (ಕನ್ನಡ). Use natural, conversational Kannada. Write all text in Kannada script.'
    }
    
    lang_instruction = language_instructions.get(language, language_instructions['en'])
    
    prompt = f"""You are a professional, experienced, and knowledgeable agricultural expert with decades of hands-on experience in farming and food supply chain management. You communicate in a polite, professional, and courteous manner while remaining approachable and helpful. You maintain a respectful tone throughout all interactions.

TONE AND STYLE REQUIREMENTS:
- Use professional and polite language at all times
- Be courteous, respectful, and well-mannered
- Avoid casual expressions, slang, contractions like "keepin'", or overly informal phrases like "Just holler!"
- Use proper grammar and complete sentences
- Be warm and helpful, but maintain professionalism
- Use phrases like "I'd be happy to help", "Please let me know", "I can provide", "Would you like to know more about"
- End responses politely with offers to help further, but avoid overly casual closings

LANGUAGE INSTRUCTION - CRITICAL:
{lang_instruction}
You MUST respond entirely in the requested language. Maintain the same accuracy, professionalism, and expertise regardless of the language.

CRITICAL RESTRICTIONS - YOU MUST FOLLOW THESE STRICTLY:
1. You ONLY answer questions about:
   - Farming, agriculture, crops, livestock, and farm operations
   - The food supply chain data from this project (FarmA, FarmB, FarmC, FarmD)
   - Farm metrics: yields, spoilage, waste, defects, satisfaction, pest risk, machinery uptime, etc.
   - Crop types, storage, processing, transportation, retail, consumption, and waste management
   - Farm comparisons, performance scores, and data analysis

2. You MUST politely refuse ALL questions that are NOT related to farming or this project's data, such as:
   - General knowledge questions (history, science, geography, etc.)
   - Technology, programming, or software questions
   - Entertainment, movies, music, sports
   - Cooking recipes or restaurant recommendations
   - Weather forecasts (unless specifically about farm weather impact)
   - Medical, legal, or financial advice
   - Any topic unrelated to agriculture or this food supply chain project

3. When refusing off-topic questions, respond professionally: "I appreciate your question, but I specialize exclusively in farming, agriculture, and the data from our farms. I'd be happy to assist you with any questions related to our food supply chain, yields, spoilage, waste management, or farm performance."

4. When answering farming questions, use the data provided below. Be accurate with numbers and specific with your answers. Present information clearly and professionally.

5. Keep your responses clear, professional, and concise while remaining helpful and approachable.

FARM DATA FROM THIS PROJECT:
{context}

USER'S QUESTION: {question}

Remember: You are a professional agricultural expert. You only discuss farming and this project's farm data. Maintain a polite, professional, and respectful tone in all responses. Be helpful and informative while staying strictly within your agricultural expertise."""
    
    return prompt

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """AI Chatbot endpoint that answers questions about farms and metrics using Gemini API"""
    try:
        data = request.json
        question = data.get('message', '').strip()
        language = data.get('language', 'en').lower()  # Default to English
        
        # Validate language
        valid_languages = ['en', 'hi', 'kn']
        if language not in valid_languages:
            language = 'en'
        
        # Get language-specific greetings
        greetings = {
            'en': 'Howdy! I\'m here to help you with questions about farming and our farm data. What would you like to know?',
            'hi': 'नमस्ते! मैं खेती और हमारे फार्म डेटा के बारे में प्रश्नों में आपकी मदद करने के लिए यहाँ हूँ। आप क्या जानना चाहेंगे?',
            'kn': 'ನಮಸ್ಕಾರ! ನಾನು ಕೃಷಿ ಮತ್ತು ನಮ್ಮ ಫಾರ್ಮ್ ಡೇಟಾದ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಸಹಾಯ ಮಾಡಲು ಇಲ್ಲಿದ್ದೇನೆ. ನೀವು ಏನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?'
        }
        
        refusal_messages = {
            'en': 'Well, I appreciate your question, but I\'m a farmer through and through - I only talk about farming, crops, livestock, and the data from our farms here. I\'d be happy to help you with anything related to our food supply chain, yields, spoilage, waste management, or farm performance though!',
            'hi': 'अच्छा, मैं आपके प्रश्न की सराहना करता हूँ, लेकिन मैं पूरी तरह से एक किसान हूँ - मैं केवल खेती, फसलों, पशुधन, और हमारे फार्मों के डेटा के बारे में बात करता हूँ। हालाँकि, मैं हमारे खाद्य आपूर्ति श्रृंखला, उपज, खराबी, अपशिष्ट प्रबंधन, या फार्म प्रदर्शन से संबंधित किसी भी चीज़ में आपकी मदद करने में खुशी होगी!',
            'kn': 'ಸರಿ, ನಾನು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಮೆಚ್ಚುತ್ತೇನೆ, ಆದರೆ ನಾನು ಸಂಪೂರ್ಣವಾಗಿ ರೈತನಾಗಿದ್ದೇನೆ - ನಾನು ಕೇವಲ ಕೃಷಿ, ಬೆಳೆಗಳು, ಪಶುಸಂಪತ್ತು ಮತ್ತು ನಮ್ಮ ಫಾರ್ಮ್ಗಳ ಡೇಟಾದ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತೇನೆ. ಆದಾಗ್ಯೂ, ನಮ್ಮ ಆಹಾರ ಸರಬರಾಜು ಸರಪಳಿ, ಇಳುವರಿ, ಕೆಡುವಿಕೆ, ತ್ಯಾಜ್ಯ ನಿರ್ವಹಣೆ, ಅಥವಾ ಫಾರ್ಮ್ ಕಾರ್ಯಕ್ಷಮತೆಗೆ ಸಂಬಂಧಿಸಿದ ಯಾವುದೇ ವಿಷಯದಲ್ಲಿ ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ನನಗೆ ಸಂತೋಷವಾಗುತ್ತದೆ!'
        }
        
        if not question:
            return jsonify({'response': greetings.get(language, greetings['en'])})
        
        # Check if question is off-topic
        if is_off_topic(question):
            return jsonify({
                'response': refusal_messages.get(language, refusal_messages['en'])
            })
        
        # Load all farm data for context
        all_farms = load_all_farms_data()
        
        # Prepare context with farm data
        context = prepare_farm_context(all_farms)
        
        # Create system prompt with farmer persona and language
        system_prompt = create_farmer_prompt(context, question, language)
        
        # Language-based AI selection:
        # - Hindi/Kannada: Gemini first (better multilingual support)
        # - English: Ollama first (local, faster)
        use_gemini_first = language in ['hi', 'kn']
        
        response_text = None
        
        # Helper function to try Ollama
        def try_ollama():
            try:
                ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
                ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
                
                ollama_payload = {
                    'model': ollama_model,
                    'prompt': system_prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 1000
                    }
                }
                
                response = requests.post(
                    f'{ollama_url}/api/generate',
                    json=ollama_payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    if response_text:
                        app.logger.info(f"Successfully used Ollama ({ollama_model}) for chatbot response")
                        return convert_markdown_to_html(response_text)
                return None
            except Exception as e:
                app.logger.warning(f"Ollama error: {e}")
                return None
        
        # Helper function to try Gemini
        def try_gemini():
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key or api_key.strip() == 'your_api_key_here' or api_key.strip() == '':
                return None
            
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                response = model.generate_content(
                    system_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1000,
                    )
                )
                
                response_text = response.text.strip() if response.text else ''
                if response_text:
                    app.logger.info("Successfully used Gemini API for chatbot response")
                    return convert_markdown_to_html(response_text)
                return None
            except Exception as e:
                app.logger.warning(f"Gemini error: {e}")
                return None
        
        # Try AI based on language preference
        if use_gemini_first:
            # Hindi/Kannada: Try Gemini first, then Ollama
            app.logger.info(f"Language {language} detected - trying Gemini first")
            response_text = try_gemini()
            if not response_text:
                app.logger.info("Gemini failed, trying Ollama as fallback...")
                response_text = try_ollama()
        else:
            # English: Try Ollama first, then Gemini
            app.logger.info("English detected - trying Ollama first")
            response_text = try_ollama()
            if not response_text:
                app.logger.info("Ollama failed, trying Gemini as fallback...")
                response_text = try_gemini()
        
        # If we got a response, return it
        if response_text:
            return jsonify({'response': response_text})
        
        # Both AI services failed, use rule-based fallback
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key or api_key.strip() == 'your_api_key_here' or api_key.strip() == '':
            # Smart fallback: Try to answer basic questions using available data
            question_lower = question.lower()
            
            # Check for farm-specific queries
            farm_mentioned = None
            for farm in ['farma', 'farm a', 'farmb', 'farm b', 'farmc', 'farm c', 'farmd', 'farm d']:
                if farm in question_lower:
                    farm_mentioned = farm.replace(' ', '').upper()
                    break
            
            # Check for comparison queries
            if any(word in question_lower for word in ['compare', 'comparison', 'which farm', 'best farm', 'worst farm']):
                if 'yield' in question_lower or 'production' in question_lower:
                    response_text = get_yield_comparison(all_farms)
                elif 'spoilage' in question_lower or 'spoil' in question_lower:
                    response_text = get_spoilage_comparison(all_farms)
                elif 'satisfaction' in question_lower or 'customer' in question_lower:
                    response_text = get_satisfaction_comparison(all_farms)
                else:
                    # General comparison
                    response_text = "Here's a quick comparison:\n\n"
                    response_text += get_yield_comparison(all_farms) + "\n\n"
                    response_text += get_spoilage_comparison(all_farms)
            # Check for specific farm summary
            elif farm_mentioned and farm_mentioned in all_farms:
                response_text = get_farm_summary(farm_mentioned, all_farms[farm_mentioned])
            # Check for summary requests
            elif any(word in question_lower for word in ['summary', 'overview', 'tell me about']):
                if farm_mentioned and farm_mentioned in all_farms:
                    response_text = get_farm_summary(farm_mentioned, all_farms[farm_mentioned])
                else:
                    response_text = "Here's an overview of all farms:\n\n"
                    for farm_name, farm_data in all_farms.items():
                        response_text += get_farm_summary(farm_name, farm_data) + "\n\n"
            # Check for yield queries
            elif 'yield' in question_lower:
                response_text = get_yield_comparison(all_farms)
            # Check for spoilage queries
            elif 'spoilage' in question_lower or 'spoil' in question_lower:
                response_text = get_spoilage_comparison(all_farms)
            # Check for satisfaction queries
            elif 'satisfaction' in question_lower or 'customer' in question_lower:
                response_text = get_satisfaction_comparison(all_farms)
            # Default helpful response
            else:
                fallback_responses = {
                    'en': 'I can help you with basic farm data queries! Try asking:\n\n' +
                          '• "Compare farms" or "Which farm has the best yield?"\n' +
                          '• "Tell me about Farm A" or "Summary of Farm B"\n' +
                          '• "Show me yield comparison" or "Compare spoilage rates"\n' +
                          '• "Which farm has highest satisfaction?"\n\n' +
                          'You can also view detailed data in the dashboard sections.\n\n' +
                          'To enable full AI chat, please:\n' +
                          '1. Install and run Ollama locally (preferred), or\n' +
                          '2. Set the GEMINI_API_KEY environment variable',
                    'hi': 'मैं आपकी बुनियादी फार्म डेटा प्रश्नों में मदद कर सकता हूं! पूछने का प्रयास करें:\n\n' +
                          '• "फार्मों की तुलना करें" या "किस फार्म की उपज सबसे अच्छी है?"\n' +
                          '• "फार्म A के बारे में बताएं" या "फार्म B का सारांश"\n' +
                          '• "उपज तुलना दिखाएं" या "खराबी दरों की तुलना करें"\n\n' +
                          'पूर्ण AI चैट सक्षम करने के लिए, कृपया:\n' +
                          '1. Ollama स्थानीय रूप से इंस्टॉल और चलाएं (पसंदीदा), या\n' +
                          '2. GEMINI_API_KEY environment variable सेट करें',
                    'kn': 'ನಾನು ನಿಮಗೆ ಮೂಲಭೂತ ಫಾರ್ಮ್ ಡೇಟಾ ಪ್ರಶ್ನೆಗಳಿಗೆ ಸಹಾಯ ಮಾಡಬಹುದು! ಕೇಳಲು ಪ್ರಯತ್ನಿಸಿ:\n\n' +
                          '• "ಫಾರ್ಮ್ಗಳನ್ನು ಹೋಲಿಸಿ" ಅಥವಾ "ಯಾವ ಫಾರ್ಮ್ಗೆ ಅತ್ಯುತ್ತಮ ಇಳುವರಿ ಇದೆ?"\n' +
                          '• "ಫಾರ್ಮ್ A ಬಗ್ಗೆ ಹೇಳಿ" ಅಥವಾ "ಫಾರ್ಮ್ B ಸಾರಾಂಶ"\n' +
                          '• "ಇಳುವರಿ ಹೋಲಿಕೆ ತೋರಿಸಿ" ಅಥವಾ "ಕೆಡುವಿಕೆ ದರಗಳನ್ನು ಹೋಲಿಸಿ"\n\n' +
                          'ಪೂರ್ಣ AI ಚಾಟ್ ಸಕ್ರಿಯಗೊಳಿಸಲು, ದಯವಿಟ್ಟು:\n' +
                          '1. Ollama ಅನ್ನು ಸ್ಥಳೀಯವಾಗಿ ಸ್ಥಾಪಿಸಿ ಮತ್ತು ಚಲಾಯಿಸಿ (ಆದ್ಯತೆ), ಅಥವಾ\n' +
                          '2. GEMINI_API_KEY environment variable ಅನ್ನು ಹೊಂದಿಸಿ'
                }
                response_text = fallback_responses.get(language, fallback_responses['en'])
            
            return jsonify({'response': response_text})
        
        # Both AI services failed - return helpful error message
        error_responses = {
            'en': 'I apologize, but both AI services (Ollama and Gemini) are currently unavailable. However, I can help you with basic farm data queries! Try asking:\n\n' +
                  '• "Compare farms" or "Which farm has the best yield?"\n' +
                  '• "Tell me about Farm A" or "Summary of Farm B"\n' +
                  '• "Show me yield comparison" or "Compare spoilage rates"\n\n' +
                  'To enable AI chat:\n' +
                  '• For English: Install and run Ollama locally (preferred)\n' +
                  '• For Hindi/Kannada: Set the GEMINI_API_KEY environment variable',
            'hi': 'मैं क्षमा चाहता हूं, लेकिन दोनों AI सेवाएं (Ollama और Gemini) वर्तमान में उपलब्ध नहीं हैं। हालाँकि, मैं आपकी बुनियादी फार्म डेटा प्रश्नों में मदद कर सकता हूं!\n\n' +
                  'AI चैट सक्षम करने के लिए, कृपया GEMINI_API_KEY environment variable सेट करें।',
            'kn': 'ನಾನು ಕ್ಷಮೆ ಕೋರುತ್ತೇನೆ, ಆದರೆ ಎರಡೂ AI ಸೇವೆಗಳು (Ollama ಮತ್ತು Gemini) ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ. ಆದಾಗ್ಯೂ, ನಾನು ನಿಮಗೆ ಮೂಲಭೂತ ಫಾರ್ಮ್ ಡೇಟಾ ಪ್ರಶ್ನೆಗಳಿಗೆ ಸಹಾಯ ಮಾಡಬಹುದು!\n\n' +
                  'AI ಚಾಟ್ ಸಕ್ರಿಯಗೊಳಿಸಲು, ದಯವಿಟ್ಟು GEMINI_API_KEY environment variable ಅನ್ನು ಹೊಂದಿಸಿ।'
        }
        return jsonify({
            'response': error_responses.get(language, error_responses['en'])
        })
            
    except Exception as e:
        return jsonify({'response': f'Well, I hit a snag there: {str(e)}. Mind trying again?'})

# COMMENTED OUT: AI4Bharat Indic-TTS models for Kannada (requires TTS library installation)
# Will use gTTS for Kannada instead until TTS library is properly set up
# 
# Initialize AI4Bharat Indic-TTS models for Kannada only
# _tts_model = None
# _tts_lock = threading.Lock()
# 
# def get_ai4bharat_tts():
#     """Get or create AI4Bharat Indic-TTS model for Kannada (thread-safe)"""
#     global _tts_model
#     
#     if _tts_model is None:
#         with _tts_lock:
#             if _tts_model is None:
#                 try:
#                     print("Loading AI4Bharat Indic-TTS for Kannada...")
#                     
#                     # Check if Indic-TTS is set up
#                     try:
#                         # Try to import and use AI4Bharat Indic-TTS
#                         import sys
#                         import os
#                         
#                         # Check if Indic-TTS directory exists
#                         indic_tts_path = os.path.join(os.getcwd(), 'Indic-TTS')
#                         if not os.path.exists(indic_tts_path):
#                             print("Indic-TTS not found. Please clone: git clone https://github.com/AI4Bharat/Indic-TTS.git")
#                             print("Then follow setup instructions in the repo.")
#                             return None
#                         
#                         # Add Indic-TTS to path
#                         if indic_tts_path not in sys.path:
#                             sys.path.insert(0, indic_tts_path)
#                         
#                         # Import TTS module from Indic-TTS
#                         from TTS.bin.synthesize import Synthesizer
#                         from TTS.utils.io import load_checkpoint
#                         from TTS.utils.audio import AudioProcessor
#                         import json
#                         
#                         # Load Kannada model (assuming models are in Indic-TTS directory)
#                         # Note: The folder is "fastptich" (typo in Indic-TTS repo), not "fastpitch"
#                         kannada_model_path = os.path.join(indic_tts_path, 'Kannada', 'fastptich', 'best_model.pth')
#                         kannada_config_path = os.path.join(indic_tts_path, 'Kannada', 'config.json')
#                         kannada_vocoder_path = os.path.join(indic_tts_path, 'Kannada', 'hifigan', 'best_model.pth')
#                         kannada_vocoder_config_path = os.path.join(indic_tts_path, 'Kannada', 'hifigan', 'config.json')
#                         
#                         if not os.path.exists(kannada_model_path):
#                             print(f"Kannada model not found at {kannada_model_path}")
#                             print("Please download Kannada models from AI4Bharat Indic-TTS repository")
#                             return None
#                         
#                         # Load synthesizer
#                         _tts_model = Synthesizer(
#                             model_path=kannada_model_path,
#                             config_path=kannada_config_path,
#                             vocoder_path=kannada_vocoder_path,
#                             vocoder_config_path=kannada_vocoder_config_path
#                         )
#                         
#                         print("AI4Bharat Indic-TTS for Kannada loaded successfully!")
#                         
#                     except ImportError as e:
#                         print(f"Failed to import Indic-TTS: {e}")
#                         print("Please ensure Indic-TTS is set up correctly.")
#                         print("Clone: git clone https://github.com/AI4Bharat/Indic-TTS.git")
#                         return None
#                     except Exception as e:
#                         print(f"Failed to load AI4Bharat Indic-TTS: {e}")
#                         import traceback
#                         traceback.print_exc()
#                         return None
#                         
#                 except Exception as e:
#                     print(f"Error initializing AI4Bharat Indic-TTS: {e}")
#                     return None
#     return _tts_model

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Generate audio from text using gTTS for Kannada (AI4Bharat Indic-TTS commented out)"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'en').lower()
        
        # COMMENTED OUT: AI4Bharat Indic-TTS for Kannada
        # Only handle Kannada via API - English and Hindi use browser TTS
        # if language != 'kn':
        #     return jsonify({'error': 'This endpoint is only for Kannada. Use browser TTS for English and Hindi.'}), 400
        
        # Now using gTTS for Kannada (and any language that calls this endpoint)
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Remove HTML tags for clean speech
        clean_text = re.sub(r'<[^>]*>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            return jsonify({'error': 'No text content after cleaning'}), 400
        
        # Limit text length for faster processing
        max_length = 5000  # gTTS limit
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "..."
        
        try:
            start_time = time.time()
            
            # COMMENTED OUT: AI4Bharat Indic-TTS approach
            # synthesizer = get_ai4bharat_tts()
            # if synthesizer is None:
            #     return jsonify({
            #         'error': 'AI4Bharat Indic-TTS not available. Please set up Indic-TTS: git clone https://github.com/AI4Bharat/Indic-TTS.git'
            #     }), 500
            # 
            # # Generate audio using AI4Bharat Indic-TTS
            # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            # temp_file.close()
            # 
            # try:
            #     # Synthesize Kannada text
            #     wav = synthesizer.tts(clean_text)
            #     synthesizer.save_wav(wav, temp_file.name)
            #     
            #     # Read the generated audio file
            #     with open(temp_file.name, 'rb') as f:
            #         audio_data = f.read()
            #     
            #     # Clean up temp file
            #     os.unlink(temp_file.name)
            #     
            #     generation_time = time.time() - start_time
            #     print(f"AI4Bharat TTS generation took {generation_time:.2f}s for Kannada ({len(clean_text)} chars)")
            
            # Using gTTS for Kannada (and other languages if needed)
            from gtts import gTTS
            
            # Language code mapping
            lang_map = {
                'en': 'en',
                'hi': 'hi',
                'kn': 'kn'
            }
            tts_lang = lang_map.get(language, 'en')
            
            # Generate audio using gTTS
            tts = gTTS(text=clean_text, lang=tts_lang, slow=False, tld='com')
            
            # Save directly to memory buffer (faster than file I/O)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            generation_time = time.time() - start_time
            print(f"TTS generation (gTTS) took {generation_time:.2f}s for {language} ({len(clean_text)} chars)")
            
            # Return audio file
            response = send_file(
                audio_buffer,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='speech.mp3'
            )
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['Content-Type'] = 'audio/mpeg'
            return response
            
        except Exception as e:
            print(f"TTS error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'TTS generation failed: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'TTS error: {str(e)}'}), 500

def prepare_farm_context(all_farms):
    """Prepare comprehensive farm data context from CSV datasets"""
    context = []
    for farm_name, data in all_farms.items():
        if data.empty:
            continue
            
        score = calculate_performance_score_from_data(data)
        
        # Core Performance Metrics
        yield_val = float(data['Yield_tonnes_per_ha'].mean())
        spoilage = float(data['SpoilageRate_%'].mean())
        defects = float(data['DefectRate_%'].mean())
        waste = float(data['WastePercentage_%'].mean())
        satisfaction = float(data['SatisfactionScore_0_10'].mean())
        pest_risk = float(data['PestRiskScore'].mean())
        machinery_uptime = float(data['MachineryUptime_%'].mean())
        harvest_uptime = float(data['HarvestRobotUptime_%'].mean())
        
        # Production Metrics
        soil_moisture = float(data['SoilMoisture_%'].mean())
        temperature = float(data['Temperature_C'].mean())
        rainfall = float(data['Rainfall_mm'].mean())
        fertilizer = float(data['Fertilizer_kg_per_ha'].mean())
        
        # Storage Metrics
        storage_temp = float(data['StorageTemperature_C'].mean())
        humidity = float(data['Humidity_%'].mean())
        shelf_life = float(data['PredictedShelfLife_days'].mean())
        storage_days = float(data['StorageDays'].mean())
        grading_score = data['GradingScore'].mode()[0] if not data['GradingScore'].mode().empty else 'N/A'
        
        # Processing Metrics
        process_types = data['ProcessType'].value_counts().to_dict()
        packaging_types = data['PackagingType'].value_counts().to_dict()
        packaging_speed = float(data['PackagingSpeed_units_per_min'].mean())
        
        # Transportation Metrics
        transport_modes = data['TransportMode'].value_counts().to_dict()
        avg_distance = float(data['TransportDistance_km'].mean())
        fuel_usage = float(data['FuelUsage_L_per_100km'].mean())
        delivery_time = float(data['DeliveryTime_hr'].mean())
        delay_percentage = float((data['DeliveryDelayFlag'].sum() / len(data)) * 100)
        spoilage_in_transit = float(data['SpoilageInTransit_%'].mean())
        
        # Retail Metrics
        inventory = float(data['RetailInventory_units'].mean())
        sales_velocity = float(data['SalesVelocity_units_per_day'].mean())
        pricing_index = float(data['DynamicPricingIndex'].mean())
        
        # Consumption Metrics
        household_waste = float(data['HouseholdWaste_kg'].mean())
        recipe_accuracy = float(data['RecipeRecommendationAccuracy_%'].mean())
        
        # Waste Management Metrics
        waste_types = data['WasteType'].value_counts().to_dict()
        segregation_accuracy = float(data['SegregationAccuracy_%'].mean())
        upcycling_rate = float(data['UpcyclingRate_%'].mean())
        biogas_output = float(data['BiogasOutput_m3'].mean())
        
        # Get crop types and their detailed metrics for this farm
        crop_yields = data.groupby('CropType')['Yield_tonnes_per_ha'].mean().to_dict()
        crop_spoilage = data.groupby('CropType')['SpoilageRate_%'].mean().to_dict()
        crop_waste = data.groupby('CropType')['WastePercentage_%'].mean().to_dict()
        crop_defects = data.groupby('CropType')['DefectRate_%'].mean().to_dict()
        
        # Build comprehensive context
        context.append(f"=== {farm_name} (Performance Score: {score:.0f}/100) ===")
        context.append(f"OVERALL METRICS: Yield:{yield_val:.1f}t/ha | Spoilage:{spoilage:.1f}% | Defects:{defects:.1f}% | Waste:{waste:.1f}% | Satisfaction:{satisfaction:.1f}/10")
        context.append(f"PRODUCTION: Soil Moisture:{soil_moisture:.1f}% | Temp:{temperature:.1f}°C | Rainfall:{rainfall:.1f}mm | Fertilizer:{fertilizer:.1f}kg/ha | Pest Risk:{pest_risk:.1f} | Machinery Uptime:{machinery_uptime:.1f}% | Harvest Robot Uptime:{harvest_uptime:.1f}%")
        context.append(f"STORAGE: Temp:{storage_temp:.1f}°C | Humidity:{humidity:.1f}% | Shelf Life:{shelf_life:.1f} days | Storage Days:{storage_days:.1f} | Grading:{grading_score}")
        context.append(f"PROCESSING: Main Process Types:{', '.join([f'{k}({v})' for k,v in list(process_types.items())[:3]])} | Packaging Types:{', '.join([f'{k}({v})' for k,v in list(packaging_types.items())[:3]])} | Packaging Speed:{packaging_speed:.0f} units/min")
        context.append(f"TRANSPORTATION: Modes:{', '.join([f'{k}({v})' for k,v in list(transport_modes.items())[:3]])} | Avg Distance:{avg_distance:.1f}km | Fuel:{fuel_usage:.1f}L/100km | Delivery Time:{delivery_time:.1f}hr | Delays:{delay_percentage:.1f}% | Spoilage in Transit:{spoilage_in_transit:.2f}%")
        context.append(f"RETAIL: Inventory:{inventory:.0f} units | Sales Velocity:{sales_velocity:.0f} units/day | Pricing Index:{pricing_index:.2f}")
        context.append(f"CONSUMPTION: Household Waste:{household_waste:.2f}kg | Recipe Accuracy:{recipe_accuracy:.1f}%")
        context.append(f"WASTE MANAGEMENT: Types:{', '.join([f'{k}({v})' for k,v in list(waste_types.items())[:3]])} | Segregation:{segregation_accuracy:.1f}% | Upcycling:{upcycling_rate:.1f}% | Biogas:{biogas_output:.1f}m³")
        context.append(f"CROP BREAKDOWN:")
        for crop in crop_yields.keys():
            context.append(f"  - {crop}: Yield:{crop_yields[crop]:.1f}t/ha | Spoilage:{crop_spoilage.get(crop,0):.1f}% | Waste:{crop_waste.get(crop,0):.1f}% | Defects:{crop_defects.get(crop,0):.1f}%")
        context.append("")  # Empty line between farms
    
    return "\n".join(context)

def answer_question(question, all_farms):
    """Answer questions based on farm data"""
    question_lower = question.lower()
    
    # Farm performance questions
    if any(word in question_lower for word in ['best', 'top', 'highest', 'excellent', 'performing']):
        return get_best_farm_info(all_farms)
    
    if any(word in question_lower for word in ['worst', 'lowest', 'poor', 'needs attention', 'worst performing']):
        return get_worst_farm_info(all_farms)
    
    # Specific farm questions
    for farm_name in ['farma', 'farm a', 'farb', 'farm b', 'farc', 'farm c', 'farmd', 'farm d']:
        if farm_name in question_lower:
            actual_farm = 'FarmA' if 'a' in farm_name else 'FarmB' if 'b' in farm_name else 'FarmC' if 'c' in farm_name else 'FarmD'
            return get_farm_summary(actual_farm, all_farms.get(actual_farm))
    
    # Metric-specific questions
    if any(word in question_lower for word in ['yield', 'production']):
        return get_yield_comparison(all_farms)
    
    if any(word in question_lower for word in ['spoilage', 'spoil']):
        return get_spoilage_comparison(all_farms)
    
    if any(word in question_lower for word in ['waste']):
        return get_waste_comparison(all_farms)
    
    if any(word in question_lower for word in ['satisfaction', 'customer']):
        return get_satisfaction_comparison(all_farms)
    
    if any(word in question_lower for word in ['pest', 'pest risk']):
        return get_pest_comparison(all_farms)
    
    if any(word in question_lower for word in ['machinery', 'uptime', 'downtime']):
        return get_machinery_comparison(all_farms)
    
    if any(word in question_lower for word in ['defect', 'defects']):
        return get_defect_comparison(all_farms)
    
    if any(word in question_lower for word in ['delay', 'delivery']):
        return get_delay_comparison(all_farms)
    
    if any(word in question_lower for word in ['storage', 'temperature', 'humidity']):
        return get_storage_comparison(all_farms)
    
    # Comparison questions
    if any(word in question_lower for word in ['compare', 'comparison', 'difference']):
        return get_general_comparison(all_farms)
    
    # Score questions
    if any(word in question_lower for word in ['score', 'performance score', 'rating']):
        return get_performance_scores(all_farms)
    
    # General help
    if any(word in question_lower for word in ['help', 'what can', 'how can', 'what do you']):
        return get_help_message()
    
    # Default response with suggestions
    return f"I understand you're asking about: '{question}'. Here's what I can help with:\n\n" + get_help_message()

def get_best_farm_info(all_farms):
    """Get information about the best performing farm"""
    scores = {}
    for farm_name, data in all_farms.items():
        score = calculate_performance_score_from_data(data)
        scores[farm_name] = score
    
    best_farm = max(scores, key=scores.get)
    best_score = scores[best_farm]
    data = all_farms[best_farm]
    
    yield_val = float(data['Yield_tonnes_per_ha'].mean())
    spoilage = float(data['SpoilageRate_%'].mean())
    waste = float(data['WastePercentage_%'].mean())
    satisfaction = float(data['SatisfactionScore_0_10'].mean())
    
    return f"🏆 {best_farm} is the best performing farm with a score of {best_score:.1f}/100!\n\n" \
           f"Key metrics:\n" \
           f"• Yield: {yield_val:.2f} tonnes/ha\n" \
           f"• Spoilage: {spoilage:.2f}%\n" \
           f"• Waste: {waste:.2f}%\n" \
           f"• Customer Satisfaction: {satisfaction:.1f}/10\n\n" \
           f"This farm demonstrates excellent performance across multiple metrics."

def get_worst_farm_info(all_farms):
    """Get information about the worst performing farm"""
    scores = {}
    for farm_name, data in all_farms.items():
        score = calculate_performance_score_from_data(data)
        scores[farm_name] = score
    
    worst_farm = min(scores, key=scores.get)
    worst_score = scores[worst_farm]
    data = all_farms[worst_farm]
    
    yield_val = float(data['Yield_tonnes_per_ha'].mean())
    spoilage = float(data['SpoilageRate_%'].mean())
    waste = float(data['WastePercentage_%'].mean())
    satisfaction = float(data['SatisfactionScore_0_10'].mean())
    
    return f"⚠️ {worst_farm} needs attention with a score of {worst_score:.1f}/100.\n\n" \
           f"Key metrics:\n" \
           f"• Yield: {yield_val:.2f} tonnes/ha\n" \
           f"• Spoilage: {spoilage:.2f}%\n" \
           f"• Waste: {waste:.2f}%\n" \
           f"• Customer Satisfaction: {satisfaction:.1f}/10\n\n" \
           f"Recommendations: Focus on reducing spoilage and waste, improving yield, and enhancing customer satisfaction."

def get_farm_summary(farm_name, data):
    """Get summary for a specific farm"""
    if data is None or data.empty:
        return f"Sorry, I couldn't find data for {farm_name}."
    
    score = calculate_performance_score_from_data(data)
    yield_val = float(data['Yield_tonnes_per_ha'].mean())
    spoilage = float(data['SpoilageRate_%'].mean())
    defects = float(data['DefectRate_%'].mean())
    waste = float(data['WastePercentage_%'].mean())
    satisfaction = float(data['SatisfactionScore_0_10'].mean())
    pest_risk = float(data['PestRiskScore'].mean())
    machinery_uptime = float(data['MachineryUptime_%'].mean())
    
    status = "Excellent" if score >= 80 else "Good" if score >= 65 else "Average" if score >= 50 else "Needs Attention"
    
    return f"📊 {farm_name} Summary:\n\n" \
           f"Performance Score: {score:.1f}/100 ({status})\n\n" \
           f"Key Metrics:\n" \
           f"• Yield: {yield_val:.2f} tonnes/ha\n" \
           f"• Spoilage Rate: {spoilage:.2f}%\n" \
           f"• Defect Rate: {defects:.2f}%\n" \
           f"• Waste: {waste:.2f}%\n" \
           f"• Customer Satisfaction: {satisfaction:.1f}/10\n" \
           f"• Pest Risk: {pest_risk:.1f}\n" \
           f"• Machinery Uptime: {machinery_uptime:.1f}%\n\n" \
           f"Total Records: {len(data)}"

def get_yield_comparison(all_farms):
    """Compare yields across farms"""
    yields = {}
    for farm_name, data in all_farms.items():
        yields[farm_name] = float(data['Yield_tonnes_per_ha'].mean())
    
    sorted_farms = sorted(yields.items(), key=lambda x: x[1], reverse=True)
    response = "🌱 Yield Comparison (tonnes/ha):\n\n"
    for farm, yield_val in sorted_farms:
        response += f"• {farm}: {yield_val:.2f}\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n🏆 Best: {best[0]} ({best[1]:.2f})\n"
    response += f"⚠️ Needs Improvement: {worst[0]} ({worst[1]:.2f})"
    
    return response

def get_spoilage_comparison(all_farms):
    """Compare spoilage rates"""
    spoilage = {}
    for farm_name, data in all_farms.items():
        spoilage[farm_name] = float(data['SpoilageRate_%'].mean())
    
    sorted_farms = sorted(spoilage.items(), key=lambda x: x[1])
    response = "❄️ Spoilage Rate Comparison (%):\n\n"
    for farm, rate in sorted_farms:
        response += f"• {farm}: {rate:.2f}%\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n✅ Lowest: {best[0]} ({best[1]:.2f}%)\n"
    response += f"⚠️ Highest: {worst[0]} ({worst[1]:.2f}%)"
    
    return response

def get_waste_comparison(all_farms):
    """Compare waste percentages"""
    waste = {}
    for farm_name, data in all_farms.items():
        waste[farm_name] = float(data['WastePercentage_%'].mean())
    
    sorted_farms = sorted(waste.items(), key=lambda x: x[1])
    response = "♻️ Waste Comparison (%):\n\n"
    for farm, rate in sorted_farms:
        response += f"• {farm}: {rate:.2f}%\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n✅ Lowest: {best[0]} ({best[1]:.2f}%)\n"
    response += f"⚠️ Highest: {worst[0]} ({worst[1]:.2f}%)"
    
    return response

def get_satisfaction_comparison(all_farms):
    """Compare customer satisfaction"""
    satisfaction = {}
    for farm_name, data in all_farms.items():
        satisfaction[farm_name] = float(data['SatisfactionScore_0_10'].mean())
    
    sorted_farms = sorted(satisfaction.items(), key=lambda x: x[1], reverse=True)
    response = "👥 Customer Satisfaction Comparison (/10):\n\n"
    for farm, score in sorted_farms:
        response += f"• {farm}: {score:.2f}\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n🏆 Highest: {best[0]} ({best[1]:.2f})\n"
    response += f"⚠️ Lowest: {worst[0]} ({worst[1]:.2f})"
    
    return response

def get_pest_comparison(all_farms):
    """Compare pest risk scores"""
    pest = {}
    for farm_name, data in all_farms.items():
        pest[farm_name] = float(data['PestRiskScore'].mean())
    
    sorted_farms = sorted(pest.items(), key=lambda x: x[1])
    response = "🐛 Pest Risk Comparison:\n\n"
    for farm, score in sorted_farms:
        response += f"• {farm}: {score:.1f}\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n✅ Lowest Risk: {best[0]} ({best[1]:.1f})\n"
    response += f"⚠️ Highest Risk: {worst[0]} ({worst[1]:.1f})"
    
    return response

def get_machinery_comparison(all_farms):
    """Compare machinery uptime"""
    uptime = {}
    for farm_name, data in all_farms.items():
        uptime[farm_name] = float(data['MachineryUptime_%'].mean())
    
    sorted_farms = sorted(uptime.items(), key=lambda x: x[1], reverse=True)
    response = "⚙️ Machinery Uptime Comparison (%):\n\n"
    for farm, rate in sorted_farms:
        response += f"• {farm}: {rate:.1f}%\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n🏆 Best: {best[0]} ({best[1]:.1f}%)\n"
    response += f"⚠️ Needs Improvement: {worst[0]} ({worst[1]:.1f}%)"
    
    return response

def get_defect_comparison(all_farms):
    """Compare defect rates"""
    defects = {}
    for farm_name, data in all_farms.items():
        defects[farm_name] = float(data['DefectRate_%'].mean())
    
    sorted_farms = sorted(defects.items(), key=lambda x: x[1])
    response = "🔧 Defect Rate Comparison (%):\n\n"
    for farm, rate in sorted_farms:
        response += f"• {farm}: {rate:.2f}%\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n✅ Lowest: {best[0]} ({best[1]:.2f}%)\n"
    response += f"⚠️ Highest: {worst[0]} ({worst[1]:.2f}%)"
    
    return response

def get_delay_comparison(all_farms):
    """Compare delivery delays"""
    delays = {}
    for farm_name, data in all_farms.items():
        delays[farm_name] = float((data['DeliveryDelayFlag'].sum() / len(data) * 100))
    
    sorted_farms = sorted(delays.items(), key=lambda x: x[1])
    response = "🚚 Delivery Delay Comparison (%):\n\n"
    for farm, rate in sorted_farms:
        response += f"• {farm}: {rate:.1f}%\n"
    
    best = sorted_farms[0]
    worst = sorted_farms[-1]
    response += f"\n✅ Lowest: {best[0]} ({best[1]:.1f}%)\n"
    response += f"⚠️ Highest: {worst[0]} ({worst[1]:.1f}%)"
    
    return response

def get_storage_comparison(all_farms):
    """Compare storage conditions"""
    response = "❄️ Storage Conditions Comparison:\n\n"
    for farm_name, data in all_farms.items():
        temp = float(data['StorageTemperature_C'].mean())
        humidity = float(data['Humidity_%'].mean())
        spoilage = float(data['SpoilageRate_%'].mean())
        response += f"{farm_name}:\n"
        response += f"  • Temperature: {temp:.1f}°C\n"
        response += f"  • Humidity: {humidity:.1f}%\n"
        response += f"  • Spoilage: {spoilage:.2f}%\n\n"
    
    return response

def get_general_comparison(all_farms):
    """Get general comparison of all farms"""
    scores = {}
    for farm_name, data in all_farms.items():
        scores[farm_name] = calculate_performance_score_from_data(data)
    
    sorted_farms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    response = "📊 Overall Farm Comparison:\n\n"
    response += "Performance Scores:\n"
    for farm, score in sorted_farms:
        status = "Excellent" if score >= 80 else "Good" if score >= 65 else "Average" if score >= 50 else "Needs Attention"
        response += f"• {farm}: {score:.1f}/100 ({status})\n"
    
    return response

def get_performance_scores(all_farms):
    """Get performance scores for all farms"""
    scores = {}
    for farm_name, data in all_farms.items():
        scores[farm_name] = calculate_performance_score_from_data(data)
    
    sorted_farms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    response = "📈 Performance Scores:\n\n"
    for farm, score in sorted_farms:
        status = "Excellent ✅" if score >= 80 else "Good ✅" if score >= 65 else "Average ⚠️" if score >= 50 else "Needs Attention ❌"
        response += f"• {farm}: {score:.1f}/100 - {status}\n"
    
    return response

def get_help_message():
    """Get help message with available commands"""
    return "I can help you with:\n\n" \
           "• Farm Performance: 'Which farm is best?', 'Show me Farm A'\n" \
           "• Metrics: 'Compare yields', 'Show spoilage rates', 'Waste comparison'\n" \
           "• Specific Data: 'Pest risk', 'Machinery uptime', 'Customer satisfaction'\n" \
           "• Comparisons: 'Compare all farms', 'Performance scores'\n" \
           "• Storage: 'Storage conditions', 'Temperature comparison'\n\n" \
           "Just ask me anything about the farms!"

def calculate_performance_score_from_data(data):
    """Calculate performance score from farm data"""
    yield_score = min(20, (float(data['Yield_tonnes_per_ha'].mean()) / 10) * 20)
    spoilage_score = max(0, 15 - (float(data['SpoilageRate_%'].mean()) / 2))
    defect_score = max(0, 15 - (float(data['DefectRate_%'].mean()) / 0.5))
    delay_score = max(0, 10 - (float(data['DeliveryDelayFlag'].sum() / len(data) * 100) / 2))
    waste_score = max(0, 10 - (float(data['WastePercentage_%'].mean()) / 2))
    satisfaction_score = (float(data['SatisfactionScore_0_10'].mean()) / 10) * 15
    pest_score = max(0, 10 - (float(data['PestRiskScore'].mean()) / 10))
    uptime_score = (float(data['MachineryUptime_%'].mean()) / 100) * 5
    
    total = yield_score + spoilage_score + defect_score + delay_score + waste_score + satisfaction_score + pest_score + uptime_score
    return min(100, max(0, total))
# --- Model Loading (Runs once at startup) ---
# Define the crops for which you have models
CROPS = ['wheat', 'corn', 'lettuce', 'tomato']  # Base crop names
MODEL_PATH = 'models/' 
# Placeholder for the loaded models
trained_models = {}
model_lambdas = {}  # Store Box-Cox lambda values

def load_models():
    """Load all trained SARIMA models into memory, handling Box-Cox transformations."""
    for crop in CROPS:
        model_file = os.path.join(MODEL_PATH, f'sarima_{crop}_price_model.pkl')
        
        if os.path.exists(model_file):
            print(f"Loading model for {crop} from {model_file}...")
            try:
                loaded_obj = joblib.load(model_file)
                # Handle both dictionary and direct ARIMA object formats
                if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                    trained_models[crop] = loaded_obj['model']
                    # Store lambda value if present (Box-Cox transformation)
                    if 'lambda' in loaded_obj:
                        model_lambdas[crop] = loaded_obj['lambda']
                        print(f"  Box-Cox lambda for {crop}: {loaded_obj['lambda']:.4f}")
                    else:
                        model_lambdas[crop] = None
                else:
                    trained_models[crop] = loaded_obj
                    model_lambdas[crop] = None
                print(f"Successfully loaded model for {crop}")
            except Exception as e:
                print(f"ERROR loading model for {crop}: {e}")
                trained_models[crop] = None
                model_lambdas[crop] = None
        else:
            print(f"WARNING: Model file not found for {crop} at {model_file}")
            trained_models[crop] = None
            model_lambdas[crop] = None

# Load models when the application starts
load_models() 

# --- New API Endpoint: Price Prediction (Updated for Box-Cox) ---
@app.route('/api/prediction/price/<crop_name>', methods=['GET'])
def predict_price(crop_name):
    """
    Predicts the crop price for the next 6 months using a loaded SARIMA model.
    Query parameters: ?months=N (default is 6)
    Applies inverse Box-Cox transformation if the model was trained with Box-Cox.
    """
    crop_name = crop_name.lower()
    
    # 1. Input Validation and Model Check
    if crop_name not in trained_models:
        return jsonify({"error": f"Model for crop '{crop_name}' not found. Available: {list(trained_models.keys())}"}), 404
    
    fitted_model = trained_models[crop_name]
    
    if fitted_model is None:
        return jsonify({"error": f"Model for crop '{crop_name}' could not be loaded."}), 500

    try:
        # Get forecast length from query string (default to 6 months)
        forecast_months = int(request.args.get('months', 6))
        
        # Perform the Forecast
        forecast_results = fitted_model.predict(
            n_periods=forecast_months, 
            return_conf_int=True, 
            alpha=0.05
        )
        
        forecast_values = forecast_results[0]
        conf_int = forecast_results[1]
        
        # Apply inverse Box-Cox transformation if lambda is available
        lam = model_lambdas.get(crop_name)
        if lam is not None:
            from scipy.special import inv_boxcox
            forecast_values = inv_boxcox(forecast_values, lam)
            conf_int[:, 0] = inv_boxcox(conf_int[:, 0], lam)
            conf_int[:, 1] = inv_boxcox(conf_int[:, 1], lam)
        
        # Generate the Date Index
        last_date = fitted_model.arima_res_.data.dates[-1] 
        forecast_index = pd.date_range(start=last_date, periods=forecast_months + 1, freq='MS')[1:]

        # Format the Output for JSON (Display calculated prices)
        forecast_data = [
            {
                "date": date.strftime('%Y-%m-%d'),
                "predicted_price": round(float(price), 2),
                "lower_ci": round(float(conf_int[i, 0]), 2),
                "upper_ci": round(float(conf_int[i, 1]), 2),
            } 
            for i, (date, price) in enumerate(zip(forecast_index, forecast_values))
        ]

        return jsonify({
            "crop": crop_name.capitalize(),
            "forecast_months": forecast_months,
            "currency": "₹",
            "forecast_data": forecast_data
        })

    except Exception as e:
        app.logger.error(f"Error during prediction for {crop_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# --- HTML Endpoint: Display all price predictions with graphs ---
@app.route('/api/prediction/price', methods=['GET'])
def predict_price_all():
    """
    Displays price predictions for all trained crops in an HTML page with interactive graphs.
    Each crop gets a bold title and a Chart.js visualization with confidence intervals.
    """
    try:
        forecast_months = int(request.args.get('months', 6))
        
        # Collect forecast data for all crops with trained models
        all_forecasts = {}
        for crop in trained_models.keys():
            if trained_models[crop] is None:
                continue
            
            try:
                model = trained_models[crop]
                
                # Perform the Forecast
                forecast_results = model.predict(
                    n_periods=forecast_months, 
                    return_conf_int=True, 
                    alpha=0.05
                )
                
                forecast_values = forecast_results[0]
                conf_int = forecast_results[1]
                
                # Apply inverse Box-Cox transformation if lambda is available
                lam = model_lambdas.get(crop)
                if lam is not None:
                    from scipy.special import inv_boxcox
                    forecast_values = inv_boxcox(forecast_values, lam)
                    conf_int[:, 0] = inv_boxcox(conf_int[:, 0], lam)
                    conf_int[:, 1] = inv_boxcox(conf_int[:, 1], lam)
                
                # Generate the Date Index
                last_date = model.arima_res_.data.dates[-1] 
                forecast_index = pd.date_range(start=last_date, periods=forecast_months + 1, freq='MS')[1:]
                
                # Format the Output
                forecast_data = {
                    "dates": [date.strftime('%Y-%m-%d') for date in forecast_index],
                    "prices": [float(round(price, 2)) for price in forecast_values],
                    "lower_ci": [float(round(conf_int[i, 0], 2)) for i in range(len(forecast_values))],
                    "upper_ci": [float(round(conf_int[i, 1], 2)) for i in range(len(forecast_values))],
                }
                
                all_forecasts[crop.capitalize()] = forecast_data
                
            except Exception as e:
                print(f"Error generating forecast for {crop}: {e}")
                all_forecasts[crop.capitalize()] = None
        
        if not all_forecasts:
            return "<h1>No trained models available</h1>", 404
        
        # Generate HTML with Chart.js graphs
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Predictions - All Crops</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
            padding: 30px 20px; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .header { 
            text-align: center; 
            color: white; 
            margin-bottom: 40px; 
        }
        .header h1 { 
            font-size: 36px; 
            margin-bottom: 10px; 
        }
        .header p { 
            font-size: 16px; 
            opacity: 0.9; 
        }
        .crop-section { 
            background: white; 
            border-radius: 12px; 
            padding: 30px; 
            margin-bottom: 30px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.2); 
        }
        .crop-title { 
            font-size: 24px; 
            font-weight: bold; 
            color: #1a5f7a; 
            margin-bottom: 20px; 
            padding-bottom: 10px; 
            border-bottom: 3px solid #667eea; 
        }
        .chart-wrapper { 
            position: relative; 
            height: 400px; 
            margin-bottom: 20px; 
        }
        .forecast-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 20px; 
            font-size: 14px; 
        }
        .forecast-table thead { 
            background-color: #1a5f7a; 
            color: white; 
        }
        .forecast-table th { 
            padding: 12px; 
            text-align: left; 
            font-weight: 600; 
        }
        .forecast-table td { 
            padding: 10px 12px; 
            border-bottom: 1px solid #ddd; 
        }
        .forecast-table tbody tr:hover { 
            background-color: #f5f5f5; 
        }
        .forecast-table tbody tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        .back-btn { 
            display: inline-block; 
            background: white; 
            color: #667eea; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 12px; 
            cursor: pointer; 
            font-size: 14px; 
            font-weight: 600; 
            text-decoration: none; 
            transition: all 0.3s; 
            margin-top: 10px; 
        }
        .back-btn:hover { 
            background: #f0f0f0; 
            transform: translateY(-2px); 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
        }
        .no-data { 
            color: #e74c3c; 
            text-align: center; 
            padding: 20px; 
            background-color: #fadbd8; 
            border-radius: 8px; 
            margin-bottom: 20px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Price Predictions</h1>
            <p>6-Month Forecast with 95% Confidence Intervals (₹/quintal)</p>
            <a href="/" class="back-btn">← Back to Dashboard</a>
        </div>
"""
        
        # Add each crop's forecast section
        for crop_name, forecast_data in all_forecasts.items():
            if forecast_data is None:
                html_content += f"""
        <div class="crop-section">
            <div class="crop-title">{crop_name}</div>
            <div class="no-data">Unable to generate forecast for {crop_name}</div>
        </div>
"""
                continue
            
            chart_id = f"chart_{crop_name.lower().replace(' ', '_')}"
            dates = forecast_data['dates']
            prices = forecast_data['prices']
            lower_ci = forecast_data['lower_ci']
            upper_ci = forecast_data['upper_ci']
            
            # Use raw calculated values directly
            
            html_content += f"""
        <div class="crop-section">
            <div class="crop-title">{crop_name}</div>
            <div class="chart-wrapper">
                <canvas id="{chart_id}"></canvas>
            </div>
            <table class="forecast-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Predicted Price (₹/quintal)</th>
                        <th>Lower CI (₹/quintal)</th>
                        <th>Upper CI (₹/quintal)</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for i, date in enumerate(dates):
                html_content += f"""
                    <tr>
                        <td>{date}</td>
                        <td>{prices[i]:,.2f}</td>
                        <td>{lower_ci[i]:,.2f}</td>
                        <td>{upper_ci[i]:,.2f}</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
"""
            
            # Add Chart.js script for this crop
            html_content += f"""
    <script>
        const ctx_{chart_id} = document.getElementById('{chart_id}').getContext('2d');
        new Chart(ctx_{chart_id}, {{
            type: 'line',
            data: {{
                labels: {dates},
                datasets: [
                    {{
                        label: 'Predicted Price',
                        data: {prices},
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 6,
                        pointBackgroundColor: '#667eea',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointHoverRadius: 8
                    }},
                    {{
                        label: 'Upper Confidence Interval (95%)',
                        data: {upper_ci},
                        borderColor: '#e74c3c',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }},
                    {{
                        label: 'Lower Confidence Interval (95%)',
                        data: {lower_ci},
                        borderColor: '#e74c3c',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                        labels: {{
                            font: {{ size: 12, weight: 'bold' }},
                            padding: 15,
                            usePointStyle: true
                        }}
                    }},
                    title: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        ticks: {{
                            callback: function(value) {{
                                return '₹' + value.toLocaleString('en-IN');
                            }},
                            font: {{ size: 11 }}
                        }},
                        title: {{
                            display: true,
                            text: 'Price (₹/quintal)',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }},
                    x: {{
                        ticks: {{
                            font: {{ size: 11 }}
                        }},
                        title: {{
                            display: true,
                            text: 'Date',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        return html_content
    
    except Exception as e:
        app.logger.error(f"Error in predict_price_all: {e}")
        import traceback
        traceback.print_exc()
        return f"<h1>Error generating predictions</h1><p>{str(e)}</p>", 500


# --- Crop Recommendation System ---
def get_average_crop_price(crop_name, months=6):
    """
    Get average predicted price for a crop over next 6 months.
    """
    try:
        if crop_name.lower() not in trained_models:
            return None
        
        model = trained_models[crop_name.lower()]
        if model is None:
            return None
        
        forecast_results = model.predict(n_periods=months, return_conf_int=True, alpha=0.05)
        forecast_values = forecast_results[0]
        
        # Apply inverse Box-Cox transformation if needed
        lam = model_lambdas.get(crop_name.lower())
        if lam is not None:
            forecast_values = inv_boxcox(forecast_values, lam)
        
        return float(np.mean(forecast_values))
    except Exception as e:
        print(f"Error predicting price for {crop_name}: {e}")
        return None


def get_farm_crop_history(farm_name):
    """
    Get historical data for each crop grown on a farm.
    Returns: dict with crop names as keys and their performance metrics as values.
    """
    data = load_farm_data(farm_name)
    if data.empty:
        return {}
    
    farm_crop_metrics = {}
    
    for crop in CROPS:
        crop_data = data[data['CropType'].str.lower() == crop.lower()]
        
        if not crop_data.empty:
            farm_crop_metrics[crop] = {
                'avg_yield': float(crop_data['Yield_tonnes_per_ha'].mean()),
                'avg_spoilage': float(crop_data['SpoilageRate_%'].mean()),
                'avg_defects': float(crop_data['DefectRate_%'].mean()),
                'avg_shelf_life': float(crop_data['PredictedShelfLife_days'].mean()),
                'avg_pest_risk': float(crop_data['PestRiskScore'].mean()),
                'records_count': len(crop_data),
                'last_grown': crop_data['HarvestDate'].max() if 'HarvestDate' in crop_data.columns else None
            }
    
    return farm_crop_metrics


def calculate_crop_profitability_score(farm_name, crop_name):
    """
    Calculate a profitability score for a crop on a specific farm.
    Considers: predicted price, historical yield, spoilage, defects, and shelf life.
    Higher score = more profitable.
    """
    # Get predicted price
    avg_price = get_average_crop_price(crop_name)
    if avg_price is None or avg_price == 0:
        # Use default price based on crop type if prediction fails
        default_prices = {
            'wheat': 2200,
            'corn': 1800,
            'lettuce': 1200,
            'tomato': 1500
        }
        avg_price = default_prices.get(crop_name.lower(), 1500)
    
    # Get farm's historical performance with this crop
    farm_metrics = get_farm_crop_history(farm_name)
    crop_metrics = farm_metrics.get(crop_name.lower(), {})
    
    # If farm hasn't grown this crop before, use industry averages
    if not crop_metrics:
        yield_score = 6.0  # Industry average
        spoilage_score = 10.0  # Industry average
        defect_score = 5.0  # Industry average
        shelf_life_score = 14.0  # Industry average
        pest_risk_score = 30.0  # Industry average
    else:
        yield_score = crop_metrics.get('avg_yield', 6.0)
        spoilage_score = crop_metrics.get('avg_spoilage', 10.0)
        defect_score = crop_metrics.get('avg_defects', 5.0)
        shelf_life_score = crop_metrics.get('avg_shelf_life', 14.0)
        pest_risk_score = crop_metrics.get('avg_pest_risk', 30.0)
    
    # Profitability calculation:
    # - Higher price = better
    # - Higher yield = better
    # - Lower spoilage = better
    # - Lower defects = better
    # - Longer shelf life = better
    # - Lower pest risk = better
    
    price_factor = (avg_price / 1000) * 30  # Price impact (normalized to ~1000 range)
    yield_factor = yield_score * 2  # Yield impact
    spoilage_factor = (20 - spoilage_score) * 1.5  # Negative impact
    defect_factor = (10 - defect_score) * 2  # Negative impact
    shelf_life_factor = shelf_life_score * 0.8  # Storage efficiency
    pest_risk_factor = (50 - pest_risk_score) * 0.5  # Health risk mitigation
    
    total_score = (
        price_factor +
        yield_factor +
        spoilage_factor +
        defect_factor +
        shelf_life_factor +
        pest_risk_factor
    )
    
    return max(0, total_score)  # Ensure non-negative


def optimize_crop_allocation():
    """
    Optimize crop allocation across all farms to prevent overlap.
    Uses a greedy algorithm to allocate crops to farms that score highest for each crop.
    
    Returns: dict with farm names as keys and recommended crop as value.
    """
    try:
        farms = list(FARM_FILES.keys())
        crops = CROPS
        
        # Calculate profitability scores for each farm-crop combination
        allocation_matrix = {}
        for farm in farms:
            allocation_matrix[farm] = {}
            for crop in crops:
                try:
                    score = calculate_crop_profitability_score(farm, crop)
                    allocation_matrix[farm][crop] = score
                except Exception as e:
                    print(f"Error calculating score for {farm}-{crop}: {e}")
                    allocation_matrix[farm][crop] = 0
        
        # Greedy allocation: assign each crop to the farm that benefits most
        allocations = {}
        remaining_crops = set(crops)
        allocated_farms = set()
        
        # Create a ranking of all farm-crop combinations
        rankings = []
        for farm in farms:
            for crop in crops:
                rankings.append({
                    'farm': farm,
                    'crop': crop,
                    'score': allocation_matrix[farm][crop]
                })
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Allocate greedily
        for ranking in rankings:
            farm = ranking['farm']
            crop = ranking['crop']
            
            # Allocate if both farm and crop are still available
            if farm not in allocated_farms and crop in remaining_crops:
                allocations[farm] = {
                    'crop': crop,
                    'score': ranking['score'],
                    'crop_scores': allocation_matrix[farm]
                }
                allocated_farms.add(farm)
                remaining_crops.remove(crop)
        
        return allocations
    except Exception as e:
        print(f"Error in optimize_crop_allocation: {e}")
        import traceback
        traceback.print_exc()
        return {}


@app.route('/api/farm/<farm_name>/crop-recommendation', methods=['GET'])
def get_crop_recommendation(farm_name):
    """
    Get personalized crop recommendation for a farm to maximize profit.
    Considers cross-farm optimization to prevent market saturation.
    """
    try:
        # Validate farm exists
        if farm_name not in FARM_FILES:
            return jsonify({'error': f'Farm {farm_name} not found'}), 404
        
        # Validate farm data exists
        farm_data = load_farm_data(farm_name)
        if farm_data.empty:
            return jsonify({'error': f'No data available for {farm_name}'}), 404
        
        # Get optimal allocation for all farms
        optimal_allocation = optimize_crop_allocation()
        
        if not optimal_allocation:
            return jsonify({'error': 'Unable to calculate allocations for any farms'}), 500
        
        if farm_name not in optimal_allocation:
            return jsonify({'error': f'Unable to calculate recommendation for {farm_name}'}), 500
        
        farm_recommendation = optimal_allocation[farm_name]
        recommended_crop = farm_recommendation['crop']
        
        # Get detailed metrics for the recommended crop
        avg_price = get_average_crop_price(recommended_crop)
        if avg_price is None:
            default_prices = {
                'wheat': 2200,
                'corn': 1800,
                'lettuce': 1200,
                'tomato': 1500
            }
            avg_price = default_prices.get(recommended_crop.lower(), 1500)
        
        farm_metrics = get_farm_crop_history(farm_name)
        crop_metrics = farm_metrics.get(recommended_crop.lower(), {})
        
        # Get all crop prices for comparison
        crop_prices = {}
        crop_scores = {}
        for crop in CROPS:
            price = get_average_crop_price(crop)
            if price is None:
                default_prices = {
                    'wheat': 2200,
                    'corn': 1800,
                    'lettuce': 1200,
                    'tomato': 1500
                }
                price = default_prices.get(crop.lower(), 1500)
            score = calculate_crop_profitability_score(farm_name, crop)
            crop_prices[crop] = round(price, 2)
            crop_scores[crop] = round(score, 2)
        
        # Calculate profit potential (based on yield, price, and efficiency)
        if crop_metrics:
            expected_yield = crop_metrics.get('avg_yield', 6.0)
            estimated_profit_per_ha = (expected_yield * avg_price) if avg_price else 0
            profit_confidence = 'High'
            experience_level = 'Experienced'
        else:
            expected_yield = 6.0  # Industry average
            estimated_profit_per_ha = (expected_yield * avg_price) if avg_price else 0
            profit_confidence = 'Medium'
            experience_level = 'New'
        
        # Get other farms' recommendations for context
        other_recommendations = {}
        for other_farm in FARM_FILES.keys():
            if other_farm != farm_name and other_farm in optimal_allocation:
                other_recommendations[other_farm] = optimal_allocation[other_farm]['crop']
        
        # Reasoning for recommendation
        reasoning = []
        if crop_metrics:
            reasoning.append(f"Your farm has strong track record with {recommended_crop.title()} (avg yield: {crop_metrics.get('avg_yield', 0):.1f}t/ha)")
        else:
            reasoning.append(f"{recommended_crop.title()} shows excellent market potential")
        
        reasoning.append(f"Predicted average price: ₹{avg_price:.2f}/quintal")
        reasoning.append(f"Profitability score: {farm_recommendation['score']:.1f} (out of 100)")
        
        if other_recommendations:
            other_crops = [v for k, v in other_recommendations.items() if v != recommended_crop]
            if other_crops:
                reasoning.append(f"Unique choice - prevents market overlap with other farms growing {', '.join(other_crops)}")
        
        return jsonify({
            'farm': farm_name,
            'recommended_crop': recommended_crop.title(),
            'recommendation_score': round(farm_recommendation['score'], 2),
            'predicted_price': round(avg_price, 2) if avg_price else 0,
            'expected_yield_tonnes_per_ha': round(expected_yield, 2),
            'estimated_profit_per_ha': round(estimated_profit_per_ha, 2),
            'profit_confidence': profit_confidence,
            'experience_level': experience_level,
            'crop_comparison': {
                'prices': crop_prices,
                'profitability_scores': crop_scores
            },
            'reasoning': reasoning,
            'other_farm_recommendations': other_recommendations,
            'optimization_note': 'Allocation optimized across all farms to prevent market saturation'
        })
    
    except Exception as e:
        app.logger.error(f"Error in crop recommendation for {farm_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/farm/crop-recommendations-all', methods=['GET'])
def get_all_crop_recommendations():
    """
    Get crop recommendations for all farms with cross-farm optimization.
    """
    try:
        optimal_allocation = optimize_crop_allocation()
        
        if not optimal_allocation:
            return jsonify({'error': 'Unable to calculate allocations'}), 500
        
        recommendations = {}
        for farm_name in FARM_FILES.keys():
            if farm_name in optimal_allocation:
                farm_rec = optimal_allocation[farm_name]
                price = get_average_crop_price(farm_rec['crop'])
                if price is None:
                    default_prices = {
                        'wheat': 2200,
                        'corn': 1800,
                        'lettuce': 1200,
                        'tomato': 1500
                    }
                    price = default_prices.get(farm_rec['crop'].lower(), 1500)
                
                recommendations[farm_name] = {
                    'recommended_crop': farm_rec['crop'].title(),
                    'profitability_score': round(farm_rec['score'], 2),
                    'predicted_price': round(price, 2)
                }
        
        return jsonify({
            'recommendations': recommendations,
            'optimization_type': 'cross-farm-optimized',
            'note': 'Each farm is assigned a unique crop to maximize market advantage'
        })
    
    except Exception as e:
        app.logger.error(f"Error in all crop recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Translation dictionaries for PDF reports
PDF_TRANSLATIONS = {
    'en': {
        'title': 'Detailed Farm Analysis Report',
        'farm': 'Farm',
        'overview': 'Overview',
        'production': 'Production',
        'storage': 'Storage',
        'processing': 'Processing',
        'transportation': 'Transportation',
        'retail': 'Retail',
        'consumption': 'Consumption',
        'waste': 'Waste',
        'carbonFootprint': 'Carbon Footprint',
        'aiInsights': 'AI Insights',
        'recommendations': 'Recommendations',
        'performanceScore': 'Performance Score',
        'yield': 'Yield',
        'spoilage': 'Spoilage',
        'defects': 'Defects',
        'delays': 'Delays',
        'waste': 'Waste',
        'satisfaction': 'Satisfaction',
        'pestRisk': 'Pest Risk',
        'machineryUptime': 'Machinery Uptime',
        'totalProduction': 'Total Production',
        'tonnesPerHa': 'tonnes/ha',
        'kgCO2e': 'kg CO2e',
        'generatedOn': 'Generated on',
        'factors': 'Factors Affecting Farm',
        'predictedProduce': 'Predicted Produce',
        'currentProduce': 'Current Produce'
    },
    'hi': {
        'title': 'विस्तृत फार्म विश्लेषण रिपोर्ट',
        'farm': 'फार्म',
        'overview': 'अवलोकन',
        'production': 'उत्पादन',
        'storage': 'भंडारण',
        'processing': 'प्रसंस्करण',
        'transportation': 'परिवहन',
        'retail': 'खुदरा',
        'consumption': 'उपभोग',
        'waste': 'अपशिष्ट',
        'carbonFootprint': 'कार्बन फुटप्रिंट',
        'aiInsights': 'AI अंतर्दृष्टि',
        'recommendations': 'सिफारिशें',
        'performanceScore': 'प्रदर्शन स्कोर',
        'yield': 'उपज',
        'spoilage': 'खराबी',
        'defects': 'दोष',
        'delays': 'देरी',
        'waste': 'अपशिष्ट',
        'satisfaction': 'संतुष्टि',
        'pestRisk': 'कीट जोखिम',
        'machineryUptime': 'मशीनरी अपटाइम',
        'totalProduction': 'कुल उत्पादन',
        'tonnesPerHa': 'टन/हे',
        'kgCO2e': 'किग्रा CO2e',
        'generatedOn': 'तैयार की गई',
        'factors': 'फार्म को प्रभावित करने वाले कारक',
        'predictedProduce': 'अनुमानित उत्पादन',
        'currentProduce': 'वर्तमान उत्पादन'
    },
    'kn': {
        'title': 'ವಿವರವಾದ ಫಾರ್ಮ್ ವಿಶ್ಲೇಷಣೆ ವರದಿ',
        'farm': 'ಫಾರ್ಮ್',
        'overview': 'ಅವಲೋಕನ',
        'production': 'ಉತ್ಪಾದನೆ',
        'storage': 'ಸಂಗ್ರಹಣೆ',
        'processing': 'ಸಂಸ್ಕರಣೆ',
        'transportation': 'ಸಾರಿಗೆ',
        'retail': 'ಚಿಲ್ಲರೆ',
        'consumption': 'ಬಳಕೆ',
        'waste': 'ಕಸ',
        'carbonFootprint': 'ಕಾರ್ಬನ್ ಫುಟ್ಪ್ರಿಂಟ್',
        'aiInsights': 'AI ಒಳನೋಟಗಳು',
        'recommendations': 'ಶಿಫಾರಸುಗಳು',
        'performanceScore': 'ಪ್ರದರ್ಶನ ಸ್ಕೋರ್',
        'yield': 'ಉತ್ಪಾದನೆ',
        'spoilage': 'ಕೆಡುವಿಕೆ',
        'defects': 'ದೋಷಗಳು',
        'delays': 'ವಿಳಂಬಗಳು',
        'waste': 'ಕಸ',
        'satisfaction': 'ತೃಪ್ತಿ',
        'pestRisk': 'ಕೀಟ ಅಪಾಯ',
        'machineryUptime': 'ಯಂತ್ರೋಪಕರಣ ಅಪ್ಟೈಮ್',
        'totalProduction': 'ಒಟ್ಟು ಉತ್ಪಾದನೆ',
        'tonnesPerHa': 'ಟನ್/ಹೆ',
        'kgCO2e': 'ಕೆಜಿ CO2e',
        'generatedOn': 'ರಚಿಸಲಾಗಿದೆ',
        'factors': 'ಫಾರ್ಮ್ ಅನ್ನು ಪ್ರಭಾವಿಸುವ ಅಂಶಗಳು',
        'predictedProduce': 'ಭವಿಷ್ಯವಾಣಿ ಉತ್ಪಾದನೆ',
        'currentProduce': 'ಪ್ರಸ್ತುತ ಉತ್ಪಾದನೆ'
    }
}


@app.route('/api/farm/<farm_name>/report', methods=['GET'])
def generate_farm_report(farm_name):
    """Generate a comprehensive PDF report for a farm in the specified language"""
    try:
        lang = request.args.get('lang', 'en')
        if lang not in ['en', 'hi', 'kn']:
            lang = 'en'
        
        t = PDF_TRANSLATIONS[lang]
        
        # Get all farm data directly
        data = load_farm_data(farm_name)
        if data.empty:
            return jsonify({'error': 'Farm not found'}), 404
        
        # Calculate KPIs
        carbon_footprint = calculate_carbon_footprint(data)
        kpis = {
            'total_production': float(data['Yield_tonnes_per_ha'].mean()),
            'storage_spoilage': float(data['SpoilageRate_%'].mean()),
            'processing_defects': float(data['DefectRate_%'].mean()),
            'transport_delays': float((data['DeliveryDelayFlag'].sum() / len(data) * 100)),
            'waste_percentage': float(data['WastePercentage_%'].mean()),
            'satisfaction': float(data['SatisfactionScore_0_10'].mean()),
            'pest_risk': float(data['PestRiskScore'].mean()),
            'machinery_uptime': float(data['MachineryUptime_%'].mean()),
            'carbon_footprint': carbon_footprint
        }
        
        # Calculate performance score
        score = (
            (kpis['total_production'] / 10 * 20) +
            ((1 - kpis['storage_spoilage'] / 30) * 15) +
            ((1 - kpis['processing_defects'] / 15) * 15) +
            ((1 - kpis['transport_delays'] / 30) * 10) +
            ((1 - kpis['waste_percentage'] / 25) * 10) +
            (kpis['satisfaction'] / 10 * 15) +
            ((1 - kpis['pest_risk'] / 100) * 10) +
            (kpis['machinery_uptime'] / 100 * 5)
        )
        kpis['performance_score'] = min(100, max(0, score))
        
        # Production data
        production = {
            'yield': float(data['Yield_tonnes_per_ha'].mean()),
            'pest_risk': float(data['PestRiskScore'].mean()),
            'machinery_uptime': float(data['MachineryUptime_%'].mean()),
            'carbon_footprint': {'fertilizer': carbon_footprint.get('fertilizer', 0) if isinstance(carbon_footprint, dict) else 0}
        }
        
        # Storage data
        storage = {
            'avg_temp': float(data['StorageTemperature_C'].mean()),
            'avg_humidity': float(data['Humidity_%'].mean()),
            'avg_spoilage': float(data['SpoilageRate_%'].mean()),
            'avg_shelf_life': float(data['PredictedShelfLife_days'].mean())
        }
        
        # Processing data
        processing = {
            'avg_defect_rate': float(data['DefectRate_%'].mean()),
            'avg_uptime': float(data['MachineryUptime_%'].mean()),
            'avg_packaging_speed': float(data['PackagingSpeed_units_per_min'].mean())
        }
        
        # Transportation data
        transportation = {
            'avg_distance': float(data['TransportDistance_km'].mean()),
            'avg_fuel': float(data['FuelUsage_L_per_100km'].mean()),
            'avg_delivery_time': float(data['DeliveryTime_hr'].mean()),
            'delay_percentage': float((data['DeliveryDelayFlag'].sum() / len(data) * 100))
        }
        
        # Retail data
        retail = {
            'total_inventory': float(data['RetailInventory_units'].sum()),
            'avg_sales_velocity': float(data['SalesVelocity_units_per_day'].mean()),
            'avg_pricing_index': float(data['DynamicPricingIndex'].mean()),
            'avg_waste': float(data['WastePercentage_%'].mean())
        }
        
        # Consumption data
        consumption = {
            'avg_household_waste': float(data['HouseholdWaste_kg'].mean()),
            'avg_recipe_accuracy': float(data['RecipeRecommendationAccuracy_%'].mean()),
            'avg_satisfaction': float(data['SatisfactionScore_0_10'].mean())
        }
        
        # Waste data
        waste = {
            'avg_segregation': float(data['SegregationAccuracy_%'].mean()),
            'avg_upcycling': float(data['UpcyclingRate_%'].mean()),
            'avg_biogas': float(data['BiogasOutput_m3'].mean())
        }
        
        # Get AI insights - ensure we always get a dict
        ai_insights = {'insights': [], 'recommendations': []}
        try:
            insights_result = get_ai_insights(farm_name, 'overview')
            if isinstance(insights_result, dict):
                # Ensure insights and recommendations are lists
                ai_insights = {
                    'insights': insights_result.get('insights', []) if isinstance(insights_result.get('insights'), list) else [],
                    'recommendations': insights_result.get('recommendations', []) if isinstance(insights_result.get('recommendations'), list) else []
                }
                # If no insights, generate some basic ones from the data
                if not ai_insights['insights']:
                    performance_score = kpis.get('performance_score', 0)
                    if performance_score >= 80:
                        ai_insights['insights'].append(f"Excellent overall performance with a score of {performance_score:.1f}/100")
                    elif performance_score >= 65:
                        ai_insights['insights'].append(f"Good performance with room for improvement (score: {performance_score:.1f}/100)")
                    else:
                        ai_insights['insights'].append(f"Performance needs attention (score: {performance_score:.1f}/100)")
                    
                    if kpis.get('storage_spoilage', 0) > 10:
                        ai_insights['recommendations'].append("Reduce storage spoilage by optimizing temperature and humidity controls")
                    if kpis.get('processing_defects', 0) > 5:
                        ai_insights['recommendations'].append("Improve quality control to reduce processing defects")
            else:
                app.logger.warning(f"AI insights returned non-dict: {type(insights_result)}")
        except Exception as e:
            app.logger.error(f"Error getting AI insights: {e}")
            import traceback
            traceback.print_exc()
            # Generate fallback insights from KPIs
            performance_score = kpis.get('performance_score', 0)
            ai_insights = {
                'insights': [f"Farm performance score: {performance_score:.1f}/100"],
                'recommendations': ["Review individual metrics in the detailed sections for specific improvements"]
            }
        
        # Get crop recommendation
        crop_rec = None
        try:
            optimal_allocation = optimize_crop_allocation()
            if optimal_allocation and isinstance(optimal_allocation, dict):
                crop_rec_data = optimal_allocation.get(farm_name)
                if crop_rec_data and isinstance(crop_rec_data, dict):
                    price = get_average_crop_price(crop_rec_data.get('crop', ''))
                    if price is None:
                        default_prices = {'wheat': 2200, 'corn': 1800, 'lettuce': 1200, 'tomato': 1500}
                        price = default_prices.get(crop_rec_data.get('crop', '').lower(), 1500)
                    crop_rec = {
                        'recommended_crop': crop_rec_data.get('crop', 'N/A').title(),
                        'recommendation_score': round(crop_rec_data.get('score', 0), 2),
                        'predicted_price': round(price, 2),
                        'reasoning': [f"High profitability score: {crop_rec_data.get('score', 0):.1f}", f"Predicted price: ₹{price:.2f}/quintal"]
                    }
        except Exception as e:
            app.logger.error(f"Error getting crop recommendation: {e}")
            import traceback
            traceback.print_exc()
            crop_rec = None
        
        # Create PDF in memory
        buffer = io.BytesIO()
        try:
            # Register Unicode fonts for Hindi and Kannada
            def register_unicode_fonts():
                """Register Unicode fonts that support Hindi and Kannada scripts"""
                try:
                    # Try to register common system fonts that support Unicode
                    font_paths = [
                        # macOS fonts (check for Hindi/Kannada specific fonts first)
                        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                        '/Library/Fonts/Arial Unicode.ttf',
                        '/System/Library/Fonts/Supplemental/Thonburi.ttc',  # Thai but has good Unicode coverage
                        '/System/Library/Fonts/Helvetica.ttc',
                        # Windows fonts
                        'C:/Windows/Fonts/arialuni.ttf',
                        'C:/Windows/Fonts/mangal.ttf',  # Hindi support
                        'C:/Windows/Fonts/nirmala.ttf',  # Devanagari
                        'C:/Windows/Fonts/nirmala-ui.ttf',  # Devanagari UI
                        'C:/Windows/Fonts/notosansdevanagari-regular.ttf',  # Noto Sans Devanagari
                        # Linux fonts
                        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
                        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
                        '/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf',
                        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    ]
                    
                    # Try to find and register a Unicode font
                    unicode_font_path = None
                    for path in font_paths:
                        if os.path.exists(path):
                            unicode_font_path = path
                            break
                    
                    if unicode_font_path:
                        try:
                            # Register the font with a unique name
                            font_name = 'UnicodeFont'
                            pdfmetrics.registerFont(TTFont(font_name, unicode_font_path))
                            app.logger.info(f"Successfully registered Unicode font: {unicode_font_path}")
                            return font_name
                        except Exception as e:
                            app.logger.warning(f"Could not register font from {unicode_font_path}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # If no system font found, we cannot generate a proper PDF for Hindi/Kannada
                    # Return None to indicate failure
                    app.logger.error("No Unicode font found for Hindi/Kannada PDF generation")
                    return None
                except Exception as e:
                    app.logger.error(f"Error registering Unicode fonts: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Helper function to safely prepare text for PDF (ensure Unicode, but preserve HTML tags)
            def safe_text(text):
                """Safely prepare text for PDF by ensuring it's a string and valid Unicode"""
                if text is None:
                    return ''
                # Convert to string if not already
                text = str(text)
                # Ensure it's valid Unicode - ReportLab handles Unicode automatically with proper font
                try:
                    # Test if text can be encoded as UTF-8
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    # If encoding fails, try to fix it by removing problematic characters
                    text = text.encode('utf-8', errors='replace').decode('utf-8')
                return text
            
            # Register fonts based on language
            font_name = None
            if lang in ['hi', 'kn']:
                font_name = register_unicode_fonts()
                # If no Unicode font is available for Hindi/Kannada, return an error
                if not font_name:
                    error_messages = {
                        'hi': 'PDF रिपोर्ट उत्पन्न करने के लिए Unicode फ़ॉन्ट उपलब्ध नहीं है। कृपया सिस्टम फ़ॉन्ट स्थापित करें या English में रिपोर्ट डाउनलोड करें।',
                        'kn': 'PDF ವರದಿಯನ್ನು ರಚಿಸಲು Unicode ಫಾಂಟ್ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸಿಸ್ಟಮ್ ಫಾಂಟ್ ಅನ್ನು ಸ್ಥಾಪಿಸಿ ಅಥವಾ ಇಂಗ್ಲೀಷ್‌ನಲ್ಲಿ ವರದಿಯನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ।'
                    }
                    return jsonify({
                        'error': error_messages.get(lang, 'Unicode font not available for PDF generation. Please install system fonts or download report in English.')
                    }), 500
            
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Update styles to use Unicode font if available
            if font_name:
                # Create custom styles with Unicode font
                # Create Unicode styles without encoding parameter (ReportLab handles Unicode automatically with proper font)
                # Get base styles first (before we create Unicode versions)
                base_normal = styles['Normal']
                base_heading1 = styles['Heading1']
                base_heading2 = styles['Heading2']
                base_heading3 = styles['Heading3']
                
                unicode_normal = ParagraphStyle(
                    'UnicodeNormal',
                    parent=base_normal,
                    fontName=font_name
                )
                unicode_heading1 = ParagraphStyle(
                    'UnicodeHeading1',
                    parent=base_heading1,
                    fontName=font_name
                )
                unicode_heading2 = ParagraphStyle(
                    'UnicodeHeading2',
                    parent=base_heading2,
                    fontName=font_name
                )
                unicode_heading3 = ParagraphStyle(
                    'UnicodeHeading3',
                    parent=base_heading3,
                    fontName=font_name
                )
                # Add Unicode styles to the stylesheet (can't replace existing styles directly)
                styles.add(unicode_normal)
                styles.add(unicode_heading1)
                styles.add(unicode_heading2)
                styles.add(unicode_heading3)
                # Store references to use later (we'll use these instead of default styles)
                unicode_styles = {
                    'Normal': unicode_normal,
                    'Heading1': unicode_heading1,
                    'Heading2': unicode_heading2,
                    'Heading3': unicode_heading3
                }
            else:
                # No Unicode font, use default styles
                unicode_styles = None
            
            # Helper to get the right style (Unicode if available, otherwise default)
            def get_style(style_name):
                if unicode_styles and style_name in unicode_styles:
                    return unicode_styles[style_name]
                return styles[style_name]
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=get_style('Heading1'),
                fontSize=24,
                textColor=colors.HexColor('#1a5f7a'),
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph(safe_text(t['title']), title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Farm name and date
            story.append(Paragraph(f"<b>{safe_text(t['farm'])}:</b> {safe_text(farm_name)}", get_style('Heading2')))
            story.append(Paragraph(f"<b>{safe_text(t['generatedOn'])}:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", get_style('Normal')))
            story.append(Spacer(1, 0.3*inch))
            
            # Overview Section
            story.append(Paragraph(f"<b>{safe_text(t['overview'])}</b>", get_style('Heading2')))
            # Helper to create table cells with proper Unicode support
            def make_table_cell(text):
                """Create a table cell with proper Unicode encoding"""
                # Just return the text as string - the table font style will handle rendering
                # Using Paragraph in tables can cause layout issues, so we rely on font styles
                return safe_text(str(text))
            
            overview_data = [
                [make_table_cell(t['performanceScore']), make_table_cell(f"{kpis.get('performance_score', 0):.1f}/100")],
                [make_table_cell(t['yield']), make_table_cell(f"{kpis.get('total_production', 0):.2f} {t['tonnesPerHa']}")],
                [make_table_cell(t['spoilage']), make_table_cell(f"{kpis.get('storage_spoilage', 0):.2f}%")],
                [make_table_cell(t['defects']), make_table_cell(f"{kpis.get('processing_defects', 0):.2f}%")],
                [make_table_cell(t['delays']), make_table_cell(f"{kpis.get('transport_delays', 0):.1f}%")],
                [make_table_cell(t['waste']), make_table_cell(f"{kpis.get('waste_percentage', 0):.2f}%")],
                [make_table_cell(t['satisfaction']), make_table_cell(f"{kpis.get('satisfaction', 0):.1f}/10")],
                [make_table_cell(t['pestRisk']), make_table_cell(f"{kpis.get('pest_risk', 0):.1f}")],
                [make_table_cell(t['machineryUptime']), make_table_cell(f"{kpis.get('machinery_uptime', 0):.1f}%")]
            ]
            if kpis.get('carbon_footprint') and isinstance(kpis['carbon_footprint'], dict):
                overview_data.append([make_table_cell(t['carbonFootprint']), make_table_cell(f"{kpis['carbon_footprint'].get('total', 0):.2f} {t['kgCO2e']}")])
            
            # Use Unicode font for tables if available, otherwise use default
            # When Unicode font is available, use it for all cells to ensure proper rendering
            if font_name:
                table_font = font_name  # Use Unicode font for labels
                table_font_normal = font_name  # Use Unicode font for all values too
            else:
                table_font = 'Helvetica-Bold'
                table_font_normal = 'Helvetica'
            
            overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(overview_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Production Section - Detailed
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(f"<b>{safe_text(t['production'])}</b>", get_style('Heading2')))
            story.append(Spacer(1, 0.15*inch))
            
            # Production metrics
            prod_data = [
                [make_table_cell(t['yield']), make_table_cell(f"{production.get('yield', 0):.2f} {t['tonnesPerHa']}")],
                [make_table_cell(t['pestRisk']), make_table_cell(f"{production.get('pest_risk', 0):.2f}")],
                [make_table_cell(t['machineryUptime']), make_table_cell(f"{production.get('machinery_uptime', 0):.2f}%")]
            ]
            
            # Add additional production metrics
            harvest_uptime = float(data['HarvestRobotUptime_%'].mean())
            fertilizer_usage = float(data['Fertilizer_kg_per_ha'].mean())
            rainfall = float(data['Rainfall_mm'].mean())
            yield_std = float(data['Yield_tonnes_per_ha'].std())
            yield_min = float(data['Yield_tonnes_per_ha'].min())
            yield_max = float(data['Yield_tonnes_per_ha'].max())
            
            prod_data.extend([
                [make_table_cell('Harvest Robot Uptime'), make_table_cell(f"{harvest_uptime:.2f}%")],
                [make_table_cell('Average Fertilizer Usage'), make_table_cell(f"{fertilizer_usage:.2f} kg/ha")],
                [make_table_cell('Average Rainfall'), make_table_cell(f"{rainfall:.2f} mm")],
                [make_table_cell('Yield Standard Deviation'), make_table_cell(f"{yield_std:.2f} {t['tonnesPerHa']}")],
                [make_table_cell('Minimum Yield'), make_table_cell(f"{yield_min:.2f} {t['tonnesPerHa']}")],
                [make_table_cell('Maximum Yield'), make_table_cell(f"{yield_max:.2f} {t['tonnesPerHa']}")]
            ])
            
            if production.get('carbon_footprint') and isinstance(production['carbon_footprint'], dict):
                prod_data.append([make_table_cell(t['carbonFootprint']), make_table_cell(f"{production['carbon_footprint'].get('fertilizer', 0):.2f} {t['kgCO2e']}")])
            
            prod_table = Table(prod_data, colWidths=[3*inch, 2*inch])
            prod_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(prod_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Yield by Crop Type
            yield_by_crop = data.groupby('CropType')['Yield_tonnes_per_ha'].agg(['mean', 'count', 'min', 'max']).round(2)
            if not yield_by_crop.empty:
                story.append(Paragraph("<b>Yield by Crop Type</b>", get_style('Heading3')))
                story.append(Spacer(1, 0.1*inch))
                crop_data = [['Crop Type', 'Avg Yield', 'Records', 'Min Yield', 'Max Yield']]
                for crop, row in yield_by_crop.iterrows():
                    crop_data.append([
                        str(crop),
                        f"{row['mean']:.2f} {t['tonnesPerHa']}",
                        str(int(row['count'])),
                        f"{row['min']:.2f} {t['tonnesPerHa']}",
                        f"{row['max']:.2f} {t['tonnesPerHa']}"
                    ])
                crop_table = Table(crop_data, colWidths=[1.2*inch, 1*inch, 0.8*inch, 1*inch, 1*inch])
                crop_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white)
                ]))
                story.append(crop_table)
                story.append(Spacer(1, 0.2*inch))
            
            # Storage, Processing, Transportation, Retail, Consumption, Waste sections
            # Define field labels for each language
            field_labels = {
                'en': {
                    'storage': [('Average Temperature', 'avg_temp', '°C'), ('Average Humidity', 'avg_humidity', '%'), ('Average Spoilage', 'avg_spoilage', '%'), ('Average Shelf Life', 'avg_shelf_life', 'days')],
                    'processing': [('Average Defect Rate', 'avg_defect_rate', '%'), ('Average Uptime', 'avg_uptime', '%'), ('Average Packaging Speed', 'avg_packaging_speed', 'units/min')],
                    'transportation': [('Average Distance', 'avg_distance', 'km'), ('Average Fuel Usage', 'avg_fuel', 'L/100km'), ('Average Delivery Time', 'avg_delivery_time', 'hours'), ('Delay Percentage', 'delay_percentage', '%')],
                    'retail': [('Total Inventory', 'total_inventory', 'units'), ('Average Sales Velocity', 'avg_sales_velocity', 'units/day'), ('Average Pricing Index', 'avg_pricing_index', ''), ('Average Waste', 'avg_waste', '%')],
                    'consumption': [('Average Household Waste', 'avg_household_waste', 'kg'), ('Average Recipe Accuracy', 'avg_recipe_accuracy', '%'), ('Average Satisfaction', 'avg_satisfaction', '/10')],
                    'waste': [('Average Segregation', 'avg_segregation', '%'), ('Average Upcycling Rate', 'avg_upcycling', '%'), ('Average Biogas Output', 'avg_biogas', 'm³')]
                },
                'hi': {
                    'storage': [('औसत तापमान', 'avg_temp', '°C'), ('औसत आर्द्रता', 'avg_humidity', '%'), ('औसत खराबी', 'avg_spoilage', '%'), ('औसत शेल्फ लाइफ', 'avg_shelf_life', 'दिन')],
                    'processing': [('औसत दोष दर', 'avg_defect_rate', '%'), ('औसत अपटाइम', 'avg_uptime', '%'), ('औसत पैकेजिंग गति', 'avg_packaging_speed', 'यूनिट/मिनट')],
                    'transportation': [('औसत दूरी', 'avg_distance', 'किमी'), ('औसत ईंधन उपयोग', 'avg_fuel', 'L/100km'), ('औसत डिलीवरी समय', 'avg_delivery_time', 'घंटे'), ('देरी प्रतिशत', 'delay_percentage', '%')],
                    'retail': [('कुल इन्वेंटरी', 'total_inventory', 'यूनिट'), ('औसत बिक्री वेग', 'avg_sales_velocity', 'यूनिट/दिन'), ('औसत मूल्य सूचकांक', 'avg_pricing_index', ''), ('औसत अपशिष्ट', 'avg_waste', '%')],
                    'consumption': [('औसत घरेलू अपशिष्ट', 'avg_household_waste', 'किग्रा'), ('औसत रेसिपी सटीकता', 'avg_recipe_accuracy', '%'), ('औसत संतुष्टि', 'avg_satisfaction', '/10')],
                    'waste': [('औसत पृथक्करण', 'avg_segregation', '%'), ('औसत अपसाइक्लिंग दर', 'avg_upcycling', '%'), ('औसत बायोगैस उत्पादन', 'avg_biogas', 'm³')]
                },
                'kn': {
                    'storage': [('ಸರಾಸರಿ ತಾಪಮಾನ', 'avg_temp', '°C'), ('ಸರಾಸರಿ ಆರ್ದ್ರತೆ', 'avg_humidity', '%'), ('ಸರಾಸರಿ ಕೆಡುವಿಕೆ', 'avg_spoilage', '%'), ('ಸರಾಸರಿ ಶೆಲ್ಫ್ ಲೈಫ್', 'avg_shelf_life', 'ದಿನಗಳು')],
                    'processing': [('ಸರಾಸರಿ ದೋಷ ದರ', 'avg_defect_rate', '%'), ('ಸರಾಸರಿ ಅಪ್ಟೈಮ್', 'avg_uptime', '%'), ('ಸರಾಸರಿ ಪ್ಯಾಕೇಜಿಂಗ್ ವೇಗ', 'avg_packaging_speed', 'ಯೂನಿಟ್/ನಿಮಿಷ')],
                    'transportation': [('ಸರಾಸರಿ ದೂರ', 'avg_distance', 'ಕಿಮೀ'), ('ಸರಾಸರಿ ಇಂಧನ ಬಳಕೆ', 'avg_fuel', 'L/100km'), ('ಸರಾಸರಿ ವಿತರಣೆ ಸಮಯ', 'avg_delivery_time', 'ಗಂಟೆಗಳು'), ('ವಿಳಂಬ ಶೇಕಡಾವಾರು', 'delay_percentage', '%')],
                    'retail': [('ಒಟ್ಟು ಸ್ಟಾಕ್', 'total_inventory', 'ಯೂನಿಟ್ಗಳು'), ('ಸರಾಸರಿ ಮಾರಾಟ ವೇಗ', 'avg_sales_velocity', 'ಯೂನಿಟ್/ದಿನ'), ('ಸರಾಸರಿ ಬೆಲೆ ಸೂಚ್ಯಂಕ', 'avg_pricing_index', ''), ('ಸರಾಸರಿ ಕಸ', 'avg_waste', '%')],
                    'consumption': [('ಸರಾಸರಿ ಮನೆ ಕಸ', 'avg_household_waste', 'ಕೆಜಿ'), ('ಸರಾಸರಿ ರೆಸಿಪಿ ನಿಖರತೆ', 'avg_recipe_accuracy', '%'), ('ಸರಾಸರಿ ತೃಪ್ತಿ', 'avg_satisfaction', '/10')],
                    'waste': [('ಸರಾಸರಿ ವಿಂಗಡಣೆ', 'avg_segregation', '%'), ('ಸರಾಸರಿ ಅಪ್ಸೈಕ್ಲಿಂಗ್ ದರ', 'avg_upcycling', '%'), ('ಸರಾಸರಿ ಬಯೋಗ್ಯಾಸ್ ಉತ್ಪಾದನೆ', 'avg_biogas', 'm³')]
                }
            }
            
            sections = [
                ('storage', storage, t['storage'], field_labels[lang]['storage']),
                ('processing', processing, t['processing'], field_labels[lang]['processing']),
                ('transportation', transportation, t['transportation'], field_labels[lang]['transportation']),
                ('retail', retail, t['retail'], field_labels[lang]['retail']),
                ('consumption', consumption, t['consumption'], field_labels[lang]['consumption']),
                ('waste', waste, t['waste'], field_labels[lang]['waste'])
            ]
            
            for section_name, section_data, section_title, section_fields in sections:
                if section_data and isinstance(section_data, dict):
                    has_content = any(section_data.get(field[1]) is not None for field in section_fields)
                    if has_content:
                        # Use Spacer instead of PageBreak to avoid blank pages
                        story.append(Spacer(1, 0.3*inch))
                        story.append(Paragraph(f"<b>{safe_text(section_title)}</b>", get_style('Heading2')))
                        story.append(Spacer(1, 0.15*inch))
                        
                        # Create table with section data
                        section_table_data = []
                        for field_label, field_key, field_unit in section_fields:
                            value = section_data.get(field_key)
                            if value is not None:
                                if isinstance(value, (int, float)):
                                    section_table_data.append([make_table_cell(field_label), make_table_cell(f"{value:.2f} {field_unit}")])
                                else:
                                    section_table_data.append([make_table_cell(field_label), make_table_cell(str(value))])
                        
                        if section_table_data:
                            section_table = Table(section_table_data, colWidths=[3*inch, 2*inch])
                            section_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('TOPPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                            ]))
                            story.append(section_table)
                            
                            # Add detailed breakdowns for each section
                            if section_name == 'storage':
                                story.append(Spacer(1, 0.15*inch))
                                story.append(Paragraph("<b>Storage Conditions Analysis</b>", get_style('Heading3')))
                                temp_min = float(data['StorageTemperature_C'].min())
                                temp_max = float(data['StorageTemperature_C'].max())
                                humidity_min = float(data['Humidity_%'].min())
                                humidity_max = float(data['Humidity_%'].max())
                                storage_days_avg = float(data['StorageDays'].mean())
                                storage_analysis = [
                                    ['Temperature Range', f"{temp_min:.1f}°C - {temp_max:.1f}°C"],
                                    ['Humidity Range', f"{humidity_min:.1f}% - {humidity_max:.1f}%"],
                                    ['Average Storage Days', f"{storage_days_avg:.1f} days"],
                                    ['Optimal Temperature Range', '2-5°C'],
                                    ['Optimal Humidity Range', '75-85%']
                                ]
                                analysis_table = Table(storage_analysis, colWidths=[3*inch, 2*inch])
                                analysis_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
                                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                    ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                                ]))
                                story.append(analysis_table)
                            
                            elif section_name == 'processing':
                                story.append(Spacer(1, 0.15*inch))
                                story.append(Paragraph("<b>Processing Details by Process Type</b>", get_style('Heading3')))
                                defect_by_process = data.groupby('ProcessType')['DefectRate_%'].agg(['mean', 'count']).round(2)
                                if not defect_by_process.empty:
                                    process_data = [['Process Type', 'Avg Defect Rate (%)', 'Records']]
                                    for process, row in defect_by_process.iterrows():
                                        process_data.append([str(process), f"{row['mean']:.2f}", str(int(row['count']))])
                                    process_table = Table(process_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                                    process_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                                    ]))
                                    story.append(process_table)
                            
                            elif section_name == 'transportation':
                                story.append(Spacer(1, 0.15*inch))
                                story.append(Paragraph("<b>Transportation Analysis</b>", get_style('Heading3')))
                                transport_modes = data['TransportMode'].value_counts()
                                total_distance = float(data['TransportDistance_km'].sum())
                                total_fuel = float((data['TransportDistance_km'] * data['FuelUsage_L_per_100km'] / 100).sum())
                                delay_count = int(data['DeliveryDelayFlag'].sum())
                                total_deliveries = len(data)
                                
                                trans_analysis = [
                                    ['Total Distance Traveled', f"{total_distance:.0f} km"],
                                    ['Total Fuel Consumed', f"{total_fuel:.2f} L"],
                                    ['Delayed Deliveries', f"{delay_count} out of {total_deliveries}"],
                                    ['On-Time Delivery Rate', f"{(total_deliveries-delay_count)/total_deliveries*100:.1f}%"]
                                ]
                                if not transport_modes.empty:
                                    trans_analysis.append(['Primary Transport Mode', transport_modes.index[0]])
                                
                                trans_table = Table(trans_analysis, colWidths=[3*inch, 2*inch])
                                trans_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8f9fa')),
                                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
                                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                                ]))
                                story.append(trans_table)
                            
                            elif section_name == 'waste':
                                story.append(Spacer(1, 0.15*inch))
                                story.append(Paragraph("<b>Waste Type Distribution</b>", get_style('Heading3')))
                                waste_types = data['WasteType'].value_counts()
                                if not waste_types.empty:
                                    waste_data = [['Waste Type', 'Count', 'Percentage']]
                                    total_waste_records = waste_types.sum()
                                    for waste_type, count in waste_types.items():
                                        pct = (count / total_waste_records * 100) if total_waste_records > 0 else 0
                                        waste_data.append([str(waste_type), str(int(count)), f"{pct:.1f}%"])
                                    waste_table = Table(waste_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                                    waste_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5f7a')),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                                    ]))
                                    story.append(waste_table)
                            
                            story.append(Spacer(1, 0.2*inch))
            
            # AI Insights and Recommendations
            insights_list = ai_insights.get('insights', []) if isinstance(ai_insights.get('insights'), list) else []
            recommendations_list = ai_insights.get('recommendations', []) if isinstance(ai_insights.get('recommendations'), list) else []
            
            # Always show AI Insights section, even if empty (with fallback content)
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(f"<b>{safe_text(t['aiInsights'])}</b>", get_style('Heading2')))
            story.append(Spacer(1, 0.15*inch))
            
            if insights_list:
                for insight in insights_list:
                    # Remove emojis for PDF compatibility
                    clean_insight = insight.encode('ascii', 'ignore').decode('ascii') if isinstance(insight, str) else str(insight)
                    story.append(Paragraph(f"• {safe_text(clean_insight)}", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
            else:
                # Generate comprehensive fallback insights from data
                performance_score = kpis.get('performance_score', 0)
                story.append(Paragraph(f"• Overall Performance Score: {performance_score:.1f}/100", get_style('Normal')))
                story.append(Spacer(1, 0.08*inch))
                
                if kpis.get('total_production', 0) > 7:
                    story.append(Paragraph(f"• Production yield is above average at {kpis.get('total_production', 0):.2f} tonnes/ha", get_style('Normal')))
                elif kpis.get('total_production', 0) < 5:
                    story.append(Paragraph(f"• Production yield is below optimal at {kpis.get('total_production', 0):.2f} tonnes/ha - improvement needed", get_style('Normal')))
                else:
                    story.append(Paragraph(f"• Production yield is at {kpis.get('total_production', 0):.2f} tonnes/ha - within acceptable range", get_style('Normal')))
                story.append(Spacer(1, 0.08*inch))
                
                if kpis.get('storage_spoilage', 0) > 10:
                    story.append(Paragraph(f"• Storage spoilage rate of {kpis.get('storage_spoilage', 0):.2f}% requires attention", get_style('Normal')))
                else:
                    story.append(Paragraph(f"• Storage spoilage is well controlled at {kpis.get('storage_spoilage', 0):.2f}%", get_style('Normal')))
                story.append(Spacer(1, 0.08*inch))
                
                if kpis.get('processing_defects', 0) > 5:
                    story.append(Paragraph(f"• Processing defect rate of {kpis.get('processing_defects', 0):.2f}% indicates quality control improvements needed", get_style('Normal')))
                else:
                    story.append(Paragraph(f"• Processing quality is good with {kpis.get('processing_defects', 0):.2f}% defect rate", get_style('Normal')))
                story.append(Spacer(1, 0.08*inch))
                
                if kpis.get('transport_delays', 0) > 10:
                    story.append(Paragraph(f"• Transportation delays at {kpis.get('transport_delays', 0):.1f}% suggest logistics optimization opportunities", get_style('Normal')))
                else:
                    story.append(Paragraph(f"• Transportation efficiency is good with {kpis.get('transport_delays', 0):.1f}% delay rate", get_style('Normal')))
                story.append(Spacer(1, 0.08*inch))
                
                if kpis.get('carbon_footprint') and isinstance(kpis['carbon_footprint'], dict):
                    total_carbon = kpis['carbon_footprint'].get('total', 0)
                    story.append(Paragraph(f"• Total carbon footprint: {total_carbon:.2f} kg CO2e", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
            
            if recommendations_list:
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph(f"<b>{safe_text(t['recommendations'])}</b>", get_style('Heading3')))
                story.append(Spacer(1, 0.1*inch))
                for rec in recommendations_list:
                    clean_rec = rec.encode('ascii', 'ignore').decode('ascii') if isinstance(rec, str) else str(rec)
                    story.append(Paragraph(f"• {safe_text(clean_rec)}", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
            else:
                # Generate fallback recommendations
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph(f"<b>{safe_text(t['recommendations'])}</b>", get_style('Heading3')))
                story.append(Spacer(1, 0.1*inch))
                
                if kpis.get('storage_spoilage', 0) > 10:
                    story.append(Paragraph("• Optimize storage temperature and humidity controls to reduce spoilage", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
                if kpis.get('processing_defects', 0) > 5:
                    story.append(Paragraph("• Implement quality checkpoints and staff training to reduce defects", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
                if kpis.get('transport_delays', 0) > 10:
                    story.append(Paragraph("• Review delivery routes and carrier performance to improve on-time delivery", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
                if kpis.get('pest_risk', 0) > 50:
                    story.append(Paragraph("• Implement integrated pest management (IPM) strategies", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
                if kpis.get('machinery_uptime', 0) < 90:
                    story.append(Paragraph("• Schedule preventive maintenance to improve machinery uptime", get_style('Normal')))
                    story.append(Spacer(1, 0.08*inch))
            
            # Crop Recommendation
            if crop_rec:
                story.append(Spacer(1, 0.3*inch))
                crop_rec_title = {
                    'en': 'Crop Recommendation',
                    'hi': 'फसल सिफारिश',
                    'kn': 'ಬೆಳೆ ಶಿಫಾರಸು'
                }
                story.append(Paragraph(f"<b>{safe_text(crop_rec_title.get(lang, 'Crop Recommendation'))}</b>", get_style('Heading2')))
                story.append(Spacer(1, 0.15*inch))
                
                recommended_crop_label = {
                    'en': 'Recommended Crop',
                    'hi': 'अनुशंसित फसल',
                    'kn': 'ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ'
                }
                profitability_label = {
                    'en': 'Profitability Score',
                    'hi': 'लाभप्रदता स्कोर',
                    'kn': 'ಲಾಭದಾಯಕತೆ ಸ್ಕೋರ್'
                }
                price_label = {
                    'en': 'Predicted Price',
                    'hi': 'अनुमानित मूल्य',
                    'kn': 'ಭವಿಷ್ಯವಾಣಿ ಬೆಲೆ'
                }
                reasoning_label = {
                    'en': 'Reasoning',
                    'hi': 'तर्क',
                    'kn': 'ತಾರ್ಕಿಕತೆ'
                }
                
                crop_rec_data = [
                    [recommended_crop_label.get(lang, 'Recommended Crop'), crop_rec.get('recommended_crop', 'N/A')],
                    [profitability_label.get(lang, 'Profitability Score'), f"{crop_rec.get('recommendation_score', 0):.1f}/100"],
                    [price_label.get(lang, 'Predicted Price'), f"₹{crop_rec.get('predicted_price', 0):.2f}/quintal"]
                ]
                
                crop_rec_table = Table(crop_rec_data, colWidths=[3*inch, 2*inch])
                crop_rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(crop_rec_table)
                
                if crop_rec.get('reasoning'):
                    story.append(Spacer(1, 0.15*inch))
                    story.append(Paragraph(f"<b>{safe_text(reasoning_label.get(lang, 'Reasoning'))}:</b>", get_style('Heading3')))
                    story.append(Spacer(1, 0.1*inch))
                    for reason in crop_rec['reasoning']:
                        story.append(Paragraph(f"• {safe_text(reason)}", get_style('Normal')))
                        story.append(Spacer(1, 0.08*inch))
            
            # Add Summary Section at the end
            story.append(Spacer(1, 0.3*inch))
            summary_title = {
                'en': 'Executive Summary',
                'hi': 'कार्यकारी सारांश',
                'kn': 'ಕಾರ್ಯನಿರ್ವಾಹಕ ಸಾರಾಂಶ'
            }
            story.append(Paragraph(f"<b>{safe_text(summary_title.get(lang, 'Executive Summary'))}</b>", get_style('Heading2')))
            story.append(Spacer(1, 0.15*inch))
            
            summary_text_en = f"""
            This comprehensive report provides a detailed analysis of {farm_name}'s operations across all stages of the food supply chain. 
            The farm achieved an overall performance score of {kpis.get('performance_score', 0):.1f}/100, with key metrics including:
            • Production yield of {kpis.get('total_production', 0):.2f} tonnes per hectare
            • Storage spoilage rate of {kpis.get('storage_spoilage', 0):.2f}%
            • Processing defect rate of {kpis.get('processing_defects', 0):.2f}%
            • Transportation delay rate of {kpis.get('transport_delays', 0):.1f}%
            • Customer satisfaction score of {kpis.get('satisfaction', 0):.1f}/10
            """
            
            summary_text_hi = f"""
            यह व्यापक रिपोर्ट खाद्य आपूर्ति श्रृंखला के सभी चरणों में {farm_name} के संचालन का विस्तृत विश्लेषण प्रदान करती है।
            फार्म ने {kpis.get('performance_score', 0):.1f}/100 का समग्र प्रदर्शन स्कोर प्राप्त किया, जिसमें प्रमुख मेट्रिक्स शामिल हैं।
            """
            
            summary_text_kn = f"""
            ಈ ಸಮಗ್ರ ವರದಿಯು ಆಹಾರ ಸರಬರಾಜು ಸರಪಳಿಯ ಎಲ್ಲಾ ಹಂತಗಳಲ್ಲಿ {farm_name} ನ ಕಾರ್ಯಾಚರಣೆಗಳ ವಿವರವಾದ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಒದಗಿಸುತ್ತದೆ।
            ಫಾರ್ಮ್ {kpis.get('performance_score', 0):.1f}/100 ರ ಒಟ್ಟು ಪ್ರದರ್ಶನ ಸ್ಕೋರ್ ಸಾಧಿಸಿದೆ, ಪ್ರಮುಖ ಮೆಟ್ರಿಕ್ಸ್ಗಳೊಂದಿಗೆ।
            """
            
            summary_text = summary_text_en if lang == 'en' else (summary_text_hi if lang == 'hi' else summary_text_kn)
            story.append(Paragraph(safe_text(summary_text), get_style('Normal')))
            story.append(Spacer(1, 0.15*inch))
            
            # Add carbon footprint breakdown if available
            if kpis.get('carbon_footprint') and isinstance(kpis['carbon_footprint'], dict):
                carbon_title = {
                    'en': 'Carbon Footprint Breakdown',
                    'hi': 'कार्बन फुटप्रिंट विभाजन',
                    'kn': 'ಕಾರ್ಬನ್ ಫುಟ್ಪ್ರಿಂಟ್ ವಿಭಜನೆ'
                }
                story.append(Paragraph(f"<b>{safe_text(carbon_title.get(lang, 'Carbon Footprint Breakdown'))}</b>", get_style('Heading3')))
                story.append(Spacer(1, 0.1*inch))
                
                carbon_data = [
                    [make_table_cell('Total Carbon Footprint'), make_table_cell(f"{kpis['carbon_footprint'].get('total', 0):.2f} {t['kgCO2e']}")],
                    [make_table_cell('Fertilizer Emissions'), make_table_cell(f"{kpis['carbon_footprint'].get('fertilizer', 0):.2f} {t['kgCO2e']}")],
                    [make_table_cell('Transportation Emissions'), make_table_cell(f"{kpis['carbon_footprint'].get('transportation', 0):.2f} {t['kgCO2e']}")],
                    [make_table_cell('Storage Emissions'), make_table_cell(f"{kpis['carbon_footprint'].get('storage', 0):.2f} {t['kgCO2e']}")],
                    [make_table_cell('Processing Emissions'), make_table_cell(f"{kpis['carbon_footprint'].get('processing', 0):.2f} {t['kgCO2e']}")],
                    [make_table_cell('Waste Emissions'), make_table_cell(f"{kpis['carbon_footprint'].get('waste', 0):.2f} {t['kgCO2e']}")]
                ]
                
                carbon_table = Table(carbon_data, colWidths=[3*inch, 2*inch])
                carbon_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), table_font_normal),  # Apply Unicode font to ALL cells
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(carbon_table)
            
            # Build PDF - ensure story is not empty
            if not story:
                raise ValueError("PDF story is empty - no content to generate")
            
            doc.build(story)
            buffer.seek(0)
            
            # Get the PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            # Verify PDF was generated (PDF files start with %PDF)
            if not pdf_bytes.startswith(b'%PDF'):
                raise ValueError("Generated file is not a valid PDF")
            
            # Create response with proper headers
            from flask import make_response
            response = make_response(pdf_bytes)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename={farm_name}_Report_{lang}.pdf'
            response.headers['Content-Length'] = len(pdf_bytes)
            
            return response
            
        except Exception as build_error:
            if not buffer.closed:
                buffer.close()
            app.logger.error(f"PDF build error: {build_error}")
            import traceback
            traceback.print_exc()
            raise build_error
    
    except Exception as e:
        app.logger.error(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500


if __name__ == '__main__':
    # Initialize detection model on startup (optional - can be lazy loaded)
    # init_detection_model()
    app.run(debug=True, port=5003) 