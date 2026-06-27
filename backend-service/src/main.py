from http.client import HTTPException
import json
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.classifier import TicketClassifier
from src.router import TicketRouter
from pathlib import Path

app = FastAPI(title="SLA Assignment Service UAT")

# Add CORS Middleware to allow requests from your Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to ["http://localhost:3456"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mendapatkan path folder 'backend-service' secara otomatis
BASE_DIR = Path(__file__).resolve().parent.parent

# Gabungkan dengan lokasi file yang sebenarnya
KB_JSON_PATH = BASE_DIR / "kb" / "kb_for_system.json" 
CONFIG_JSON_PATH = BASE_DIR / "config.json"

SVM_PATH = BASE_DIR / "models" / "pipeline_svm_initial_model.joblib"
INDOBERT_DIR = BASE_DIR / "models" / "indobert-finetuned"
# 1. Load Knowledge Base dari JSON
with open(KB_JSON_PATH, "r", encoding="utf-8") as file:
    kb_data = json.load(file)

# 2. Inisialisasi Service
classifier = TicketClassifier(SVM_PATH, INDOBERT_DIR)
router = TicketRouter(kb_data)

# Baca konfigurasi awal untuk router
if CONFIG_JSON_PATH.exists():
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as file:
        config_data = json.load(file)
        router.THRESHOLD = float(config_data.get("matching_score_threshold", 85.0))

class ComplaintRequest(BaseModel):
    text: str

@app.get("/api/v1/config")
async def get_config():
    if not CONFIG_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail="File konfigurasi tidak ditemukan.")
        
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

@app.post("/api/v1/assign_sla")
async def assign_sla(request: ComplaintRequest):
    t_start_whole = time.perf_counter()
    
    # Hot-reload Knowledge Base and Config so testing is always synced
    with open(KB_JSON_PATH, "r", encoding="utf-8") as file:
        router.kb_data = json.load(file)
        
    if not CONFIG_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail="File konfigurasi tidak ditemukan.")
        
    with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as file:
        config_data = json.load(file)
        
    try:
        indobert_conf_threshold = float(config_data["indobert_confidence_threshold"])
        svm_conf_threshold = float(config_data["svm_confidence_threshold"])
        match_threshold = float(config_data["matching_score_threshold"])
        router.THRESHOLD = match_threshold
    except KeyError as e:
        raise ValueError(f"Key {e} wajib ada di file konfigurasi.")
    
    aduan = request.text
    
    # Eksekusi Klasifikasi SVM
    t_start_svm = time.perf_counter()
    svm_pred_unit, svm_conf = classifier.predict_svm(aduan)
    t_end_svm = time.perf_counter()
    
    # Eksekusi Klasifikasi IndoBERT
    t_start_indobert = time.perf_counter()
    indobert_pred_unit, indobert_conf = classifier.predict_indobert(aduan)
    t_end_indobert = time.perf_counter()
    
    # Eksekusi Knowledge Base Routing
    t_start_routing = time.perf_counter()
    svm_result = router.assign_ticket_sla(aduan, svm_pred_unit)
    indobert_result = router.assign_ticket_sla(aduan, indobert_pred_unit)
    t_end_routing = time.perf_counter()
    
    t_end_whole = time.perf_counter()
    
    # Kalkulasi Latency dalam ms
    latency_ms = {
        "svm_prediction": round((t_end_svm - t_start_svm) * 1000, 2),
        "indobert_prediction": round((t_end_indobert - t_start_indobert) * 1000, 2),
        "routing": round((t_end_routing - t_start_routing) * 1000, 2),
        "total_request": round((t_end_whole - t_start_whole) * 1000, 2)
    }
    
    return {
        "text": aduan,
        "indobert_confidence_threshold": indobert_conf_threshold,
        "svm_confidence_threshold": svm_conf_threshold,
        "matching_score_threshold": match_threshold,
        "latency_ms": latency_ms,
        "svm_pipeline": {
            "predicted_unit": svm_pred_unit,
            "confidence": round(svm_conf, 3),
            "routing_and_sla": svm_result
        },
        "indobert_pipeline": {
            "predicted_unit": indobert_pred_unit,
            "confidence": round(indobert_conf, 3),
            "routing_and_sla": indobert_result
        }
    }