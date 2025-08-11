# app.py
from __future__ import annotations

import os
import csv
import math
import pickle
from typing import Tuple, List

from flask import (
    Flask, request, jsonify, render_template, send_from_directory, abort
)

# ---------------- App / Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)

EXPECTED_FEATURES: List[str] = [
    "age", "sex", "specific_gravity", "albumin", "blood_pressure",
    "hypertension", "diabetes_mellitus", "coronary_artery_disease",
    "appetite", "pedal_edema", "anemia"
]

# ---------------- Load model/scaler (optional) ----------------
MODEL = None
SCALER = None
MODEL_LOADED = False

def try_load_model():
    """โหลดโมเดล/สเกลเลอร์ ถ้ามีไฟล์อยู่ข้าง app.py"""
    global MODEL, SCALER, MODEL_LOADED
    model_path = os.path.join(BASE_DIR, "stacked_ckd_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "stacked_ckd_scaler.pkl")
    MODEL = SCALER = None
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODEL = pickle.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                SCALER = pickle.load(f)
        MODEL_LOADED = MODEL is not None
        print(f"[Model] loaded={MODEL_LOADED} (model={bool(MODEL)}, scaler={bool(SCALER)})")
    except Exception as e:
        print(f"[Model] load failed: {e}")
        MODEL = SCALER = None
        MODEL_LOADED = False

try_load_model()

# ---------------- Helpers ----------------
def validate_payload(d: dict) -> Tuple[bool, str]:
    missing = [k for k in EXPECTED_FEATURES if k not in d]
    if missing:
        return False, f"missing fields: {missing}"

    # numeric checks
    for k in ("age", "specific_gravity", "albumin", "blood_pressure"):
        try:
            float(d[k])
        except Exception:
            return False, f"invalid numeric value for '{k}': {d[k]}"

    # categorias
    valid = {
        "sex": {"male", "female"},
        "hypertension": {"yes", "no", "0", "1"},
        "diabetes_mellitus": {"yes", "no", "0", "1"},
        "coronary_artery_disease": {"yes", "no", "0", "1"},
        "appetite": {"good", "poor", "0", "1"},
        "pedal_edema": {"yes", "no", "0", "1"},
        "anemia": {"yes", "no", "0", "1"},
    }
    for k, allowed in valid.items():
        v = str(d[k]).lower()
        if v not in allowed:
            return False, f"invalid value for '{k}': {d[k]}"
    return True, ""


def yn_to01(v):  # yes/no -> 1/0
    return 1 if str(v).lower() in ("yes", "1", "true") else 0

def gp_to01(v):  # appetite good/poor -> 0/1 (poor=1)
    return 1 if str(v).lower() in ("poor", "1", "true") else 0

def sex_to01(v):  # male/female -> 1/0 (กำหนด male=1)
    return 1 if str(v).lower() == "male" else 0

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def build_feature_vector(d: dict) -> List[float]:
    """แปลง payload -> เวคเตอร์ตัวเลขตามลำดับ EXPECTED_FEATURES"""
    return [
        float(d["age"]),
        sex_to01(d["sex"]),
        float(d["specific_gravity"]),
        float(d["albumin"]),
        float(d["blood_pressure"]),
        yn_to01(d["hypertension"]),
        yn_to01(d["diabetes_mellitus"]),
        yn_to01(d["coronary_artery_disease"]),
        gp_to01(d["appetite"]),
        yn_to01(d["pedal_edema"]),
        yn_to01(d["anemia"]),
    ]

def map_score_to_level(score01: float) -> int:
    """รับค่า 0..1 -> ระดับ 1..5"""
    if score01 < 0.20: return 1
    if score01 < 0.40: return 2
    if score01 < 0.60: return 3
    if score01 < 0.80: return 4
    return 5

# def rule_level(payload: dict) -> int:
#     """กฎง่ายๆ สำหรับ fallback กรณีไม่มีโมเดล"""
#     def scale01(v, lo, hi):
#         v = max(min(float(v), hi), lo)
#         return (v - lo) / (hi - lo) if hi > lo else 0.0

#     p = payload
#     score = 0.0
#     score += scale01(p["blood_pressure"], 80, 180) * 1.0
#     score += yn_to01(p["hypertension"]) * 1.0
#     score += yn_to01(p["diabetes_mellitus"]) * 1.2
#     score += yn_to01(p["coronary_artery_disease"]) * 1.0
#     score += yn_to01(p["pedal_edema"]) * 1.0
#     score += yn_to01(p["anemia"]) * 0.8
#     score += gp_to01(p["appetite"]) * 0.7
#     score += (float(p["albumin"]) / 5.0) * 1.5
#     score += (1.025 - float(p["specific_gravity"])) * 15.0
#     score += scale01(p["age"], 20, 85) * 0.8

#     MAX_SCORE = 9.3  # ค่าสูงสุดโดยประมาณจากน้ำหนักด้านบน
#     score01 = max(0.0, min(score / MAX_SCORE, 1.0))  # normalize หยาบ
#     print("Score from Rule", score01, flush=True)
#     return map_score_to_level(score01)

def rule_level(payload: dict) -> int:
    """กฎ fallback ปรับน้ำหนักตามฟีเจอร์ที่มีในสูตรและสอดคล้องกับภาพ Top-10"""
    def scale01(v, lo, hi):
        v = float(v)
        if hi <= lo:
            return 0.0
        v = max(min(v, hi), lo)
        return (v - lo) / (hi - lo)

    def inv_scale01(v, lo, hi):
        # สำหรับค่าที่ต่ำ = เสี่ยง
        return 1.0 - scale01(v, lo, hi)

    p = payload
    score = 0.0

    # ฟีเจอร์จากภาพ (ให้ความสำคัญสูงขึ้น)
    score += (float(p["albumin"]) / 5.0) * 1.6               # สูง = เสี่ยง
    score += inv_scale01(p["specific_gravity"], 1.005, 1.025) * 1.5
    score += yn_to01(p["diabetes_mellitus"]) * 1.2
    score += yn_to01(p["hypertension"]) * 1.0

    # ฟีเจอร์อื่นในสูตรเดิม (ให้ความสำคัญน้อยลง)
    score += scale01(p["blood_pressure"], 80, 180) * 0.5
    score += yn_to01(p["coronary_artery_disease"]) * 0.4
    score += yn_to01(p["pedal_edema"]) * 0.4
    score += yn_to01(p["anemia"]) * 0.3
    score += gp_to01(p["appetite"]) * 0.3
    score += scale01(p["age"], 20, 85) * 0.3

    MAX_SCORE = 1.6 + 1.5 + 1.2 + 1.0 + 0.5 + 0.4 + 0.4 + 0.3 + 0.3 + 0.3  # = 7.5
    score01 = max(0.0, min(score / MAX_SCORE, 1.0))
    print("Score from Rule (adjusted)", round(score01, 4), flush=True)
    return map_score_to_level(score01)

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/result", methods=["GET"])
def result_page():
    # result.html อ่าน query string เอง (level, message)
    return render_template("result.html")

@app.route("/<path:filename>", methods=["GET"])
def serve_template_assets(filename: str):
    """
    อนุญาตให้ index.html/result.html อ้างรูป/ไฟล์อื่นใน templates/ ได้ตรงๆ
    (สะดวกตามโครงที่ผู้ใช้ต้องการ)
    """
    target = os.path.join(TEMPLATES_DIR, filename)
    if os.path.isfile(target):
        return send_from_directory(TEMPLATES_DIR, filename)
    abort(404)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    print("data", data, flush=True)
    ok, msg = validate_payload(data)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 400

    if MODEL_LOADED:
        print("Data -> ML Model", data, flush=True)
        try:
            x = [build_feature_vector(data)]
            if SCALER is not None:
                # รองรับทั้ง list-of-list และ numpy array
                try:
                    x = SCALER.transform(x)
                except Exception:
                    import numpy as np
                    x = SCALER.transform(np.asarray(x, dtype=float))
            # ได้คะแนนความเสี่ยง 0..1
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(x)
                score01 = float(proba[0][1])  # สมมติ class 1=CKD
            elif hasattr(MODEL, "decision_function"):
                df = float(MODEL.decision_function(x)[0])
                score01 = sigmoid(df)
            else:
                pred = int(MODEL.predict(x)[0])
                score01 = 0.8 if pred == 1 else 0.2
            level = map_score_to_level(score01)
        except Exception as e:
            level = rule_level(data)
            return jsonify({
                "ok": True, "level": level,
                # "message": "fallback to rule-based due to model error",
                "model_error": str(e)
            }), 200
    else:
        print("Data -> Rule", data, flush=True)
        level = rule_level(data)

    messages = {
        1: "สุขภาพไตแข็งแรง แนะนำดูแลสุขภาพต่อเนื่องและตรวจประจำปี",
        2: "เฝ้าระวังเล็กน้อย ควบคุมความดัน/น้ำตาล ลดเค็ม และติดตามอาการ",
        3: "เริ่มมีความเสี่ยง ควรพบแพทย์เพื่อตรวจเพิ่มเติม (eGFR/Albuminuria)",
        4: "เสี่ยงสูง ควรรีบพบแพทย์ ตรวจทางห้องปฏิบัติการและควบคุมโรคร่วม",
        5: "เสี่ยงสูงมาก พบแพทย์โดยด่วนเพื่อวินิจฉัยและรักษา",
    }
    return jsonify({"ok": True, "level": level, "message": messages[level]}), 200


# ----- (ตัวเลือก) เช็ก dataset header -----
def dataset_has_expected_columns(csv_path: str) -> Tuple[bool, List[str]]:
    if not os.path.exists(csv_path):
        return False, []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
    headers = [h.strip() for h in headers]
    ok = all(c in headers for c in EXPECTED_FEATURES)
    return ok, headers

@app.route("/check-dataset", methods=["GET"])
def check_dataset():
    csv_path = os.path.join(BASE_DIR, "Final_CKD_Assigned_Sex.csv")
    ok, headers = dataset_has_expected_columns(csv_path)
    return jsonify({
        "file_found": os.path.exists(csv_path),
        "ok": ok,
        "expected": EXPECTED_FEATURES,
        "headers_in_file": headers
    })


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    # log เล็กน้อย
    csv_file = os.path.join(BASE_DIR, "Final_CKD_Assigned_Sex.csv")
    if os.path.exists(csv_file):
        ok, headers = dataset_has_expected_columns(csv_file)
        print(f"[Dataset] found: {csv_file}")
        print(f" - headers: {headers}")
        print(f" - has all expected: {ok}")
    else:
        print("[Dataset] Final_CKD_Assigned_Sex.csv not found (optional)")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
