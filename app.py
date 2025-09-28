# app.py
import pickle
from pathlib import Path

from flask import Flask, render_template, request, jsonify, abort

app = Flask(__name__)

MODEL_FILE = Path("model.pkl")
if not MODEL_FILE.exists():
    raise FileNotFoundError("model.pkl not found. Run `python train_model.py` first to create it.")

with open(MODEL_FILE, "rb") as f:
    obj = pickle.load(f)
pipeline = obj["pipeline"]
FEATURE_NAMES = obj["feature_order"]

# Mapping for UI (Yes/No -> 1/0, Male/Female -> 1/0)
BINARY_MAP = {"Yes": 1, "No": 0}
SEX_MAP = {"Male": 1, "Female": 0}

# Input constraints for HTML validation (min, max, step). Adjust if you want narrower ranges.
INPUT_CONSTRAINTS = {
    "age": {"min": 1, "max": 120, "step": "0.1"},
    "anaemia": {},
    "creatinine_phosphokinase": {"min": 0, "max": 10000, "step": "1"},
    "diabetes": {},
    "ejection_fraction": {"min": 1, "max": 100, "step": "1"},
    "high_blood_pressure": {},
    "platelets": {"min": 10000, "max": 2000000, "step": "1"},
    "serum_creatinine": {"min": 0.01, "max": 20, "step": "0.01"},
    "serum_sodium": {"min": 80, "max": 200, "step": "1"},
    "sex": {},
    "smoking": {},
    "time": {"min": 0, "max": 1000, "step": "1"},
}

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        binary_options=list(BINARY_MAP.keys()),
        sex_options=list(SEX_MAP.keys()),
        constraints=INPUT_CONSTRAINTS
    )

def parse_form_values(form):
    values = []
    for feat in FEATURE_NAMES:
        if feat in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
            # binary dropdowns
            v = form.get(feat)
            if v is None:
                raise ValueError(f"Missing value for {feat}")
            mapped = BINARY_MAP.get(v)
            if mapped is None:
                raise ValueError(f"Invalid value for {feat}: {v}")
            values.append(float(mapped))
        elif feat == "sex":
            v = form.get("sex")
            if v is None:
                raise ValueError("Missing value for sex")
            mapped = SEX_MAP.get(v)
            if mapped is None:
                raise ValueError(f"Invalid value for sex: {v}")
            values.append(float(mapped))
        else:
            raw = form.get(feat)
            if raw is None or raw.strip() == "":
                raise ValueError(f"Missing value for {feat}")
            try:
                values.append(float(raw))
            except Exception:
                raise ValueError(f"Invalid numeric value for {feat}: {raw}")
    return values

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = parse_form_values(request.form)
        import numpy as np
        X = np.array(values).reshape(1, -1)
        pred = pipeline.predict(X)[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = float(pipeline.predict_proba(X)[0, 1])
        return render_template("result.html", prediction=int(pred), probability=proba)
    except Exception as e:
        return render_template("error.html", error=str(e)), 400

# JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts JSON like:
    {
      "age": 75,
      "anaemia": "Yes",
      "creatinine_phosphokinase": 582,
      ...
    }
    Binary fields accept "Yes"/"No" (or 1/0), sex accepts "Male"/"Female" (or 1/0).
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Expecting JSON body"}), 400
    # Normalize data: allow 1/0 or Yes/No for binaries
    form_like = {}
    for feat in FEATURE_NAMES:
        if feat in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
            v = data.get(feat)
            if v is None:
                return jsonify({"error": f"Missing field {feat}"}), 400
            if isinstance(v, (int, float)):
                form_like[feat] = "Yes" if int(v) == 1 else "No"
            else:
                if str(v).lower() in ["yes", "y", "true", "1"]:
                    form_like[feat] = "Yes"
                elif str(v).lower() in ["no", "n", "false", "0"]:
                    form_like[feat] = "No"
                else:
                    return jsonify({"error": f"Invalid value for {feat}: {v}"}), 400
        elif feat == "sex":
            v = data.get("sex")
            if v is None:
                return jsonify({"error": "Missing field sex"}), 400
            if isinstance(v, (int, float)):
                form_like["sex"] = "Male" if int(v) == 1 else "Female"
            else:
                if str(v).lower() in ["male", "m", "1"]:
                    form_like["sex"] = "Male"
                elif str(v).lower() in ["female", "f", "0"]:
                    form_like["sex"] = "Female"
                else:
                    return jsonify({"error": f"Invalid value for sex: {v}"}), 400
        else:
            # numeric expected
            v = data.get(feat)
            if v is None:
                return jsonify({"error": f"Missing field {feat}"}), 400
            try:
                float(v)
            except Exception:
                return jsonify({"error": f"Invalid numeric value for {feat}: {v}"}), 400
            form_like[feat] = str(v)
    # reuse parse_form_values
    try:
        values = parse_form_values(form_like)
        import numpy as np
        X = np.array(values).reshape(1, -1)
        pred = int(pipeline.predict(X)[0])
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = float(pipeline.predict_proba(X)[0, 1])
        return jsonify({"prediction": pred, "probability": proba}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error="Page not found."), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
