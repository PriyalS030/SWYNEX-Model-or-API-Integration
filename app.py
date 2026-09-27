"""
app.py
======
Flask web application for Student Performance Risk Prediction.

Loads the trained ML model and provides a web interface for predictions.
Designed with a pluggable API integration layer for future enhancements.
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env (if it exists)
load_dotenv()

# Import the AI API service (placeholder for future integration)
from services.ai_api import get_ai_explanation

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load the trained model
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join('model', 'model.pkl')

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_list = model_data['features']
    model_name = model_data['model_name']
    binary_columns = model_data['binary_columns']
    print(f"✓ Model loaded: {model_name}")
    print(f"  Features: {feature_list}")
except FileNotFoundError:
    print("✗ Model file not found. Run 'python train_model.py' first.")
    model = None
    scaler = None
    feature_list = []
    model_name = "Not loaded"
    binary_columns = []

# ─────────────────────────────────────────────
# Feature metadata for form rendering and validation
# ─────────────────────────────────────────────
FEATURE_META = {
    'studytime': {
        'label': 'Weekly Study Time',
        'type': 'select',
        'options': [
            ('1', '1 — Less than 2 hours'),
            ('2', '2 — 2 to 5 hours'),
            ('3', '3 — 5 to 10 hours'),
            ('4', '4 — More than 10 hours')
        ],
        'default': '2'
    },
    'failures': {
        'label': 'Past Class Failures',
        'type': 'select',
        'options': [
            ('0', '0'),
            ('1', '1'),
            ('2', '2'),
            ('3', '3 or more')
        ],
        'default': '0'
    },
    'absences': {
        'label': 'Number of School Absences',
        'type': 'number',
        'min': 0,
        'max': 93,
        'default': '4'
    },
    'schoolsup': {
        'label': 'Extra Educational Support',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '0'
    },
    'famsup': {
        'label': 'Family Educational Support',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '1'
    },
    'paid': {
        'label': 'Extra Paid Classes',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '0'
    },
    'activities': {
        'label': 'Extracurricular Activities',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '1'
    },
    'higher': {
        'label': 'Wants Higher Education',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '1'
    },
    'internet': {
        'label': 'Internet Access at Home',
        'type': 'select',
        'options': [('1', 'Yes'), ('0', 'No')],
        'default': '1'
    },
    'freetime': {
        'label': 'Free Time After School',
        'type': 'select',
        'options': [
            ('1', '1 — Very Low'),
            ('2', '2 — Low'),
            ('3', '3 — Medium'),
            ('4', '4 — High'),
            ('5', '5 — Very High')
        ],
        'default': '3'
    },
    'goout': {
        'label': 'Going Out with Friends',
        'type': 'select',
        'options': [
            ('1', '1 — Very Low'),
            ('2', '2 — Low'),
            ('3', '3 — Medium'),
            ('4', '4 — High'),
            ('5', '5 — Very High')
        ],
        'default': '3'
    },
    'health': {
        'label': 'Current Health Status',
        'type': 'select',
        'options': [
            ('1', '1 — Very Bad'),
            ('2', '2 — Bad'),
            ('3', '3 — Average'),
            ('4', '4 — Good'),
            ('5', '5 — Very Good')
        ],
        'default': '3'
    },
    'G1': {
        'label': 'First Period Grade (G1)',
        'type': 'number',
        'min': 0,
        'max': 20,
        'default': '10'
    },
    'G2': {
        'label': 'Second Period Grade (G2)',
        'type': 'number',
        'min': 0,
        'max': 20,
        'default': '10'
    }
}


@app.route('/', methods=['GET'])
def index():
    """Render the prediction form."""
    # Build ordered feature metadata for the template
    features_for_form = []
    for feat in feature_list:
        if feat in FEATURE_META:
            meta = FEATURE_META[feat].copy()
            meta['name'] = feat
            features_for_form.append(meta)

    return render_template(
        'index.html',
        features=features_for_form,
        model_name=model_name,
        prediction=None,
        ai_explanation=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    if model is None:
        return render_template(
            'index.html',
            features=[],
            model_name="Not loaded",
            prediction=None,
            error="Model not loaded. Run 'python train_model.py' first.",
            ai_explanation=None
        )

    try:
        # Extract and validate input values
        input_values = []
        user_inputs = {}

        for feat in feature_list:
            raw_value = request.form.get(feat, '')
            if raw_value == '' or raw_value is None:
                raise ValueError(f"Missing value for '{FEATURE_META.get(feat, {}).get('label', feat)}'")

            value = float(raw_value)
            meta = FEATURE_META.get(feat, {})

            # Validate ranges for number inputs
            if meta.get('type') == 'number':
                min_val = meta.get('min', 0)
                max_val = meta.get('max', 100)
                if value < min_val or value > max_val:
                    raise ValueError(
                        f"'{meta.get('label', feat)}' must be between {min_val} and {max_val}"
                    )

            input_values.append(value)
            user_inputs[feat] = raw_value

        # Make prediction
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        risk_label = "At Risk" if prediction == 1 else "Low Risk"

        # Attempt to get AI explanation (returns None if not configured)
        ai_explanation = get_ai_explanation(
            features=user_inputs,
            prediction=risk_label,
            model_name=model_name
        )

        # Build features for form (preserve user inputs)
        features_for_form = []
        for feat in feature_list:
            if feat in FEATURE_META:
                meta = FEATURE_META[feat].copy()
                meta['name'] = feat
                meta['value'] = user_inputs.get(feat, meta.get('default', ''))
                features_for_form.append(meta)

        return render_template(
            'index.html',
            features=features_for_form,
            model_name=model_name,
            prediction=risk_label,
            ai_explanation=ai_explanation
        )

    except ValueError as e:
        # Rebuild form with current values
        features_for_form = []
        for feat in feature_list:
            if feat in FEATURE_META:
                meta = FEATURE_META[feat].copy()
                meta['name'] = feat
                meta['value'] = request.form.get(feat, meta.get('default', ''))
                features_for_form.append(meta)

        return render_template(
            'index.html',
            features=features_for_form,
            model_name=model_name,
            prediction=None,
            error=str(e),
            ai_explanation=None
        )

    except Exception as e:
        return render_template(
            'index.html',
            features=[],
            model_name=model_name,
            prediction=None,
            error=f"An unexpected error occurred: {str(e)}",
            ai_explanation=None
        )


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Student Performance Risk Prediction")
    print(f"Model: {model_name}")
    print("Starting Flask server...")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
