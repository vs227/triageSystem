from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import threading
import time
from datetime import datetime

# Load environment variables from .env file (if exists)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads .env file if it exists
except ImportError:
    pass  # python-dotenv not installed, skip

# Optional OpenAI API support
try:
    import openai
    OPENAI_AVAILABLE = True
    # Try to create client (works with both old and new API versions)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        try:
            # New API (v1.0+)
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            OPENAI_NEW_API = True
        except:
            OPENAI_NEW_API = False
            openai_client = None
    else:
        OPENAI_NEW_API = False
        openai_client = None
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_NEW_API = False
    openai_client = None
    OPENAI_API_KEY = None
    print("Note: OpenAI library not installed. Install with: pip install openai")

USE_OPENAI = OPENAI_AVAILABLE and OPENAI_API_KEY is not None
if USE_OPENAI:
    print(f"✅ OpenAI API enabled (Using {'new' if OPENAI_NEW_API else 'old'} API)")
else:
    print("ℹ️  OpenAI API not enabled. Set OPENAI_API_KEY environment variable to enable.")

app = Flask(__name__)
CORS(app)

# Global variables
triage_system = None
model = None
preprocessor = None
patient_queue = []
patient_history = []  # Store processed patients
waiting_times = {}
patient_counter = 0
processed_csv_ids = set()  # Track CSV IDs that have been added to active queue
lock = threading.Lock()

# Load patient data from CSV
def load_patient_data(file_path):
    """Load patient data from a CSV file."""
    return pd.read_csv(file_path)

# Save patient data to CSV
def save_patient_to_csv(patient_data, triage_level, csv_path):
    """Append a new patient to the CSV file. This is CRITICAL for ML model training."""
    try:
        # Create a new row with the same structure as the CSV
        new_row = {
            'age': patient_data['age'],
            'heart_rate': patient_data['heart_rate'],
            'blood_pressure': patient_data['blood_pressure'],
            'temperature': patient_data['temperature'],
            'gender': patient_data['gender'],
            'chief_complaint': patient_data['chief_complaint'],
            'medical_history': patient_data.get('medical_history', 'None'),
            'triage_level': triage_level  # This is the predicted triage level from ML model
        }
        
        # Read existing CSV
        try:
            existing_data = pd.read_csv(csv_path)
            if len(existing_data) > 0:
                print(f"CSV file exists with {len(existing_data)} existing patients")
        except FileNotFoundError:
            # Create new DataFrame if file doesn't exist
            existing_data = pd.DataFrame(columns=[
                'age', 'heart_rate', 'blood_pressure', 'temperature', 
                'gender', 'chief_complaint', 'medical_history', 'triage_level'
            ])
            print(f"Creating new CSV file: {csv_path}")
        
        # Append new row
        new_df = pd.DataFrame([new_row])
        updated_data = pd.concat([existing_data, new_df], ignore_index=True)
        
        # Save to CSV (this updates the training data file)
        updated_data.to_csv(csv_path, index=False)
        print(f"✅ Patient saved to CSV: {csv_path} (Total patients in CSV: {len(updated_data)})")
        return True
    except Exception as e:
        print(f"❌ Error saving patient to CSV: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

# Preprocess data: Impute, scale and encode
def preprocess_data(data):
    """Preprocess the data, handle missing values, encode categorical variables, etc."""
    # Ensure we have the required columns
    required_columns = ['age', 'gender', 'chief_complaint', 'heart_rate', 'blood_pressure', 'temperature', 'medical_history', 'triage_level']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    numeric_features = ['age', 'heart_rate', 'blood_pressure', 'temperature']
    categorical_features = ['gender', 'chief_complaint', 'medical_history']
    
    # Ensure numeric columns are actually numeric
    for col in numeric_features:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, drop='if_binary'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any extra columns
    )

    X = data.drop('triage_level', axis=1)
    y = data['triage_level'].astype(str).str.strip()  # Ensure triage_level is string and trimmed
    
    # Validate triage levels
    valid_triage_levels = ['Critical', 'Moderate', 'Low']
    invalid_levels = y[~y.isin(valid_triage_levels)]
    if len(invalid_levels) > 0:
        print(f"Warning: Found invalid triage levels: {invalid_levels.unique()}")
        print("Mapping to valid levels...")
        # Try to map common variations
        y = y.replace({
            'critical': 'Critical',
            'moderate': 'Moderate',
            'low': 'Low',
            '1': 'Critical',
            '2': 'Moderate',
            '3': 'Low'
        })
    
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Preprocessed data: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
    print(f"Triage level distribution:\n{y.value_counts()}")
    
    return X_processed, y, preprocessor

# Map triage level labels to numeric priority (for sorting queue)
def triage_label_to_priority(label):
    """Convert triage level label to numeric priority."""
    label_lower = str(label).lower()
    if 'critical' in label_lower:
        return 1
    elif 'moderate' in label_lower:
        return 2
    elif 'low' in label_lower:
        return 3
    else:
        return 3  # Default to low priority

# Use ML model to predict triage level
def predict_triage_with_model(patient_data):
    """Use the trained ML model to predict triage level."""
    global model, preprocessor
    
    if model is None or preprocessor is None:
        print("Warning: Model or preprocessor not initialized")
        return None, None, None
    
    try:
        # Ensure all required fields are present and valid
        required_fields = ['age', 'gender', 'chief_complaint', 'heart_rate', 'blood_pressure', 'temperature']
        for field in required_fields:
            if field not in patient_data:
                print(f"Warning: Missing field '{field}' in patient data")
                return None, None, None
        
        # Convert to DataFrame for preprocessing (order matters!)
        df = pd.DataFrame([patient_data])
        
        # Ensure correct column order and types
        numeric_fields = ['age', 'heart_rate', 'blood_pressure', 'temperature']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Preprocess (this will handle missing values and encoding)
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        
        # Get class names for probabilities
        class_names = model.classes_
        prob_dict = dict(zip(class_names, probabilities))
        
        # Get confidence (max probability)
        confidence = float(max(probabilities))
        
        # Map prediction to numeric priority
        priority = triage_label_to_priority(prediction)
        
        # Debug output
        print(f"ML Prediction: {prediction} (Priority: {priority}, Confidence: {confidence:.2%})")
        print(f"  Probabilities: {prob_dict}")
        
        return str(prediction), priority, confidence
    except Exception as e:
        import traceback
        print(f"Error predicting with ML model: {str(e)}")
        print(traceback.format_exc())
        return None, None, None

# Define a function to determine triage level based on symptoms and vitals (fallback)
def determine_triage_level_rule_based(patient_data):
    """Determine triage level based on patient data using rule-based approach."""
    high_priority_symptoms = ['chest pain', 'shortness of breath', 'severe bleeding', 'stroke symptoms', 'heart attack', 'extreme chest pain']
    medium_priority_symptoms = ['abdominal pain', 'head injury', 'moderate bleeding', 'fever', 'vomiting']
    
    # Low priority indicators (default to low for these)
    low_priority_indicators = ['headache', 'cough', 'leg pain', 'routine checkup', 'small cut', 'muscle strain', 
                               'skin rash', 'cold symptoms', 'minor', 'checkup', 'rash', 'cold']

    chief_complaint = str(patient_data.get('chief_complaint', '')).lower()
    heart_rate = patient_data.get('heart_rate', 0)
    blood_pressure = patient_data.get('blood_pressure', 0)
    temperature = patient_data.get('temperature', 0)
    
    # Count abnormal vitals
    abnormal_vitals = sum([
        heart_rate > 120 or heart_rate < 50,  # Abnormal but not extreme
        blood_pressure > 180 or blood_pressure < 90,  # Abnormal but not extreme
        temperature > 39.5 or temperature < 35  # Abnormal but not extreme
    ])
    
    # Check for extreme vitals (immediate critical)
    extreme_hr = heart_rate > 130 or heart_rate < 40
    extreme_bp = blood_pressure > 200 or blood_pressure < 70
    extreme_temp = temperature > 40.5 or temperature < 33
    
    # Check for high priority symptoms
    severe_symptom = any(symptom in chief_complaint for symptom in high_priority_symptoms)
    
    # Critical: Severe symptoms OR extreme vitals OR multiple abnormal vitals
    if severe_symptom:
        return 'Critical', 1
    elif extreme_hr or extreme_bp or extreme_temp:
        return 'Critical', 1
    elif abnormal_vitals >= 2:  # Two or more abnormal vitals
        return 'Critical', 1
    
    # Moderate: Medium symptoms OR single abnormal vital
    if any(symptom in chief_complaint for symptom in medium_priority_symptoms):
        return 'Moderate', 2
    elif abnormal_vitals == 1:  # Single abnormal vital
        return 'Moderate', 2
    elif temperature > 38.5 or temperature < 36.5:  # Moderate fever/hypothermia
        return 'Moderate', 2
    
    # Low priority: Default for minor complaints with normal vitals
    if any(indicator in chief_complaint for indicator in low_priority_indicators):
        return 'Low', 3
    elif heart_rate >= 60 and heart_rate <= 100 and blood_pressure >= 90 and blood_pressure <= 140 and temperature >= 36.5 and temperature <= 37.5:
        # All vitals normal = low priority
        return 'Low', 3
    
    # Default to low priority (conservative approach)
    return 'Low', 3

# OpenAI API triage prediction (optional)
def predict_triage_with_openai(patient_data):
    """Use OpenAI API for intelligent triage classification."""
    if not USE_OPENAI:
        return None, None, None
    
    try:
        prompt = f"""You are a medical triage expert. Classify the following patient case into one of three triage levels: Critical, Moderate, or Low.

Patient Information:
- Age: {patient_data.get('age', 'N/A')}
- Gender: {patient_data.get('gender', 'N/A')}
- Chief Complaint: {patient_data.get('chief_complaint', 'N/A')}
- Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm
- Blood Pressure: {patient_data.get('blood_pressure', 'N/A')} mmHg
- Temperature: {patient_data.get('temperature', 'N/A')}°C
- Medical History: {patient_data.get('medical_history', 'None')}

Triage Guidelines (STRICT - Default to Low unless clearly critical):
- Critical: ONLY life-threatening conditions OR extreme vital signs:
  * Severe symptoms: chest pain, extreme chest pain, shortness of breath, severe bleeding, stroke symptoms, heart attack
  * Extreme vitals: HR <40 or >130, BP <70 or >200, Temp <33 or >40.5°C
  * Multiple (2+) abnormal vitals together
  
- Moderate: Moderate symptoms OR single abnormal vital:
  * Moderate symptoms: fever, abdominal pain, head injury, moderate bleeding, vomiting
  * Single abnormal vital: HR 50-120 but outside normal, BP 90-180 but outside normal, Temp 35-39.5°C but outside normal
  
- Low: DEFAULT for most cases - Minor complaints with stable vitals:
  * Minor complaints: headache, cough, leg pain, routine checkup, minor cuts, muscle strain, skin rash, cold symptoms
  * Normal vitals: HR 60-100, BP 90-140, Temp 36.5-37.5°C
  * Non-urgent conditions

IMPORTANT: When in doubt, choose Low. Only mark Critical if there are clear life-threatening indicators.

Patient Information:
Age: {patient_data.get('age', 'N/A')}
Gender: {patient_data.get('gender', 'N/A')}
Chief Complaint: {patient_data.get('chief_complaint', 'N/A')}
Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm
Blood Pressure: {patient_data.get('blood_pressure', 'N/A')} mmHg
Temperature: {patient_data.get('temperature', 'N/A')}°C
Medical History: {patient_data.get('medical_history', 'None')}

Respond with ONLY one word: Critical, Moderate, or Low."""

        # Try new API first (v1.0+)
        if OPENAI_NEW_API and openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical triage expert. Always respond with only one word: Critical, Moderate, or Low."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            prediction = response.choices[0].message.content.strip()
        else:
            # Fallback to old API
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical triage expert. Always respond with only one word: Critical, Moderate, or Low."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            prediction = response.choices[0].message.content.strip()
        
        # Validate response
        valid_levels = ['Critical', 'Moderate', 'Low']
        if prediction not in valid_levels:
            # Try to normalize
            prediction_lower = prediction.lower()
            if 'critical' in prediction_lower:
                prediction = 'Critical'
            elif 'moderate' in prediction_lower:
                prediction = 'Moderate'
            else:
                prediction = 'Low'
        
        priority = triage_label_to_priority(prediction)
        confidence = 0.95  # High confidence for OpenAI
        
        print(f"🤖 OpenAI Prediction: {prediction} (Priority: {priority}, Confidence: {confidence:.2%})")
        return prediction, priority, confidence
        
    except Exception as e:
        import traceback
        print(f"Error with OpenAI API: {str(e)}")
        print(traceback.format_exc())
        return None, None, None

# Main function to determine triage level (uses rule-based > OpenAI > ML model)
def determine_triage_level(patient_data):
    """Determine triage level using rule-based (most conservative), then OpenAI, then ML model."""
    # First check with rule-based (most conservative - defaults to Low)
    rule_label, rule_priority = determine_triage_level_rule_based(patient_data)
    
    # If rule-based says Low, trust it (conservative approach)
    if rule_label == 'Low':
        print(f"Rule-Based Prediction: {rule_label} (Priority: {rule_priority}) - Conservative default")
        return rule_label, rule_priority, 0.9  # High confidence for conservative rule-based
    
    # If rule-based says Critical/Moderate, double-check with OpenAI or ML model
    # Try OpenAI first if available
    if USE_OPENAI:
        prediction_label, priority, confidence = predict_triage_with_openai(patient_data)
        if prediction_label is not None:
            # Only use OpenAI if it agrees with rule-based OR if rule-based was Critical
            if rule_label == 'Critical' or prediction_label == rule_label:
                return prediction_label, priority, confidence
            # If conflict, trust rule-based for Critical, OpenAI for Moderate/Low
            if rule_label == 'Critical' and prediction_label != 'Critical':
                print(f"⚠️ Conflict: Rule-based says {rule_label}, OpenAI says {prediction_label}. Using rule-based (Critical).")
                return rule_label, rule_priority, 0.8
    
    # Try ML model
    prediction_label, priority, confidence = predict_triage_with_model(patient_data)
    
    if prediction_label is not None:
        print(f"ML Model Prediction: {prediction_label} (Priority: {priority}, Confidence: {confidence:.2%})")
        # Only use ML if it agrees with rule-based OR if rule-based was Critical
        if rule_label == 'Critical' or prediction_label == rule_label:
            return prediction_label, priority, confidence
        # If conflict and rule-based says Low but ML says Critical, trust rule-based (conservative)
        if rule_label == 'Low' and prediction_label == 'Critical':
            print(f"⚠️ Conflict: Rule-based says {rule_label}, ML says {prediction_label}. Using rule-based (Low - conservative).")
            return rule_label, rule_priority, 0.8
    
    # Fallback to rule-based (most conservative)
    print(f"Using rule-based triage: {rule_label} (Priority: {rule_priority})")
    return rule_label, rule_priority, 1.0

# Train the Random Forest model
def train_triage_model(X_train, y_train):
    """Train a Random Forest model for triage classification."""
    # Use more trees and better parameters for better accuracy
    model = RandomForestClassifier(
        n_estimators=200,  # More trees for better accuracy
        max_depth=15,  # Limit depth to prevent overfitting
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    # Print feature importance for debugging
    if hasattr(model, 'feature_importances_'):
        print(f"\nModel trained with {len(model.feature_importances_)} features")
        print(f"Top 5 most important features:")
        if isinstance(X_train, (list, tuple)) or hasattr(X_train, 'shape'):
            # For sparse matrices, we can't easily show feature names, but we can show importance distribution
            importances = model.feature_importances_
            top_indices = importances.argsort()[-5:][::-1]
            for idx in top_indices:
                print(f"  Feature {idx}: {importances[idx]:.4f}")
    
    return model

# Initialize the system
def initialize_system():
    global model, preprocessor, triage_system, processed_csv_ids
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'patient_data.csv')
    processed_csv_path = os.path.join(script_dir, 'processed_from_training.csv')
    
    # Load processed CSV IDs from tracking file (if exists)
    if os.path.exists(processed_csv_path):
        try:
            processed_df = pd.read_csv(processed_csv_path)
            if 'csv_id' in processed_df.columns:
                processed_csv_ids = set(processed_df['csv_id'].tolist())
                print(f"✅ Loaded {len(processed_csv_ids)} processed CSV IDs from processed_from_training.csv")
            else:
                processed_csv_ids = set()
        except Exception as e:
            print(f"Warning: Could not load processed CSV IDs: {str(e)}")
            processed_csv_ids = set()
    else:
        processed_csv_ids = set()
        print("No processed_from_training.csv found - starting fresh")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Creating a basic model without training data.")
        return
    
    # Load the data
    data = load_patient_data(csv_path)
    
    # Validate and auto-correct triage levels (same logic as retrain)
    print("\nValidating triage levels in CSV data...")
    corrections_made = 0
    
    for idx, row in data.iterrows():
        chief_complaint = str(row.get('chief_complaint', '')).lower()
        heart_rate = pd.to_numeric(row.get('heart_rate', 0), errors='coerce')
        blood_pressure = pd.to_numeric(row.get('blood_pressure', 0), errors='coerce')
        temperature = pd.to_numeric(row.get('temperature', 0), errors='coerce')
        current_triage = str(row.get('triage_level', '')).strip()
        
        # More conservative logic - default to Low unless clearly critical
        correct_triage = None
        high_priority_symptoms = ['chest pain', 'shortness of breath', 'severe bleeding', 'stroke symptoms', 'heart attack', 'extreme chest pain']
        medium_priority_symptoms = ['abdominal pain', 'head injury', 'moderate bleeding', 'fever', 'vomiting']
        low_priority_indicators = ['headache', 'cough', 'leg pain', 'routine checkup', 'small cut', 'muscle strain', 
                                   'skin rash', 'cold symptoms', 'minor', 'checkup', 'rash', 'cold', 'leg', 'hand', 'finger']
        
        is_critical = False
        is_moderate = False
        
        chief_lower = chief_complaint.lower()
        severe_symptom = any(symptom in chief_lower for symptom in high_priority_symptoms)
        extreme_hr = heart_rate > 130 or heart_rate < 40
        extreme_bp = blood_pressure > 200 or blood_pressure < 70
        extreme_temp = temperature > 40.5 or temperature < 33
        
        abnormal_vitals = sum([
            heart_rate > 120 or heart_rate < 50,
            blood_pressure > 180 or blood_pressure < 90,
            temperature > 39.5 or temperature < 35
        ])
        
        # Check for low priority indicators first
        low_priority_complaint = any(indicator in chief_lower for indicator in low_priority_indicators)
        
        # Critical: ONLY if severe symptoms OR extreme vitals (not borderline)
        if severe_symptom:
            is_critical = True
        elif (extreme_hr or extreme_bp or extreme_temp) and not low_priority_complaint:
            # Extreme vitals but not for minor complaints
            is_critical = True
        elif abnormal_vitals >= 2 and not low_priority_complaint:
            # Multiple abnormal vitals but not for minor complaints
            is_critical = True
        
        if not is_critical:
            # Moderate: medium symptoms OR single abnormal vital (but not minor complaints with borderline vitals)
            if any(symptom in chief_lower for symptom in medium_priority_symptoms):
                is_moderate = True
            elif abnormal_vitals == 1 and not low_priority_complaint:
                # Single abnormal vital but not minor complaints
                is_moderate = True
            elif temperature > 38.5 or temperature < 36.5:
                if not low_priority_complaint:
                    is_moderate = True
            
            if is_moderate:
                correct_triage = 'Moderate'
            else:
                # Default to Low for minor complaints or normal vitals
                correct_triage = 'Low'
        else:
            correct_triage = 'Critical'
        
        if current_triage.lower() != correct_triage.lower():
            print(f"  Row {idx+1}: Correcting '{current_triage}' -> '{correct_triage}'")
            data.at[idx, 'triage_level'] = correct_triage
            corrections_made += 1
    
    if corrections_made > 0:
        print(f"✅ Corrected {corrections_made} triage level(s)")
        data.to_csv(csv_path, index=False)
    
    # Preprocess the data
    X, y, preprocessor = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_triage_model(X_train, y_train)
    
    print("✅ Model trained successfully!")

# Retrain the model with updated CSV data
def retrain_model():
    """Retrain the model with the latest CSV data."""
    global model, preprocessor
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'patient_data.csv')
    
    if not os.path.exists(csv_path):
        return {'error': 'CSV file not found', 'success': False}, False
    
    try:
        # Load the data
        data = load_patient_data(csv_path)
        
        if len(data) < 2:
            return {'error': 'Not enough data to train model (need at least 2 samples)', 'success': False}, False
        
        # Validate and auto-correct triage levels using rule-based logic
        print("\nValidating and correcting triage levels in CSV data...")
        corrections_made = 0
        
        for idx, row in data.iterrows():
            # Get current values
            chief_complaint = str(row.get('chief_complaint', '')).lower()
            heart_rate = pd.to_numeric(row.get('heart_rate', 0), errors='coerce')
            blood_pressure = pd.to_numeric(row.get('blood_pressure', 0), errors='coerce')
            temperature = pd.to_numeric(row.get('temperature', 0), errors='coerce')
            current_triage = str(row.get('triage_level', '')).strip()
            
            # Determine correct triage using more nuanced rule-based logic
            correct_triage = None
            high_priority_symptoms = ['chest pain', 'shortness of breath', 'severe bleeding', 'stroke symptoms', 'heart attack', 'extreme chest pain']
            medium_priority_symptoms = ['abdominal pain', 'head injury', 'moderate bleeding', 'fever', 'vomiting']
            
            # Conservative logic - default to Low for minor complaints
            is_critical = False
            is_moderate = False
            
            chief_lower = chief_complaint.lower()
            low_priority_indicators = ['headache', 'cough', 'leg pain', 'routine checkup', 'small cut', 'muscle strain', 
                                       'skin rash', 'cold symptoms', 'minor', 'checkup', 'rash', 'cold', 'leg', 'hand', 'finger']
            low_priority_complaint = any(indicator in chief_lower for indicator in low_priority_indicators)
            
            # Critical: Need severe symptoms OR extreme vitals (but not minor complaints)
            severe_symptom = any(symptom in chief_lower for symptom in high_priority_symptoms)
            extreme_hr = heart_rate > 130 or heart_rate < 40
            extreme_bp = blood_pressure > 200 or blood_pressure < 70
            extreme_temp = temperature > 40.5 or temperature < 33
            
            # Count abnormal vitals
            abnormal_vitals = sum([
                heart_rate > 120 or heart_rate < 50,
                blood_pressure > 180 or blood_pressure < 90,
                temperature > 39.5 or temperature < 35
            ])
            
            # Only mark as Critical if severe symptom OR (extreme vital AND not minor complaint)
            if severe_symptom:
                is_critical = True
            elif (extreme_hr or extreme_bp or extreme_temp) and not low_priority_complaint:
                is_critical = True
            elif abnormal_vitals >= 2 and not low_priority_complaint:
                is_critical = True
            
            # Moderate: Medium symptoms OR single abnormal vital (but not minor complaints)
            if not is_critical:
                if any(symptom in chief_lower for symptom in medium_priority_symptoms):
                    is_moderate = True
                elif abnormal_vitals == 1 and not low_priority_complaint:
                    is_moderate = True
                elif (temperature > 38.5 or temperature < 36.5) and not low_priority_complaint:
                    is_moderate = True
                
                if is_moderate:
                    correct_triage = 'Moderate'
                else:
                    # Default to Low for minor complaints or normal vitals
                    correct_triage = 'Low'
            else:
                correct_triage = 'Critical'
            
            # If current triage doesn't match correct triage, correct it
            if current_triage.lower() != correct_triage.lower():
                print(f"  Row {idx+1}: Correcting '{current_triage}' -> '{correct_triage}' (HR: {heart_rate}, BP: {blood_pressure}, Temp: {temperature:.1f}, Complaint: {chief_complaint[:30]})")
                data.at[idx, 'triage_level'] = correct_triage
                corrections_made += 1
        
        if corrections_made > 0:
            print(f"✅ Corrected {corrections_made} triage level(s) based on patient vitals and symptoms")
            # Save corrected data back to CSV
            data.to_csv(csv_path, index=False)
            print(f"✅ Saved corrected data to {csv_path}")
        else:
            print("✅ All triage levels are correct")
        
        # Preprocess the data
        X, y, new_preprocessor = preprocess_data(data)
        
        # Handle case where we have very few samples
        # If we have less than 5 samples, use all data for training and don't split
        use_all_data = len(data) < 5
        if use_all_data:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        new_model = train_triage_model(X_train, y_train)
        
        # Update global variables
        model = new_model
        preprocessor = new_preprocessor
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score, classification_report
        
        # Handle sparse matrices - use shape[0] instead of len()
        if use_all_data:
            # If we used all data for training, calculate accuracy on training set
            predictions = model.predict(X_train)
            accuracy = accuracy_score(y_train, predictions)
            test_samples = 0
            training_samples = X_train.shape[0]  # Use shape[0] for sparse matrices
        else:
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            test_samples = X_test.shape[0]  # Use shape[0] for sparse matrices
            training_samples = X_train.shape[0]  # Use shape[0] for sparse matrices
        
        # Print detailed classification report for debugging
        print("\n" + "="*60)
        print("MODEL RETRAINING COMPLETE")
        print("="*60)
        print(f"Training samples: {training_samples}")
        print(f"Test samples: {test_samples}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if not use_all_data:
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))
        else:
            print("\nTraining Set Classification Report:")
            print(classification_report(y_train, predictions))
        print("="*60 + "\n")
        
        return {
            'success': True,
            'message': 'Model retrained successfully',
            'training_samples': int(training_samples),
            'test_samples': int(test_samples),
            'accuracy': float(accuracy)
        }, True
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error retraining model: {error_msg}")
        print(traceback.format_exc())
        return {'error': error_msg, 'success': False}, False

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/patients', methods=['POST'])
def add_patient():
    global patient_queue, waiting_times, patient_counter
    
    try:
        if not request.json:
            print("Error: No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        patient_data = request.json
        print(f"Received patient data: {patient_data}")
        
        # Validate required fields
        required_fields = ['age', 'gender', 'chief_complaint', 'heart_rate', 'blood_pressure', 'temperature']
        missing_fields = [field for field in required_fields if field not in patient_data or patient_data[field] == '']
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate data types and values
        try:
            age = int(patient_data['age'])
            if age < 0 or age > 150:
                return jsonify({'error': 'Age must be between 0 and 150'}), 400
            
            heart_rate = int(patient_data['heart_rate'])
            if heart_rate < 30 or heart_rate > 250:
                return jsonify({'error': 'Heart rate must be between 30 and 250 bpm'}), 400
            
            blood_pressure = int(patient_data['blood_pressure'])
            if blood_pressure < 50 or blood_pressure > 250:
                return jsonify({'error': 'Blood pressure must be between 50 and 250 mmHg'}), 400
            
            temperature = float(patient_data['temperature'])
            if temperature < 30 or temperature > 45:
                return jsonify({'error': 'Temperature must be between 30 and 45°C'}), 400
            
            patient_data['age'] = age
            patient_data['heart_rate'] = heart_rate
            patient_data['blood_pressure'] = blood_pressure
            patient_data['temperature'] = temperature
            
            # Set default for medical_history if not provided
            if 'medical_history' not in patient_data or not patient_data['medical_history']:
                patient_data['medical_history'] = 'None'
                
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid data type: {str(e)}. Please check that age, heart rate, and blood pressure are numbers, and temperature is a decimal number.'}), 400
        
        # Determine triage level using ML model (compares with CSV training data)
        triage_label, triage_priority, confidence = determine_triage_level(patient_data)
        
        # Get CSV path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'patient_data.csv')
        
        # Save to CSV file (this will append the new patient to the training data)
        csv_saved = save_patient_to_csv(patient_data, triage_label, csv_path)
        
        with lock:
            patient_counter += 1
            patient_id = patient_counter
            
            patient_entry = {
                'id': patient_id,
                'patient_data': patient_data,
                'triage_level': triage_label,  # Store the label (e.g., "Critical", "Moderate", "Low")
                'triage_priority': triage_priority,  # Numeric priority for sorting (1, 2, 3)
                'confidence': confidence,  # Model confidence score
                'prediction_method': 'ML Model' if model is not None else 'Rule-Based',
                'saved_to_csv': csv_saved,  # Indicate if successfully saved to CSV
                'timestamp': datetime.now().isoformat()
            }
            
            patient_queue.append(patient_entry)
            waiting_times[patient_id] = 0
            
            # Sort queue by triage priority (1=Critical, 2=Moderate, 3=Low)
            # This ensures highest priority (lowest number) patients are at the front
            patient_queue.sort(key=lambda x: (x.get('triage_priority', 3), x.get('timestamp', '')))
            
            # Log queue position for debugging
            queue_position = next((i for i, p in enumerate(patient_queue) if p['id'] == patient_id), -1)
            print(f"  Queue position after sorting: {queue_position + 1} of {len(patient_queue)}")
        
        print(f"Successfully added patient with ID: {patient_id}")
        print(f"  Triage Level: {triage_label} (Priority: {triage_priority})")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Method: {patient_entry['prediction_method']}")
        print(f"  Saved to CSV: {csv_saved}")
        
        return jsonify({
            'success': True,
            'patient': patient_entry,
            'saved_to_csv': csv_saved
        }), 201
        
    except Exception as e:
        import traceback
        print(f"Error adding patient: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get all patients in the queue"""
    try:
        return jsonify({
            'patients': patient_queue,
            'total': len(patient_queue),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'patients': [],
            'total': 0
        }), 500

@app.route('/api/patients/<int:patient_id>', methods=['DELETE'])
def remove_patient(patient_id):
    global patient_queue, waiting_times, processed_csv_ids
    
    try:
        csv_id_to_restore = None
        patient_source = None
        
        with lock:
            # Find the patient before removing
            patient_to_remove = next((p for p in patient_queue if p.get('id') == patient_id), None)
            
            if patient_to_remove:
                # Check if this patient came from CSV import
                if patient_to_remove.get('source') == 'CSV Import' and 'csv_id' in patient_to_remove:
                    csv_id_to_restore = patient_to_remove['csv_id']
                    patient_source = 'CSV Import'
            
            # Remove from queue
            initial_count = len(patient_queue)
            patient_queue = [p for p in patient_queue if p.get('id') != patient_id]
            removed = initial_count > len(patient_queue)
            
            if patient_id in waiting_times:
                del waiting_times[patient_id]
            
            if not removed:
                return jsonify({'error': f'Patient with ID {patient_id} not found'}), 404
            
            # If this was a CSV import, restore it to CSV training data view
            if csv_id_to_restore is not None:
                if csv_id_to_restore in processed_csv_ids:
                    processed_csv_ids.remove(csv_id_to_restore)
                    print(f"✅ CSV patient #{csv_id_to_restore} restored to training data (removed from queue)")
        
        message = f'Patient {patient_id} removed successfully'
        if patient_source == 'CSV Import' and csv_id_to_restore:
            message += f'. CSV patient #{csv_id_to_restore} is now available in training data again'
        
        return jsonify({
            'success': True,
            'message': message,
            'restored_to_training': csv_id_to_restore is not None,
            'csv_id': csv_id_to_restore
        }), 200
    except Exception as e:
        import traceback
        print(f"Error removing patient: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients/next', methods=['GET'])
def get_next_patient():
    """Get the next patient in the queue and move to history"""
    global patient_queue, waiting_times, patient_history, processed_csv_ids
    
    if patient_queue:
        with lock:
            next_patient = patient_queue.pop(0)
            patient_id = next_patient['id']
            
            # Remove from waiting times
            if patient_id in waiting_times:
                del waiting_times[patient_id]
            
            # Add to history with processed timestamp
            next_patient['processed_at'] = datetime.now().isoformat()
            next_patient['status'] = 'Processed'
            patient_history.append(next_patient)
            
            # Keep only last 1000 patients in history
            if len(patient_history) > 1000:
                patient_history = patient_history[-1000:]
            
            # Note: We don't remove CSV ID from processed_csv_ids here
            # Once processed, CSV patients stay in history and don't return to training data
            # Only removal (not processing) restores them to training data
        
        print(f"✅ Patient {patient_id} processed and moved to history")
        if next_patient.get('source') == 'CSV Import':
            print(f"   CSV Import patient - will remain in history (not restored to training data)")
        
        return jsonify({
            'success': True,
            'patient': next_patient,
            'message': 'Patient processed and moved to history'
        }), 200
    else:
        return jsonify({'message': 'No patients in queue'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the triage queue"""
    try:
        if not patient_queue:
            return jsonify({
                'total_patients': 0,
                'triage_distribution': {},
                'waiting_times': {}
            })
        
        triage_distribution = {}
        for patient in patient_queue:
            level = patient.get('triage_level', 'Unknown')
            triage_distribution[level] = triage_distribution.get(level, 0) + 1
        
        # Calculate average waiting time
        if waiting_times:
            avg_waiting = sum(waiting_times.values()) / len(waiting_times)
        else:
            avg_waiting = 0
        
        return jsonify({
            'total_patients': len(patient_queue),
            'triage_distribution': triage_distribution,
            'average_waiting_time': avg_waiting,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'total_patients': 0,
            'triage_distribution': {}
        }), 500

@app.route('/api/csv/patients', methods=['GET'])
def get_csv_patients():
    """Get all patients from the CSV training data (excluding those already added to queue)"""
    global processed_csv_ids
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'patient_data.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({
                'error': 'CSV file not found', 
                'patients': [],
                'total': 0
            }), 404
        
        # Load CSV data
        data = load_patient_data(csv_path)
        
        # Check if CSV is empty
        if len(data) == 0:
            return jsonify({
                'error': 'CSV file is empty', 
                'patients': [],
                'total': 0
            }), 200
        
        # Convert to list of dictionaries, EXCLUDING patients already added to queue
        csv_patients = []
        for idx, row in data.iterrows():
            csv_id = idx + 1
            
            # Skip if this CSV ID has already been added to the queue
            if csv_id in processed_csv_ids:
                continue  # This patient is now in active queue, don't show in training data
            
            patient = {
                'csv_id': csv_id,
                'patient_data': {
                    'age': int(row['age']),
                    'gender': str(row['gender']),
                    'chief_complaint': str(row['chief_complaint']),
                    'heart_rate': int(row['heart_rate']),
                    'blood_pressure': int(row['blood_pressure']),
                    'temperature': float(row['temperature']),
                    'medical_history': str(row['medical_history'])
                },
                'triage_level': str(row['triage_level']),
                'triage_priority': triage_label_to_priority(row['triage_level']),
                'source': 'CSV Training Data',
                'is_training_data': True
            }
            csv_patients.append(patient)
        
        print(f"CSV Training Data: {len(csv_patients)} available patients (excluded {len(processed_csv_ids)} already in queue)")
        
        return jsonify({
            'patients': csv_patients,
            'total': len(csv_patients),
            'excluded_count': len(processed_csv_ids)
        })
        
    except Exception as e:
        import traceback
        print(f"Error loading CSV patients: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'patients': []}), 500

@app.route('/api/csv/patients/<int:csv_id>/add-to-queue', methods=['POST'])
def add_csv_patient_to_queue(csv_id):
    """Add a CSV patient to the current queue and mark it as processed"""
    global patient_queue, waiting_times, patient_counter, processed_csv_ids
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'patient_data.csv')
        processed_csv_path = os.path.join(script_dir, 'processed_from_training.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found'}), 404
        
        # Load CSV data
        data = load_patient_data(csv_path)
        
        if csv_id < 1 or csv_id > len(data):
            return jsonify({'error': f'Invalid CSV patient ID. Valid range: 1-{len(data)}'}), 400
        
        # Check if already processed
        if csv_id in processed_csv_ids:
            return jsonify({'error': f'CSV patient #{csv_id} has already been added to the queue'}), 400
        
        # Get the specific patient
        row = data.iloc[csv_id - 1]
        patient_data = {
            'age': int(row['age']),
            'gender': str(row['gender']),
            'chief_complaint': str(row['chief_complaint']),
            'heart_rate': int(row['heart_rate']),
            'blood_pressure': int(row['blood_pressure']),
            'temperature': float(row['temperature']),
            'medical_history': str(row['medical_history'])
        }
        
        # Determine triage level using ML model
        triage_label, triage_priority, confidence = determine_triage_level(patient_data)
        
        with lock:
            patient_counter += 1
            patient_id = patient_counter
            
            patient_entry = {
                'id': patient_id,
                'patient_data': patient_data,
                'triage_level': triage_label,
                'triage_priority': triage_priority,
                'confidence': confidence,
                'prediction_method': 'ML Model' if model is not None else 'Rule-Based',
                'source': 'CSV Import',
                'csv_id': csv_id,
                'original_triage': str(row.get('triage_level', 'Unknown')),  # Store original from CSV
                'timestamp': datetime.now().isoformat()
            }
            
            patient_queue.append(patient_entry)
            waiting_times[patient_id] = 0
            # Sort queue by triage priority (Critical patients first)
            patient_queue.sort(key=lambda x: (x.get('triage_priority', 3), x.get('timestamp', '')))
            
            # Mark this CSV ID as processed (removed from training data view)
            processed_csv_ids.add(csv_id)
        
        # Save to processed CSV file for tracking
        try:
            processed_row = {
                'csv_id': csv_id,
                'age': patient_data['age'],
                'gender': patient_data['gender'],
                'chief_complaint': patient_data['chief_complaint'],
                'heart_rate': patient_data['heart_rate'],
                'blood_pressure': patient_data['blood_pressure'],
                'temperature': patient_data['temperature'],
                'medical_history': patient_data['medical_history'],
                'original_triage': str(row.get('triage_level', 'Unknown')),
                'predicted_triage': triage_label,
                'confidence': confidence,
                'added_to_queue_at': datetime.now().isoformat(),
                'queue_patient_id': patient_id
            }
            
            if os.path.exists(processed_csv_path):
                processed_df = pd.read_csv(processed_csv_path)
            else:
                processed_df = pd.DataFrame(columns=list(processed_row.keys()))
            
            new_processed_df = pd.DataFrame([processed_row])
            updated_processed = pd.concat([processed_df, new_processed_df], ignore_index=True)
            updated_processed.to_csv(processed_csv_path, index=False)
            print(f"✅ CSV patient #{csv_id} saved to processed_from_training.csv")
        except Exception as e:
            print(f"Warning: Could not save to processed CSV: {str(e)}")
        
        print(f"✅ CSV patient #{csv_id} added to queue (Patient ID: {patient_id})")
        print(f"   This patient will no longer appear in CSV Training Data view")
        
        return jsonify({
            'success': True,
            'patient': patient_entry,
            'message': f'CSV patient #{csv_id} moved from training data to active queue'
        }), 201
        
    except Exception as e:
        import traceback
        print(f"Error adding CSV patient to queue: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/predict', methods=['POST'])
def predict_triage():
    """Use the ML model to predict triage level"""
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        patient_data = request.json
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0]
        
        return jsonify({
            'prediction': str(prediction),
            'probabilities': {
                str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model_endpoint():
    """Retrain the ML model with latest CSV data"""
    result, success = retrain_model()
    if success:
        return jsonify(result), 200
    else:
        return jsonify(result), 400

@app.route('/api/history', methods=['GET'])
def get_patient_history():
    """Get patient history (processed patients)"""
    global patient_history
    
    # Get query parameters for filtering
    limit = request.args.get('limit', type=int, default=100)
    triage_level = request.args.get('triage_level', type=str)
    start_date = request.args.get('start_date', type=str)
    end_date = request.args.get('end_date', type=str)
    
    filtered_history = patient_history.copy()
    
    # Filter by triage level
    if triage_level:
        filtered_history = [p for p in filtered_history if str(p.get('triage_level', '')).lower() == triage_level.lower()]
    
    # Filter by date range
    if start_date or end_date:
        filtered_history = [p for p in filtered_history if p.get('processed_at')]
        if start_date:
            filtered_history = [p for p in filtered_history if p.get('processed_at', '') >= start_date]
        if end_date:
            filtered_history = [p for p in filtered_history if p.get('processed_at', '') <= end_date]
    
    # Sort by processed_at (newest first) and limit
    filtered_history.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
    filtered_history = filtered_history[:limit]
    
    return jsonify({
        'patients': filtered_history,
        'total': len(filtered_history),
        'total_history': len(patient_history)
    })

@app.route('/api/patients/search', methods=['GET'])
def search_patients():
    """Search patients in queue"""
    global patient_queue
    
    try:
        query = request.args.get('q', '').strip()
        triage_filter = request.args.get('triage_level', type=str)
        min_age = request.args.get('min_age', type=int)
        max_age = request.args.get('max_age', type=int)
        
        # Validate that at least one filter is provided
        if not query and not triage_filter and min_age is None and max_age is None:
            return jsonify({
                'error': 'At least one search criteria must be provided',
                'patients': [],
                'total': 0
            }), 400
        
        filtered = patient_queue.copy()
        
        # Text search (only if query is not empty)
        if query:
            query_lower = query.lower()
            filtered = [p for p in filtered if 
                       query_lower in str(p['patient_data'].get('chief_complaint', '')).lower() or
                       query_lower in str(p['patient_data'].get('gender', '')).lower() or
                       query_lower in str(p['patient_data'].get('medical_history', '')).lower()]
        
        # Filter by triage level
        if triage_filter:
            triage_lower = triage_filter.lower()
            filtered = [p for p in filtered if str(p.get('triage_level', '')).lower() == triage_lower]
        
        # Filter by age range
        if min_age is not None and min_age >= 0:
            filtered = [p for p in filtered if p['patient_data'].get('age', 0) >= min_age]
        if max_age is not None and max_age >= 0:
            filtered = [p for p in filtered if p['patient_data'].get('age', 0) <= max_age]
        
        return jsonify({
            'patients': filtered,
            'total': len(filtered),
            'filters_applied': {
                'query': bool(query),
                'triage_level': bool(triage_filter),
                'age_range': min_age is not None or max_age is not None
            }
        })
        
    except Exception as e:
        import traceback
        print(f"Error in search: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'patients': [],
            'total': 0
        }), 500

@app.route('/api/export/queue', methods=['GET'])
def export_queue():
    """Export current queue to CSV format"""
    from flask import Response
    
    import io
    output = io.StringIO()
    
    # Create CSV
    output.write('id,age,gender,chief_complaint,heart_rate,blood_pressure,temperature,medical_history,triage_level,confidence,prediction_method,timestamp\n')
    
    for patient in patient_queue:
        pd = patient['patient_data']
        output.write(f"{patient['id']},{pd.get('age', '')},{pd.get('gender', '')},{pd.get('chief_complaint', '')},")
        output.write(f"{pd.get('heart_rate', '')},{pd.get('blood_pressure', '')},{pd.get('temperature', '')},")
        output.write(f"{pd.get('medical_history', '')},{patient.get('triage_level', '')},")
        output.write(f"{patient.get('confidence', 0)},{patient.get('prediction_method', '')},{patient.get('timestamp', '')}\n")
    
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=triage_queue_export.csv'}
    )

@app.route('/api/export/history', methods=['GET'])
def export_history():
    """Export patient history to CSV format"""
    from flask import Response
    
    import io
    output = io.StringIO()
    
    # Create CSV
    output.write('id,age,gender,chief_complaint,heart_rate,blood_pressure,temperature,medical_history,triage_level,confidence,prediction_method,processed_at\n')
    
    for patient in patient_history:
        pd = patient['patient_data']
        output.write(f"{patient['id']},{pd.get('age', '')},{pd.get('gender', '')},{pd.get('chief_complaint', '')},")
        output.write(f"{pd.get('heart_rate', '')},{pd.get('blood_pressure', '')},{pd.get('temperature', '')},")
        output.write(f"{pd.get('medical_history', '')},{patient.get('triage_level', '')},")
        output.write(f"{patient.get('confidence', 0)},{patient.get('prediction_method', '')},{patient.get('processed_at', '')}\n")
    
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=triage_history_export.csv'}
    )

if __name__ == '__main__':
    print("Initializing AI-Powered Triage System...")
    initialize_system()
    print("Starting Flask server on http://localhost:5000")
    print("API endpoints available:")
    print("  POST /api/patients - Add a patient")
    print("  GET /api/patients - Get all patients")
    print("  GET /api/patients/search - Search/filter patients")
    print("  GET /api/patients/next - Process next patient")
    print("  GET /api/stats - Get statistics")
    print("  GET /api/history - Get patient history")
    print("  POST /api/model/retrain - Retrain ML model")
    print("  GET /api/export/queue - Export queue to CSV")
    print("  GET /api/export/history - Export history to CSV")
    print("  GET /api/csv/patients - Get CSV training data")
    app.run(debug=True, port=5000, host='0.0.0.0')
