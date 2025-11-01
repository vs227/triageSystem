#!/usr/bin/env python3
"""Add more Low priority samples to CSV to balance training data"""

import pandas as pd

# Load CSV
df = pd.read_csv('patient_data.csv')

# Add more Low priority examples
low_priority_samples = [
    {'age': 30, 'heart_rate': 72, 'blood_pressure': 120, 'temperature': 37.0, 'gender': 'Female', 'chief_complaint': 'headache', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 25, 'heart_rate': 75, 'blood_pressure': 115, 'temperature': 36.8, 'gender': 'Male', 'chief_complaint': 'cough', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 45, 'heart_rate': 70, 'blood_pressure': 125, 'temperature': 37.1, 'gender': 'Male', 'chief_complaint': 'leg pain', 'medical_history': 'Diabetes', 'triage_level': 'Low'},
    {'age': 35, 'heart_rate': 65, 'blood_pressure': 110, 'temperature': 36.7, 'gender': 'Female', 'chief_complaint': 'routine checkup', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 28, 'heart_rate': 68, 'blood_pressure': 118, 'temperature': 37.0, 'gender': 'Male', 'chief_complaint': 'cold symptoms', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 20, 'heart_rate': 70, 'blood_pressure': 120, 'temperature': 37.0, 'gender': 'Female', 'chief_complaint': 'small cut on finger', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 40, 'heart_rate': 72, 'blood_pressure': 122, 'temperature': 36.9, 'gender': 'Female', 'chief_complaint': 'muscle strain in back', 'medical_history': 'None', 'triage_level': 'Low'},
    {'age': 32, 'heart_rate': 74, 'blood_pressure': 115, 'temperature': 37.2, 'gender': 'Female', 'chief_complaint': 'skin rash', 'medical_history': 'None', 'triage_level': 'Low'},
]

# Convert to DataFrame
new_samples = pd.DataFrame(low_priority_samples)

# Combine
df_updated = pd.concat([df, new_samples], ignore_index=True)

# Save
df_updated.to_csv('patient_data.csv', index=False)

print(f"✅ Added {len(low_priority_samples)} Low priority samples")
print(f"New distribution:")
print(df_updated['triage_level'].value_counts())

