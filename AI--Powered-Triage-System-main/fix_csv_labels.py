#!/usr/bin/env python3
"""Fix CSV labels - correct Critical labels for minor complaints to Low"""

import pandas as pd

# Load CSV
df = pd.read_csv('patient_data.csv')

print("Before correction:")
print(df['triage_level'].value_counts())
print()

# Minor complaints that should be Low even with borderline vitals
minor_complaints = ['headache', 'cough', 'leg', 'hand', 'finger', 'cold', 'rash', 'minor cut', 'small cut', 'routine checkup']

corrections = 0
for idx, row in df.iterrows():
    chief_complaint = str(row.get('chief_complaint', '')).lower()
    current_triage = str(row.get('triage_level', '')).strip()
    
    # If it's marked Critical but is a minor complaint, change to Low
    if current_triage == 'Critical':
        if any(minor in chief_complaint for minor in minor_complaints):
            heart_rate = row.get('heart_rate', 0)
            blood_pressure = row.get('blood_pressure', 0)
            temperature = row.get('temperature', 0)
            
            # Check if vitals are truly extreme (not just borderline)
            extreme_hr = heart_rate > 130 or heart_rate < 40
            extreme_bp = blood_pressure > 200 or blood_pressure < 70
            extreme_temp = temperature > 40.5 or temperature < 33
            
            # If minor complaint with non-extreme vitals, change to Low
            if not (extreme_hr or extreme_bp or extreme_temp):
                print(f"Row {idx+1}: Changing '{current_triage}' -> 'Low' for '{chief_complaint}' (HR: {heart_rate}, BP: {blood_pressure}, Temp: {temperature:.1f})")
                df.at[idx, 'triage_level'] = 'Low'
                corrections += 1

if corrections > 0:
    df.to_csv('patient_data.csv', index=False)
    print(f"\n✅ Corrected {corrections} rows from Critical to Low")
else:
    print("\n✅ No corrections needed")

print("\nAfter correction:")
print(df['triage_level'].value_counts())

