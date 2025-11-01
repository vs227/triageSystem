# Test Cases for Low Priority (Normal/Non-Urgent) Triage

Use these patient inputs to test that the system correctly identifies **Low** priority cases:

## ✅ Test Case 1: Simple Headache (Low)
```
Age: 30
Gender: Female
Chief Complaint: headache
Heart Rate: 72
Blood Pressure: 120
Temperature: 37.0
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 2: Cough/Cold (Low)
```
Age: 25
Gender: Male
Chief Complaint: cough
Heart Rate: 75
Blood Pressure: 115
Temperature: 36.8
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 3: Minor Leg Pain (Low)
```
Age: 45
Gender: Male
Chief Complaint: leg pain
Heart Rate: 70
Blood Pressure: 125
Temperature: 37.1
Medical History: Diabetes
```
**Expected:** Low Priority

## ✅ Test Case 4: Routine Checkup (Low)
```
Age: 35
Gender: Female
Chief Complaint: routine checkup
Heart Rate: 65
Blood Pressure: 110
Temperature: 36.7
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 5: Minor Cut (Low)
```
Age: 28
Gender: Male
Chief Complaint: small cut on finger
Heart Rate: 68
Blood Pressure: 118
Temperature: 37.0
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 6: Muscle Strain (Low)
```
Age: 40
Gender: Female
Chief Complaint: muscle strain in back
Heart Rate: 72
Blood Pressure: 122
Temperature: 36.9
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 7: Skin Rash (Low)
```
Age: 20
Gender: Male
Chief Complaint: skin rash
Heart Rate: 70
Blood Pressure: 120
Temperature: 37.0
Medical History: None
```
**Expected:** Low Priority

## ✅ Test Case 8: Cold Symptoms (Low)
```
Age: 32
Gender: Female
Chief Complaint: cold symptoms
Heart Rate: 74
Blood Pressure: 115
Temperature: 37.2
Medical History: None
```
**Expected:** Low Priority

---

## Key Characteristics for Low Priority:
- ✅ Normal heart rate (60-100 bpm)
- ✅ Normal blood pressure (90-140 mmHg)
- ✅ Normal temperature (36.5-37.5°C)
- ✅ Minor, non-life-threatening complaints
- ✅ No critical symptoms (chest pain, shortness of breath, etc.)
- ✅ No abnormal vital signs

## What SHOULD be Critical:
- Chest pain, extreme chest pain
- Shortness of breath
- Heart rate < 40 or > 130
- Blood pressure < 70 or > 200
- Temperature < 33 or > 40.5°C

## What SHOULD be Moderate:
- Fever (temperature > 38.5°C)
- Abdominal pain
- Single abnormal vital (but not extreme)

---

**If all of these are still coming out as Critical, the logic needs adjustment!**

