# AI-Powered Triage System for Emergency Rooms

This project implements an AI-powered triage system designed to streamline the triage process in emergency rooms by prioritizing patients based on the severity of their condition using machine learning techniques.

## Features
- *Patient Data Management*: Load, preprocess, and manage patient data from a CSV file.
- *Triage Classification*: The system determines a patient's triage level based on their symptoms and vitals.
- *Machine Learning*: Utilizes a Random Forest Classifier to predict triage levels based on historical data.
- *Real-Time Data Integration*: Simulates real-time patient data and integrates it into the triage queue.
- *Modern Web Interface*: Built with React for a beautiful, responsive UI that displays patient queue, statistics, and visualizations.
- *RESTful API*: Flask backend API that exposes all ML functionality via REST endpoints.
- *Data Security*: Incorporates encryption and decryption of sensitive patient data using cryptography's Fernet module.

## Architecture

The system consists of two main components:

1. **Backend (Flask API)**: Python-based REST API that handles:
   - Patient data processing
   - Machine learning model training and prediction
   - Triage queue management
   - Real-time statistics

2. **Frontend (React)**: Modern web interface that provides:
   - Patient data entry form
   - Real-time patient queue display
   - Triage level statistics and visualizations
   - Interactive charts using Recharts

## Prerequisites

### Python Dependencies
- Python 3.7 or higher
- pip (Python package manager)

### Node.js Dependencies
- Node.js 14.0 or higher
- npm (Node package manager)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- pandas
- numpy
- scikit-learn
- flask
- flask-cors
- matplotlib
- seaborn
- cryptography

### 2. Install Node.js Dependencies

Navigate to the project directory and run:

```bash
npm install
```

This will install:
- react
- react-dom
- react-scripts
- axios
- recharts

## Running the System

### Step 1: Start the Flask Backend

Open a terminal and run:

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`

### Step 2: Start the React Frontend

Open a new terminal and run:

```bash
npm start
```

The React app will automatically open in your browser at `http://localhost:3000`

## System Overview

### Backend API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/patients` - Add a new patient to the queue
- `GET /api/patients` - Get all patients in the queue
- `GET /api/patients/next` - Get and remove the next patient from queue
- `DELETE /api/patients/<id>` - Remove a specific patient
- `GET /api/stats` - Get queue statistics and triage distribution
- `POST /api/model/predict` - Get ML model prediction for patient data

### Frontend Components

1. **Patient Form**: Input form for adding new patients with fields for:
   - Age
   - Gender
   - Chief Complaint
   - Heart Rate
   - Blood Pressure
   - Temperature
   - Medical History

2. **Patient Queue**: Displays all patients in the queue, sorted by triage level:
   - Critical (Level 1) - Red badge
   - Moderate (Level 2) - Orange badge
   - Low (Level 3) - Green badge

3. **Statistics Dashboard**: Shows:
   - Total number of patients in queue
   - Bar chart of triage level distribution
   - Breakdown of patients by priority level

## Triage Levels

The system automatically assigns triage levels based on:

**Level 1 (Critical):**
- High priority symptoms (chest pain, shortness of breath, severe bleeding, stroke symptoms, heart attack)
- Abnormal heart rate (< 50 or > 120 bpm)
- Abnormal blood pressure (< 90 or > 180 mmHg)

**Level 2 (Moderate):**
- Medium priority symptoms (abdominal pain, head injury, moderate bleeding, fever, vomiting)
- Abnormal temperature (< 35°C or > 39°C)

**Level 3 (Low):**
- All other cases

## Usage

1. **Adding a Patient**: Fill out the patient form on the left side and click "Add Patient". The patient will be automatically assigned a triage level and added to the queue.

2. **Viewing the Queue**: The patient queue is displayed on the right side, sorted by triage level (most critical first).

3. **Processing Patients**: Click "Process Next Patient" to remove the highest priority patient from the queue.

4. **Statistics**: View real-time statistics and charts at the bottom of the page to monitor the current state of the emergency room.

## Legacy Tkinter GUI

The original Tkinter-based GUI is still available in `triage_system.py`. To use it:

```bash
python triage_system.py
```

## Encryption & Decryption

Patient data can be encrypted using the Fernet encryption system to ensure confidentiality. The encryption functions are available in the backend code.

## Development

### Backend Development

The Flask API runs in debug mode by default. To modify the backend:
- Edit `app.py` for API endpoints and logic
- Modify `triage_system.py` for ML model and triage logic

### Frontend Development

React development server supports hot-reloading. To modify the frontend:
- Edit files in `src/` directory
- Components are in `src/components/`
- API service layer is in `src/services/api.js`

## Build for Production

To create a production build of the React app:

```bash
npm run build
```

This creates an optimized build in the `build/` folder that can be served by any static file server.

## Troubleshooting

**Backend won't start:**
- Ensure all Python dependencies are installed
- Check that port 5000 is not in use
- Verify `patient_data.csv` exists in the project directory

**Frontend won't start:**
- Ensure Node.js and npm are installed
- Run `npm install` to install dependencies
- Check that port 3000 is not in use

**API Connection Errors:**
- Ensure the Flask backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Verify the API base URL in `src/services/api.js`

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

This system helps emergency room staff efficiently manage patient triage using AI-powered prioritization.