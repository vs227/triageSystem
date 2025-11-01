import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from cryptography.fernet import Fernet
import threading
import time
import logging

# Load patient data from CSV
def load_patient_data(file_path):
    """Load patient data from a CSV file."""
    return pd.read_csv(r"C:\Users\vaish\Downloads\AI--Powered-Triage-System-main\AI--Powered-Triage-System-main\patient_data.csv")


# Preprocess data: Impute, scale and encode
def preprocess_data(data):
    """Preprocess the data, handle missing values, encode categorical variables, etc."""
    numeric_features = ['age', 'heart_rate', 'blood_pressure', 'temperature']
    categorical_features = ['gender', 'chief_complaint', 'medical_history']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X = data.drop('triage_level', axis=1)
    y = data['triage_level']
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

# Define a function to determine triage level based on symptoms and vitals
def determine_triage_level(patient_data):
    """Determine triage level based on patient data."""
    high_priority_symptoms = ['chest pain', 'shortness of breath', 'severe bleeding', 'stroke symptoms', 'heart attack']
    medium_priority_symptoms = ['abdominal pain', 'head injury', 'moderate bleeding', 'fever', 'vomiting']

    chief_complaint = patient_data['chief_complaint'].lower()

    # Check for high priority symptoms
    if any(symptom in chief_complaint for symptom in high_priority_symptoms):
        return 1  # Highest priority

    # Check for medium priority symptoms
    if any(symptom in chief_complaint for symptom in medium_priority_symptoms):
        return 2  # Medium priority

    # Check vital signs
    if patient_data['heart_rate'] > 120 or patient_data['heart_rate'] < 50:
        return 1  # Abnormal heart rate
    if patient_data['blood_pressure'] > 180 or patient_data['blood_pressure'] < 90:
        return 1  # Abnormal blood pressure
    if patient_data['temperature'] > 39 or patient_data['temperature'] < 35:
        return 2  # Fever or hypothermia

    # Default to low priority if no other conditions are met
    return 3

# Train the Random Forest model
def train_triage_model(X_train, y_train):
    """Train a Random Forest model for triage classification."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model on test data
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

class TriageSystem:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.patient_queue = []
        self.waiting_times = {}
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.real_time_data_integration, daemon=True).start()

    # Simulate real-time data fetching
    def real_time_data_integration(self):
        while self.running:
            new_patient_data = self.fetch_new_patient_data()
            if new_patient_data:
                self.add_patient(new_patient_data)
            time.sleep(5)  # Simulates data fetching interval

    # Mock method to simulate receiving new patient data
    def fetch_new_patient_data(self):
        import random
        if random.choice([True, False]):
            return {
                'age': random.randint(0, 100),
                'gender': random.choice(['Male', 'Female']),
                'chief_complaint': random.choice(['Chest Pain', 'Headache', 'Fever', 'Shortness of Breath', 'Abdominal Pain']),
                'heart_rate': random.randint(60, 100),
                'blood_pressure': random.randint(90, 140),
                'temperature': round(random.uniform(36.1, 37.5), 1),
                'medical_history': random.choice(['None', 'Hypertension', 'Diabetes', 'Asthma'])
            }

    # Add a new patient to the queue
    def add_patient(self, patient_data):
        with self.lock:
            triage_level = determine_triage_level(patient_data)
            patient_id = len(self.patient_queue) + 1
            self.patient_queue.append((patient_id, patient_data, triage_level))
            self.waiting_times[patient_id] = 0
            self._sort_queue()

    # Sort the queue based on triage level and waiting time
    def _sort_queue(self):
        self.patient_queue.sort(key=lambda x: (x[2], self.waiting_times[x[0]]))

    # Get next patient in the queue
    def get_next_patient(self):
        if self.patient_queue:
            return self.patient_queue.pop(0)
        return None

    # Update waiting times of all patients in the queue
    def update_waiting_times(self):
        for patient_id, _, _ in self.patient_queue:
            self.waiting_times[patient_id] += 1
        self._sort_queue()

class TriageUI:
    def __init__(self, triage_system):
        self.triage_system = triage_system
        self.root = tk.Tk()
        self.root.title("AI-Powered Triage System")
        self.setup_ui()

    # Setup the UI elements
    def setup_ui(self):
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input form for patient data
        ttk.Label(input_frame, text="Age:").grid(column=0, row=0, sticky=tk.W)
        self.age_entry = ttk.Entry(input_frame)
        self.age_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Gender:").grid(column=0, row=1, sticky=tk.W)
        self.gender_entry = ttk.Entry(input_frame)
        self.gender_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Chief Complaint:").grid(column=0, row=2, sticky=tk.W)
        self.complaint_entry = ttk.Entry(input_frame)
        self.complaint_entry.grid(column=1, row=2, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Heart Rate:").grid(column=0, row=3, sticky=tk.W)
        self.heart_rate_entry = ttk.Entry(input_frame)
        self.heart_rate_entry.grid(column=1, row=3, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Blood Pressure:").grid(column=0, row=4, sticky=tk.W)
        self.blood_pressure_entry = ttk.Entry(input_frame)
        self.blood_pressure_entry.grid(column=1, row=4, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Temperature:").grid(column=0, row=5, sticky=tk.W)
        self.temperature_entry = ttk.Entry(input_frame)
        self.temperature_entry.grid(column=1, row=5, sticky=(tk.W, tk.E))

        ttk.Label(input_frame, text="Medical History:").grid(column=0, row=6, sticky=tk.W)
        self.history_entry = ttk.Entry(input_frame)
        self.history_entry.grid(column=1, row=6, sticky=(tk.W, tk.E))

        ttk.Button(input_frame, text="Add Patient", command=self.add_patient).grid(column=1, row=7, sticky=tk.E)

        # Queue display
        queue_frame = ttk.Frame(self.root, padding="10")
        queue_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.queue_display = ttk.Treeview(queue_frame, columns=('ID', 'Age', 'Gender', 'Complaint', 'Triage Level'))
        self.queue_display.heading('ID', text='ID')
        self.queue_display.heading('Age', text='Age')
        self.queue_display.heading('Gender', text='Gender')
        self.queue_display.heading('Complaint', text='Chief Complaint')
        self.queue_display.heading('Triage Level', text='Triage Level')
        self.queue_display.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Visualization frame
        viz_frame = ttk.Frame(self.root, padding="10")
        viz_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Add patient to the system
    def add_patient(self):
        patient_data = {
            'age': int(self.age_entry.get()),
            'gender': self.gender_entry.get(),
            'chief_complaint': self.complaint_entry.get(),
            'heart_rate': int(self.heart_rate_entry.get()),
            'blood_pressure': int(self.blood_pressure_entry.get()),
            'temperature': float(self.temperature_entry.get()),
            'medical_history': self.history_entry.get()
        }
        self.triage_system.add_patient(patient_data)
        self.update_display()

    # Update the display of the queue
    def update_display(self):
        for item in self.queue_display.get_children():
            self.queue_display.delete(item)
        for patient in self.triage_system.patient_queue:
            self.queue_display.insert('', 'end', values=(patient[0], patient[1]['age'], patient[1]['gender'], patient[1]['chief_complaint'], patient[2]))
        self.update_visualization()

    # Update the visualization
    def update_visualization(self):
        self.ax.clear()
        triage_levels = [patient[2] for patient in self.triage_system.patient_queue]
        sns.countplot(x=triage_levels, ax=self.ax)
        self.ax.set_title('Current Triage Level Distribution')
        self.ax.set_xlabel('Triage Level')
        self.ax.set_ylabel('Number of Patients')
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

# Encryption and decryption using cryptography
def encrypt_data(data, key):
    """Encrypt sensitive patient data."""
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    """Decrypt sensitive patient data."""
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()

# Generate a key for encryption
def generate_encryption_key():
    """Generate a key for encryption using Fernet."""
    return Fernet.generate_key()

if __name__ == "__main__":
    # Load the data
    data = load_patient_data('patient_data.csv')

    # Preprocess the data
    X, y, preprocessor = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_triage_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Initialize the triage system
    triage_system = TriageSystem(model, preprocessor)

    # Launch the GUI
    triage_ui = TriageUI(triage_system)
    triage_ui.run()