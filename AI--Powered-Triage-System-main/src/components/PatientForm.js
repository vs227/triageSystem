import React, { useState } from 'react';
import { addPatient } from '../services/api';
import './PatientForm.css';

const PatientForm = ({ onPatientAdded }) => {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    chief_complaint: '',
    heart_rate: '',
    blood_pressure: '',
    temperature: '',
    medical_history: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const patientData = {
        age: parseInt(formData.age),
        gender: formData.gender,
        chief_complaint: formData.chief_complaint,
        heart_rate: parseInt(formData.heart_rate),
        blood_pressure: parseInt(formData.blood_pressure),
        temperature: parseFloat(formData.temperature),
        medical_history: formData.medical_history || 'None'
      };

      const response = await addPatient(patientData);
      const savedToCSV = response?.saved_to_csv ?? false;
      const patient = response?.patient;
      
      const triageInfo = patient?.triage_level ? ` - Triage: ${patient.triage_level} (Priority ${patient.triage_priority || 'N/A'})` : '';
      const csvStatus = savedToCSV ? ' (Saved to CSV)' : '';
      setSuccess(`Patient added successfully!${triageInfo}${csvStatus}`);
      
      setFormData({
        age: '',
        gender: '',
        chief_complaint: '',
        heart_rate: '',
        blood_pressure: '',
        temperature: '',
        medical_history: ''
      });
      
      if (onPatientAdded) {
        onPatientAdded();
      }
    } catch (err) {
      console.error('Error adding patient:', err);
      let errorMessage = 'Failed to add patient';
      
      if (err.response) {
        const errorData = err.response.data;
        errorMessage = errorData?.error || errorData?.message || `Server error: ${err.response.status}`;
      } else if (err.request) {
        errorMessage = 'Cannot connect to server. Please make sure the backend is running on http://localhost:5000';
      } else {
        errorMessage = err.message || 'An unexpected error occurred';
      }
      
      setError(errorMessage);
      
      setTimeout(() => {
        setError('');
      }, 5000);
    } finally {
      setLoading(false);
      setTimeout(() => {
        setSuccess('');
        setError('');
      }, 3000);
    }
  };

  return (
    <div className="patient-form">
      <h2>Add New Patient</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="age">Age *</label>
          <input
            type="number"
            id="age"
            name="age"
            value={formData.age}
            onChange={handleChange}
            required
            min="0"
            max="120"
          />
        </div>

        <div className="form-group">
          <label htmlFor="gender">Gender *</label>
          <select
            id="gender"
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            required
          >
            <option value="">Select Gender</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="chief_complaint">Chief Complaint *</label>
          <input
            type="text"
            id="chief_complaint"
            name="chief_complaint"
            value={formData.chief_complaint}
            onChange={handleChange}
            required
            placeholder="e.g., Chest Pain, Headache, Fever"
          />
        </div>

        <div className="form-group">
          <label htmlFor="heart_rate">Heart Rate (bpm) *</label>
          <input
            type="number"
            id="heart_rate"
            name="heart_rate"
            value={formData.heart_rate}
            onChange={handleChange}
            required
            min="30"
            max="220"
          />
        </div>

        <div className="form-group">
          <label htmlFor="blood_pressure">Blood Pressure (mmHg) *</label>
          <input
            type="number"
            id="blood_pressure"
            name="blood_pressure"
            value={formData.blood_pressure}
            onChange={handleChange}
            required
            min="50"
            max="250"
          />
        </div>

        <div className="form-group">
          <label htmlFor="temperature">Temperature (°C) *</label>
          <input
            type="number"
            id="temperature"
            name="temperature"
            value={formData.temperature}
            onChange={handleChange}
            required
            min="30"
            max="45"
            step="0.1"
          />
        </div>

        <div className="form-group">
          <label htmlFor="medical_history">Medical History</label>
          <input
            type="text"
            id="medical_history"
            name="medical_history"
            value={formData.medical_history}
            onChange={handleChange}
            placeholder="e.g., Diabetes, Hypertension, Asthma"
          />
        </div>

        {error && <div className="error-message">{error}</div>}
        {success && <div className="success-message">{success}</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Adding...' : 'Add Patient'}
        </button>
      </form>
    </div>
  );
};

export default PatientForm;
