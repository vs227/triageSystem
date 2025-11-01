import React from 'react';
import { removePatient, addCSVPatientToQueue } from '../services/api';
import './PatientQueue.css';

const PatientQueue = ({ patients, onPatientRemoved, onNextPatient, loading, onExport, isFiltered, onPatientAdded }) => {
  const getTriageLevelLabel = (level) => {
    const levelStr = String(level).toLowerCase();
    
    if (levelStr.includes('critical') || level === 1 || level === '1') {
      return { text: 'Critical', color: 'critical' };
    } else if (levelStr.includes('moderate') || level === 2 || level === '2') {
      return { text: 'Moderate', color: 'moderate' };
    } else if (levelStr.includes('low') || level === 3 || level === '3') {
      return { text: 'Low', color: 'low' };
    }
    return { text: String(level), color: 'low' };
  };

  const handleRemove = async (patientId) => {
    const patient = patients.find(p => (p.id || p.csv_id) === patientId);
    const isFromCsv = patient?.source === 'CSV Import';
    
    const confirmMessage = isFromCsv 
      ? 'Remove this patient from the queue? They will return to CSV Training Data.'
      : 'Are you sure you want to remove this patient from the queue?';
    
    if (!window.confirm(confirmMessage)) {
      return;
    }
    
    try {
      const response = await removePatient(patientId);
      if (response?.restored_to_training) {
        alert(`Patient removed from queue.\n\nCSV patient has been restored to Training Data and can be added again.`);
      }
      onPatientRemoved(patientId);
    } catch (error) {
      console.error('Error removing patient:', error);
      const errorMsg = error.response?.data?.error || error.message || 'Failed to remove patient. Please try again.';
      alert(errorMsg);
    }
  };

  return (
    <div className="patient-queue">
      <div className="queue-header">
        <h2>Patient Queue ({patients.length})</h2>
        <div className="queue-header-buttons">
          {onExport && (
            <button 
              onClick={onExport}
              className="export-queue-btn"
              title="Export queue to CSV"
            >
              Export
            </button>
          )}
          <button 
            onClick={onNextPatient} 
            disabled={loading || patients.length === 0}
            className="next-patient-btn"
          >
            {loading ? 'Processing...' : 'Process Next Patient'}
          </button>
        </div>
      </div>
      
      {!patients || patients.length === 0 ? (
        <div className="empty-queue">
          <p>No patients found</p>
          <p className="empty-hint">Try adjusting your search criteria or clear filters</p>
        </div>
      ) : (
        <div className="queue-list">
          {patients.map((patient) => {
            const triageInfo = getTriageLevelLabel(patient.triage_level || patient.triage_priority);
            const pd = patient.patient_data;
            const isCsvTrainingData = patient.isCsvTrainingData;
            const isFromCsvImport = patient.source === 'CSV Import' && !isCsvTrainingData;
            
            return (
              <div 
                key={patient.id || patient.csv_id} 
                className={`patient-card ${isCsvTrainingData ? 'csv-training-data' : ''} ${isFromCsvImport ? 'csv-imported-patient' : ''}`}
              >
                <div className="patient-header">
                  <div className="patient-id-section">
                    <span className="patient-id">
                      {isCsvTrainingData ? `CSV #${patient.csv_id}` : `ID: ${patient.id || 'N/A'}`}
                    </span>
                    {isCsvTrainingData && (
                      <span className="csv-badge" title="CSV Training Data - Not in active queue">Training Data</span>
                    )}
                    {isFromCsvImport && (
                      <span className="csv-imported-badge" title="Added to queue from CSV">CSV Import</span>
                    )}
                  </div>
                  <span className={`triage-badge ${triageInfo.color}`}>
                    {triageInfo.text}
                  </span>
                </div>
                
                {isCsvTrainingData && (
                  <div className="csv-comparison-info">
                    <span className="csv-label">Original Triage:</span>
                    <span className="csv-value">{patient.triage_level}</span>
                    {patient.triage_priority && (
                      <>
                        <span className="csv-label">Priority:</span>
                        <span className="csv-value">{patient.triage_priority}</span>
                      </>
                    )}
                    <div className="csv-note">
                      This is training data for comparison. Click "Add to Queue" to process.
                    </div>
                  </div>
                )}
                
                <div className="patient-details">
                  <div className="detail-row">
                    <span className="detail-label">Age:</span>
                    <span className="detail-value">{pd.age}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Gender:</span>
                    <span className="detail-value">{pd.gender}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Complaint:</span>
                    <span className="detail-value">{pd.chief_complaint}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Heart Rate:</span>
                    <span className="detail-value">{pd.heart_rate} bpm</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">BP:</span>
                    <span className="detail-value">{pd.blood_pressure} mmHg</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Temperature:</span>
                    <span className="detail-value">{pd.temperature}°C</span>
                  </div>
                  {pd.medical_history && (
                    <div className="detail-row">
                      <span className="detail-label">History:</span>
                      <span className="detail-value">{pd.medical_history}</span>
                    </div>
                  )}
                  {patient.confidence && (
                    <div className="detail-row">
                      <span className="detail-label">Confidence:</span>
                      <span className="detail-value">{(patient.confidence * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {patient.prediction_method && (
                    <div className="detail-row">
                      <span className="detail-label">Method:</span>
                      <span className="detail-value">{patient.prediction_method}</span>
                    </div>
                  )}
                </div>
                
                {!isCsvTrainingData && (
                  <button 
                    onClick={() => handleRemove(patient.id)}
                    className="remove-btn"
                    title="Remove from active queue"
                  >
                    Remove from Queue
                  </button>
                )}
                {isCsvTrainingData && (
                  <div className="csv-action-section">
                    <button 
                      onClick={async () => {
                        try {
                          const csvId = patient.csv_id;
                          const data = await addCSVPatientToQueue(csvId);
                          
                          if (data.success) {
                            alert(`Patient successfully added to queue!\n\nPatient will be processed with ML model prediction.`);
                            
                            if (onPatientAdded) {
                              onPatientAdded();
                            }
                          } else {
                            alert(data.error || 'Failed to add patient to queue');
                          }
                        } catch (error) {
                          console.error('Error adding CSV patient:', error);
                          const errorMsg = error.response?.data?.error || error.message || 'Failed to add patient to queue. Please try again.';
                          alert(errorMsg);
                        }
                      }}
                      className="add-from-csv-btn"
                      title="Add this CSV training patient to the active queue for processing"
                    >
                      Add to Active Queue
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default PatientQueue;
