import React, { useState, useEffect } from 'react';
import { getCSVPatients, addCSVPatientToQueue } from '../services/api';
import './CSVDataViewer.css';

const CSVDataViewer = ({ onPatientAdded }) => {
  const [csvPatients, setCsvPatients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [addingIds, setAddingIds] = useState(new Set());

  useEffect(() => {
    fetchCSVPatients();
  }, []);

  useEffect(() => {
    window.csvDataViewerRefresh = () => {
      console.log('CSV Viewer: Refreshing data...');
      setTimeout(() => {
        fetchCSVPatients();
      }, 500);
    };
    
    return () => {
      delete window.csvDataViewerRefresh;
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchCSVPatients();
    }, 15000); 

    return () => clearInterval(interval);
  }, []);

  const handleAddToQueue = async (csvId) => {
    setAddingIds(prev => new Set(prev).add(csvId));
    try {
      const data = await addCSVPatientToQueue(csvId);
      if (data.success) {
        alert(`Patient successfully moved from training data to active queue!\n\nThis patient will no longer appear in CSV Training Data.`);
        
        setCsvPatients(prev => prev.filter(p => p.csv_id !== csvId));
        
        setTimeout(async () => {
          await fetchCSVPatients();
          if (onPatientAdded) {
            onPatientAdded();
          }
        }, 300);
      } else {
        alert(data.error || 'Failed to add patient to queue');
        setAddingIds(prev => {
          const next = new Set(prev);
          next.delete(csvId);
          return next;
        });
      }
    } catch (err) {
      alert(err.response?.data?.error || 'Failed to add patient to queue');
      console.error('Error adding CSV patient:', err);
      setAddingIds(prev => {
        const next = new Set(prev);
        next.delete(csvId);
        return next;
      });
    }
  };

  const fetchCSVPatients = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await getCSVPatients();
      if (data.error) {
        setError(data.error);
      } else {
        setCsvPatients(data.patients || []);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load CSV data');
      console.error('Error fetching CSV patients:', err);
    } finally {
      setLoading(false);
    }
  };

  const getTriageColor = (level) => {
    const levelStr = String(level).toLowerCase();
    if (levelStr.includes('critical')) return 'critical';
    if (levelStr.includes('moderate')) return 'moderate';
    return 'low';
  };

  return (
    <div className="csv-data-viewer">
      <div className="csv-header" onClick={() => setExpanded(!expanded)}>
        <h3>CSV Training Data ({csvPatients.length} available patients)</h3>
        <span className="toggle-icon">{expanded ? '▼' : '▶'}</span>
      </div>
      
      {expanded && (
        <div className="csv-content">
          {loading && <div className="loading">Loading CSV data...</div>}
          {error && (
            <div className="error-message">
              {error}
              <button onClick={fetchCSVPatients} className="retry-btn" style={{marginTop: '0.5rem', padding: '0.5rem 1rem'}}>
                Retry
              </button>
            </div>
          )}
          
          {!loading && !error && csvPatients.length === 0 && (
            <div className="empty-message">
              <p>No CSV data available</p>
              <p style={{fontSize: '0.85rem', color: '#999', marginTop: '0.5rem'}}>
                Add patients through the form to populate the CSV file
              </p>
            </div>
          )}
          
          {!loading && !error && csvPatients.length > 0 && (
            <div className="csv-patients-list">
              {csvPatients.map((patient) => {
                const pd = patient.patient_data;
                const isAdding = addingIds.has(patient.csv_id);
                
                return (
                  <div key={patient.csv_id} className="csv-patient-card">
                    <div className="csv-patient-header">
                      <span className="csv-patient-id">CSV #{patient.csv_id}</span>
                      <span className={`csv-triage-badge ${getTriageColor(patient.triage_level)}`}>
                        {patient.triage_level}
                      </span>
                    </div>
                    
                    <div className="csv-patient-details">
                      <div className="csv-detail-item">
                        <span className="csv-label">Age:</span>
                        <span className="csv-value">{pd.age}</span>
                      </div>
                      <div className="csv-detail-item">
                        <span className="csv-label">Gender:</span>
                        <span className="csv-value">{pd.gender}</span>
                      </div>
                      <div className="csv-detail-item">
                        <span className="csv-label">Complaint:</span>
                        <span className="csv-value">{pd.chief_complaint}</span>
                      </div>
                      <div className="csv-detail-item">
                        <span className="csv-label">HR:</span>
                        <span className="csv-value">{pd.heart_rate} bpm</span>
                      </div>
                      <div className="csv-detail-item">
                        <span className="csv-label">BP:</span>
                        <span className="csv-value">{pd.blood_pressure} mmHg</span>
                      </div>
                      <div className="csv-detail-item">
                        <span className="csv-label">Temp:</span>
                        <span className="csv-value">{pd.temperature}°C</span>
                      </div>
                      {pd.medical_history && (
                        <div className="csv-detail-item">
                          <span className="csv-label">History:</span>
                          <span className="csv-value">{pd.medical_history}</span>
                        </div>
                      )}
                    </div>
                    
                    <button
                      onClick={() => handleAddToQueue(patient.csv_id)}
                      disabled={isAdding}
                      className="add-to-queue-btn"
                    >
                      {isAdding ? 'Adding...' : '+ Add to Queue'}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CSVDataViewer;
