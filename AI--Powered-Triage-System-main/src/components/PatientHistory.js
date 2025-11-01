import React, { useState, useEffect } from 'react';
import './PatientHistory.css';

const PatientHistory = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [filter, setFilter] = useState({ triage_level: '', limit: 50 });

  useEffect(() => {
    if (expanded) {
      fetchHistory();
    }
  }, [expanded, filter]);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      let url = 'http://localhost:5000/api/history?';
      if (filter.triage_level) {
        url += `triage_level=${filter.triage_level}&`;
      }
      url += `limit=${filter.limit}`;

      const response = await fetch(url);
      const data = await response.json();
      setHistory(data.patients || []);
    } catch (error) {
      console.error('Error fetching history:', error);
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

  const handleExport = () => {
    window.open('http://localhost:5000/api/export/history', '_blank');
  };

  return (
    <div className="patient-history">
      <div className="history-header" onClick={() => setExpanded(!expanded)}>
        <h3>Patient History ({history.length} processed)</h3>
        <span className="toggle-icon">{expanded ? '▼' : '▶'}</span>
      </div>

      {expanded && (
        <div className="history-content">
          <div className="history-controls">
            <select
              value={filter.triage_level}
              onChange={(e) => setFilter({ ...filter, triage_level: e.target.value })}
              className="filter-select"
            >
              <option value="">All Triage Levels</option>
              <option value="Critical">Critical</option>
              <option value="Moderate">Moderate</option>
              <option value="Low">Low</option>
            </select>
            <button onClick={handleExport} className="export-btn">
               Export CSV
            </button>
          </div>

          {loading && <div className="loading">Loading history...</div>}

          {!loading && history.length === 0 && (
            <div className="empty-message">No patient history available</div>
          )}

          {!loading && history.length > 0 && (
            <div className="history-list">
              {history.map((patient) => {
                const pd = patient.patient_data;
                return (
                  <div key={patient.id} className="history-card">
                    <div className="history-card-header">
                      <span className="history-id">ID: {patient.id}</span>
                      <span className={`history-triage-badge ${getTriageColor(patient.triage_level)}`}>
                        {patient.triage_level}
                      </span>
                    </div>
                    <div className="history-card-body">
                      <div className="history-detail">
                        <span className="history-label">Complaint:</span>
                        <span className="history-value">{pd.chief_complaint}</span>
                      </div>
                      <div className="history-detail">
                        <span className="history-label">Age:</span>
                        <span className="history-value">{pd.age} | {pd.gender}</span>
                      </div>
                      <div className="history-detail">
                        <span className="history-label">Vitals:</span>
                        <span className="history-value">
                          HR: {pd.heart_rate} | BP: {pd.blood_pressure} | Temp: {pd.temperature}°C
                        </span>
                      </div>
                      {patient.confidence && (
                        <div className="history-detail">
                          <span className="history-label">Confidence:</span>
                          <span className="history-value">{(patient.confidence * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {patient.processed_at && (
                        <div className="history-detail">
                          <span className="history-label">Processed:</span>
                          <span className="history-value">
                            {new Date(patient.processed_at).toLocaleString()}
                          </span>
                        </div>
                      )}
                    </div>
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

export default PatientHistory;
