import React, { useState } from 'react';
import './ModelControl.css';

const ModelControl = ({ onModelRetrained }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleRetrain = async () => {
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/model/retrain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      let data;
      try {
        data = await response.json();
      } catch (parseError) {
        console.error('Failed to parse response:', parseError);
        setError('Invalid response from server. Check backend logs.');
        setLoading(false);
        return;
      }

      console.log('Retrain response:', response.status, data);

      if (response.ok && data.success) {
        setResult(data);
        if (onModelRetrained) {
          onModelRetrained();
        }
        setTimeout(() => {
          setResult(null);
        }, 10000);
      } else {
        const errorMsg = data.error || data.message || `Failed to retrain model (Status: ${response.status})`;
        setError(errorMsg);
        console.error('Retrain failed:', data);
      }
    } catch (err) {
      const errorMsg = err.message || 'Failed to retrain model. Please check if the backend is running.';
      setError(errorMsg);
      console.error('Error retraining model:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="model-control">
      <h3>Model Management</h3>
      <p className="model-description">
        Retrain the ML model with the latest CSV data to improve accuracy with newly added patients.
      </p>
      
      <button 
        onClick={handleRetrain} 
        disabled={loading}
        className="retrain-btn"
      >
        {loading ? 'Retraining...' : 'Retrain Model'}
      </button>

      {error && <div className="error-message">{error}</div>}
      
      {result && (
        <div className="success-message">
          <h4>Model Retrained Successfully!</h4>
          <div className="result-details">
            <p><strong>Accuracy:</strong> {(result.accuracy * 100).toFixed(2)}%</p>
            <p><strong>Training Samples:</strong> {result.training_samples}</p>
            <p><strong>Test Samples:</strong> {result.test_samples}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelControl;
