import React, { useState, useEffect, useMemo, useRef } from 'react';
import PatientForm from './components/PatientForm';
import PatientQueue from './components/PatientQueue';
import Stats from './components/Stats';
import CSVDataViewer from './components/CSVDataViewer';
import ModelControl from './components/ModelControl';
import PatientHistory from './components/PatientHistory';
import SearchFilter from './components/SearchFilter';
import './App.css';

function App() {
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [isFiltered, setIsFiltered] = useState(false);
  const [csvPatients, setCsvPatients] = useState([]);
  const [showCsvInQueue, setShowCsvInQueue] = useState(false);
  const [stats, setStats] = useState({ total_patients: 0, triage_distribution: {} });
  const [loading, setLoading] = useState(false);
  const isFilteredRef = useRef(false); 

  const fetchPatients = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/patients');
      const data = await response.json();
      const fetchedPatients = data.patients || [];
      setPatients(fetchedPatients);
      if (!isFilteredRef.current) {
        setFilteredPatients(fetchedPatients);
      }
    } catch (error) {
      console.error('Error fetching patients:', error);
    }
  };

  const handleSearch = async (filters) => {
    try {
      let url = 'http://localhost:5000/api/patients/search?';
      const params = new URLSearchParams();
      
      Object.keys(filters).forEach(key => {
        if (filters[key] !== undefined && filters[key] !== null && filters[key] !== '') {
          params.append(key, filters[key]);
        }
      });
      
      url += params.toString();
      
      const [queueResponse, csvResponse] = await Promise.all([
        fetch(url),
        showCsvInQueue ? fetch('http://localhost:5000/api/csv/patients') : Promise.resolve(null)
      ]);
      
      if (!queueResponse.ok) {
        const errorData = await queueResponse.json();
        throw new Error(errorData.error || `Search failed: ${queueResponse.status}`);
      }
      
      const queueData = await queueResponse.json();
      const queueResults = queueData.patients || [];
      
      let csvPatientsToFilter = csvPatients; 
      if (showCsvInQueue && csvResponse && csvResponse.ok) {
        const csvData = await csvResponse.json();
        if (csvData.patients && csvData.patients.length > 0) {
          csvPatientsToFilter = csvData.patients; 
          setCsvPatients(csvData.patients);
        }
      }
      
      let csvResults = [];
      if (showCsvInQueue && csvPatientsToFilter.length > 0) {
        console.log(`Filtering ${csvPatientsToFilter.length} CSV patients with filters:`, filters);
        
        csvResults = csvPatientsToFilter.filter(patient => {
          const pd = patient.patient_data || patient;
          if (!pd) return false;
          
          let matches = true;
          
          if (filters.q) {
            const query = filters.q.toLowerCase().trim();
            if (query) {
              const complaintMatch = pd.chief_complaint?.toLowerCase().includes(query) || false;
              const genderMatch = pd.gender?.toLowerCase().includes(query) || false;
              const historyMatch = pd.medical_history?.toLowerCase().includes(query) || false;
              matches = matches && (complaintMatch || genderMatch || historyMatch);
            }
          }
          
          if (filters.triage_level) {
            const patientTriage = String(patient.triage_level || '').toLowerCase().trim();
            const filterTriage = String(filters.triage_level || '').toLowerCase().trim();
            matches = matches && (patientTriage === filterTriage);
          }
          
          if (filters.min_age !== undefined && filters.min_age !== null && filters.min_age !== '') {
            const age = parseInt(pd.age);
            const minAge = parseInt(filters.min_age);
            if (!isNaN(age) && !isNaN(minAge)) {
              matches = matches && (age >= minAge);
            }
          }
          if (filters.max_age !== undefined && filters.max_age !== null && filters.max_age !== '') {
            const age = parseInt(pd.age);
            const maxAge = parseInt(filters.max_age);
            if (!isNaN(age) && !isNaN(maxAge)) {
              matches = matches && (age <= maxAge);
            }
          }
          
          return matches;
        }).map(p => ({
          ...p,
          isCsvTrainingData: true,
          source: 'CSV Training Data'
        }));
        
        console.log(`Found ${csvResults.length} matching CSV patients`);
      } else if (showCsvInQueue && csvPatientsToFilter.length === 0) {
        console.warn('CSV toggle is on but no CSV patients found. CSV file may be empty or not loaded.');
      }
      
      const combinedResults = [...queueResults, ...csvResults];
      combinedResults.sort((a, b) => {
        const priorityA = a.triage_priority || a.priority || 3;
        const priorityB = b.triage_priority || b.priority || 3;
        return priorityA - priorityB;
      });
      
      console.log(`Search results: ${queueResults.length} queue patients, ${csvResults.length} CSV patients, ${combinedResults.length} total`);
      setFilteredPatients(combinedResults);
      setIsFiltered(true);
      isFilteredRef.current = true; 
      
      if (combinedResults.length === 0) {
        console.warn('No patients found matching search criteria');
      }
      
      if (showCsvInQueue && csvResults.length > 0) {
      }
    } catch (error) {
      console.error('Error searching patients:', error);
      alert('Error searching patients. Please try again.');
    }
  };

  const handleClearSearch = () => {
    setIsFiltered(false);
    isFilteredRef.current = false; 
    setFilteredPatients([]);
    fetchPatients();
  };

  const handleModelRetrained = () => {
    fetchPatients();
    fetchStats();
  };

  const handleExportQueue = () => {
    window.open('http://localhost:5000/api/export/queue', '_blank');
  };

  const displayPatients = useMemo(() => {
    if (isFiltered) {
      const filtered = Array.isArray(filteredPatients) ? filteredPatients : [];
      return filtered;
    }
    
    let result = Array.isArray(patients) ? [...patients] : [];
    
    result.sort((a, b) => {
      const priorityA = a.triage_priority || a.priority || 3;
      const priorityB = b.triage_priority || b.priority || 3;
      if (priorityA === priorityB) {
        const timeA = a.timestamp || '';
        const timeB = b.timestamp || '';
        return timeB.localeCompare(timeA);
      }
      return priorityA - priorityB;
    });
    
    if (showCsvInQueue) {
      const csvToAdd = Array.isArray(csvPatients) && csvPatients.length > 0 ? csvPatients : [];
      
      if (csvToAdd.length > 0) {
        const csvWithSource = csvToAdd.map(p => ({
          ...p,
          isCsvTrainingData: true,
          source: 'CSV Training Data'
        }));
        
        const combined = [...result, ...csvWithSource];
        return combined;
      }
    }
    
    return result;
  }, [patients, csvPatients, showCsvInQueue, isFiltered, filteredPatients]);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchCSVPatients = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/csv/patients');
      if (!response.ok) {
        console.error('Failed to fetch CSV patients:', response.status);
        return;
      }
      const data = await response.json();
      if (data.patients !== undefined) {
        setCsvPatients(data.patients || []);
        console.log(`Loaded ${data.patients?.length || 0} CSV patients`);
      } else if (data.error) {
        console.warn('CSV patients error:', data.error);
        if (data.error.includes('not found')) {
          setCsvPatients([]);
        }
      }
    } catch (error) {
      console.error('Error fetching CSV patients:', error);
    }
  };

  useEffect(() => {
    fetchPatients();
    fetchStats();
    fetchCSVPatients();
    
    const interval = setInterval(() => {
      if (!isFilteredRef.current) {
        fetchPatients();
        fetchStats();
      }
    }, 3000);

    return () => clearInterval(interval);
  }, []); 
  
  useEffect(() => {
    if (!showCsvInQueue) return; 
    
    if (!isFilteredRef.current) {
      fetchCSVPatients();
    }
    
    const csvInterval = setInterval(() => {
      if (!isFilteredRef.current) {
        fetchCSVPatients();
      }
    }, 10000);

    return () => clearInterval(csvInterval);
  }, [showCsvInQueue]); 

  const handlePatientAdded = async () => {
    setIsFiltered(false);
    isFilteredRef.current = false;
    setFilteredPatients([]);
    
    await Promise.all([
      fetchPatients(),  
      fetchStats(),    
      fetchCSVPatients()
    ]);
    
    setTimeout(() => {
      if (window.csvDataViewerRefresh) {
        window.csvDataViewerRefresh();
      }
    }, 600);
    
    if (showCsvInQueue) {
      console.log('Patient added - Queue and CSV data refreshed');
    }
  };


  const handlePatientRemoved = async (patientId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/patients/${patientId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.restored_to_training && data.csv_id) {
          console.log(`CSV patient #${data.csv_id} restored to training data`);
          setTimeout(() => {
            fetchCSVPatients();
            if (window.csvDataViewerRefresh) {
              window.csvDataViewerRefresh();
            }
          }, 300);
        }
        
        await Promise.all([
          fetchPatients(),
          fetchStats()
        ]);
      }
    } catch (error) {
      console.error('Error removing patient:', error);
      fetchPatients();
      fetchStats();
    }
  };

  const handleNextPatient = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/patients/next', {
        method: 'GET',
      });
      if (response.ok) {
        const data = await response.json();
        console.log('Patient processed:', data.patient?.id || data.patient?.csv_id);
        
        await Promise.all([
          fetchPatients(),  
          fetchStats()     
        ]);
        
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('Error processing next patient:', errorData.message || 'Unknown error');
        if (errorData.message) {
          alert(errorData.message);
        }
      }
    } catch (error) {
      console.error('Error getting next patient:', error);
      alert('Failed to process next patient. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI-Powered Triage System</h1>
        <p>Emergency Room Patient Management</p>
      </header>
      
      <div className="container">
        <div className="main-content">
          <div className="left-panel">
            <PatientForm onPatientAdded={handlePatientAdded} />
          </div>
          
          <div className="right-panel">
            <SearchFilter onSearch={handleSearch} onClear={handleClearSearch} />
            {isFiltered && (
              <div className="search-active-indicator">
                <span>Search Active - Showing {filteredPatients.length} result(s)</span>
                <button onClick={handleClearSearch} className="clear-search-btn">
                  Clear Search
                </button>
              </div>
            )}
            <div className="queue-toggle-control">
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={showCsvInQueue}
                  onChange={(e) => setShowCsvInQueue(e.target.checked)}
                />
                <span>Show CSV Training Data in Queue ({csvPatients.length} patients)</span>
              </label>
              <p className="toggle-hint">
                {showCsvInQueue 
                  ? 'CSV training data is shown with blue border for comparison' 
                  : 'Enable to compare new predictions with training data'}
              </p>
            </div>
            <PatientQueue 
              patients={displayPatients} 
              onPatientRemoved={handlePatientRemoved}
              onNextPatient={handleNextPatient}
              loading={loading}
              onExport={handleExportQueue}
              isFiltered={isFiltered}
              onPatientAdded={handlePatientAdded}
            />
          </div>
        </div>
        
        <div className="features-grid">
          <div className="feature-panel">
            <ModelControl onModelRetrained={handleModelRetrained} />
          </div>
          <div className="feature-panel">
            <PatientHistory />
          </div>
        </div>
        
        <div className="csv-panel">
          <CSVDataViewer onPatientAdded={handlePatientAdded} />
        </div>
        
        <div className="stats-panel">
          <Stats stats={stats} />
        </div>
      </div>
    </div>
  );
}

export default App;
