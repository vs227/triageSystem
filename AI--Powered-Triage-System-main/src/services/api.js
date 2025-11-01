import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const addPatient = async (patientData) => {
  const response = await api.post('/patients', patientData);
  return response.data;
};

export const getPatients = async () => {
  const response = await api.get('/patients');
  return response.data;
};

export const removePatient = async (patientId) => {
  const response = await api.delete(`/patients/${patientId}`);
  return response.data;
};

export const getNextPatient = async () => {
  const response = await api.get('/patients/next');
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const predictTriage = async (patientData) => {
  const response = await api.post('/model/predict', patientData);
  return response.data;
};

export const getCSVPatients = async () => {
  const response = await api.get('/csv/patients');
  return response.data;
};

export const addCSVPatientToQueue = async (csvId) => {
  const response = await api.post(`/csv/patients/${csvId}/add-to-queue`);
  return response.data;
};

export default api;
