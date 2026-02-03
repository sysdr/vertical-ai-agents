import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const executePlanning = async (task, maxIterations = 10) => {
  const response = await axios.post(`${API_BASE_URL}/plan`, {
    task,
    max_iterations: maxIterations,
    initial_context: {}
  });
  return response.data;
};
