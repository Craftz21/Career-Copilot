import axios from 'axios';

export function getApiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
}

const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 15000,
});

export async function getHealth() {
  const { data } = await api.get('/health');
  return data;
}

export default api;
