
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
})

api.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('mediguide_access_token')

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

export function getApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (typeof error.response?.data?.detail === 'string') {
      return error.response.data.detail
    }

    if (!error.response) {
      return 'Cannot connect to MediGuide. Check that the backend is running.'
    }

    if (error.response.status === 422) {
      return 'Please check the information you entered.'
    }

    return 'The request could not be completed. Please try again.'
  }

  return 'An unexpected error occurred.'
}

export default api
