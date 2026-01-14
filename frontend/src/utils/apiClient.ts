import axios from 'axios';
import { NavigateFunction } from 'react-router-dom'; // For navigation

const BASE_URL = '/api/v1'; // Assuming API base URL

// Create an Axios instance
const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 10000, // 10s timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Basic retry mechanism for GET requests
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1s

// Function to get tokens from local storage
const getTokens = () => {
  const accessToken = localStorage.getItem('accessToken');
  const refreshToken = localStorage.getItem('refreshToken');
  return { accessToken, refreshToken };
};

// Function to set tokens (e.g., after login or refresh)
const setTokens = (accessToken: string, refreshToken: string) => {
  localStorage.setItem('accessToken', accessToken);
  localStorage.setItem('refreshToken', refreshToken);
};

// Function to clear tokens (e.g., on logout)
const clearTokens = () => {
  localStorage.removeItem('accessToken');
  localStorage.removeItem('refreshToken');
  localStorage.removeItem('userId'); // Clear other relevant user info
  localStorage.removeItem('userTier');
};

// Function to handle redirection after logout or auth failure
const redirectToLogin = (navigate: NavigateFunction) => {
  clearTokens();
  navigate('/login', { replace: true });
};

// Request interceptor to add Authorization header and handle refresh
apiClient.interceptors.request.use(
  async (config) => {
    const { accessToken } = getTokens();
    if (accessToken) {
      config.headers.Authorization = `Bearer ${accessToken}`;
    }
    // If CSRF is handled via cookies and requires header, add it here
    // const csrfToken = document.cookie.split('; ').find(row => row.startsWith('csrf_token='));
    // if (csrfToken) {
    //   config.headers['X-CSRF-Token'] = csrfToken.split('=')[1];
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling token refresh and other errors
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle Token Refresh (401)
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      try {
        const { refreshToken } = getTokens();
        if (refreshToken) {
          const refreshResponse = await axios.post(`${BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });
          const { access_token, refresh_token: newRefreshToken } = refreshResponse.data.data;
          setTokens(access_token, newRefreshToken);
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return axios(originalRequest);
        }
      } catch (refreshError) {
        clearTokens();
        return Promise.reject(refreshError);
      }
    }

    // Handle 429 (Rate Limit)
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['retry-after'];
      error.response.data = {
        ...error.response.data,
        retryAfter: retryAfter ? parseInt(retryAfter, 10) : 60
      };
      console.warn(`Rate limit exceeded. Retry after ${error.response.data.retryAfter}s`);
    }

    // Handle 503 (Circuit Breaker)
    if (error.response?.status === 503) {
      console.error('Service temporarily unavailable (Circuit Breaker OPEN). Please try again later.');
    }

    // Handle Network/Timeout Retries for GET requests
    if (
      originalRequest.method === 'get' &&
      (error.code === 'ECONNABORTED' || !error.response) &&
      (!originalRequest._retryCount || originalRequest._retryCount < MAX_RETRIES)
    ) {
      originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * originalRequest._retryCount));
      return apiClient(originalRequest);
    }
    
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Function to perform logout
const logout = async (navigate: NavigateFunction) => {
  try {
    // Call backend logout endpoint to invalidate token server-side if it exists
    await apiClient.post('/auth/logout'); 
  } catch (err: unknown) { // Changed to unknown
    console.error('Logout API call failed (may still be logged out client-side):', err);
  } finally {
    // Always clear tokens and redirect client-side
    clearTokens();
    navigate('/login', { replace: true });
  }
};

// Expose necessary functions
export { getTokens, setTokens, clearTokens, redirectToLogin, logout };
export default apiClient;
