import axios from 'axios';
import { NavigateFunction } from 'react-router-dom'; // For navigation

const BASE_URL = '/api/v1'; // Assuming API base URL

// Create an Axios instance
const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

    // If the error is 401 (Unauthorized) and it's not a token refresh request itself
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true; // Mark this request as retried to prevent infinite loops

      try {
        const { refreshToken } = getTokens();
        if (refreshToken) {
          // Make a request to the refresh token endpoint
          const refreshResponse = await axios.post(`${BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token, refresh_token: newRefreshToken } = refreshResponse.data;
          setTokens(access_token, newRefreshToken);

          // Retry the original request with the new access token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return axios(originalRequest); // Retry the request
        } else {
          // No refresh token available, redirect to login
          console.error('No refresh token available, redirecting to login.');
          // For programmatic navigation, we'd need access to the navigate function.
          // This is a placeholder; actual navigation would happen where apiClient is used or via a global listener.
          // redirectToLogin(navigate); // This would require passing navigate or using a global router instance
          // For now, just clear tokens and log. A real app would redirect.
          clearTokens();
          console.warn('User needs to re-authenticate.');
        }
      } catch (refreshError: unknown) { // Changed to unknown
        console.error('Token refresh failed:', refreshError);
        clearTokens(); // Clear tokens on refresh failure
        // Redirect to login page or handle accordingly
        // redirectToLogin(navigate); // Placeholder for navigation
        console.warn('User needs to re-authenticate.');
      }
    }
    
    // Handle other errors (e.g., 400, 403, 404, 500)
    // You might want to display user-friendly messages based on error.response.data.detail
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
