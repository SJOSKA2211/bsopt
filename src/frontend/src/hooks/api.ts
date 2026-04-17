import { useState, useEffect } from 'react';
import { save_memory } from '~/src/lib/memory'; // Assuming memory functions are available

// Define API base URL, preferably from environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

interface FetchOptions extends RequestInit {
    headers?: Record<string, string>;
}

export function useFetchData<T>(endpoint: string, options?: FetchOptions) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            
            // Get token from local storage or a context
            const token = localStorage.getItem('authToken'); // Example: retrieve token
            const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

            try {
                const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
                    method: 'GET', // Default method for fetch
                    headers: {
                        'Content-Type': 'application/json',
                        ...authHeaders,
                        ...(options?.headers || {}),
                    },
                    ...options,
                });
                
                if (!response.ok) {
                    let errorMsg = `HTTP error! status: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        errorMsg = errorData.detail || errorMsg;
                    } catch (e) {
                        // Ignore if response is not JSON or empty
                    }
                    throw new Error(errorMsg);
                }
                
                const result: T = await response.json();
                setData(result);
            } catch (err: any) {
                setError(err.message || 'Failed to fetch data');
                logger.error(`Fetch error for ${endpoint}: ${err.message}`);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [endpoint, JSON.stringify(options)]); // Re-fetch if endpoint or options change (deep compare options)

    return { data, loading, error };
}

export function useMutateData<T>(endpoint: string, method: 'POST' | 'PUT' | 'DELETE' | 'PATCH', options?: FetchOptions) {
    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const mutate = async (payload?: any) => {
        setLoading(true);
        setError(null);
        
        const token = localStorage.getItem('authToken'); // Example: retrieve token
        const authHeaders = token ? { Authorization: `Bearer ${token}` } : {};

        try {
            const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    ...authHeaders,
                    ...(options?.headers || {}),
                },
                body: payload ? JSON.stringify(payload) : undefined,
                ...options,
            });

            if (!response.ok) {
                let errorMsg = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.detail || errorMsg;
                } catch (e) {
                    // Ignore if response is not JSON or empty
                }
                throw new Error(errorMsg);
            }
            
            let result = null;
            // Handle responses: empty body for 204, JSON for others
            if (response.status !== 204) {
                // Check content type before parsing JSON
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    result = await response.json();
                } else {
                    // Handle non-JSON responses if necessary, or just return null/text
                    result = await response.text(); // Or handle as appropriate
                }
            }
            setData(result);
            return result;
        } catch (err: any) {
            setError(err.message || `Failed to ${method} data`);
            logger.error(`Mutation error for ${endpoint} (${method}): ${err.message}`);
            throw err; // Re-throw to allow calling component to handle
        } finally {
            setLoading(false);
        }
    };

    return { mutate, data, loading, error };
}

// --- Memory Saving Functions (Example - Assuming these are globally available or imported) ---
// These are examples based on the prompt's mention of save_memory.
// You'll need to ensure these functions are correctly implemented and accessible.

// Example: Save user preferences
export const saveUserPreference = async (key: string, value: any) => {
    try {
        // Assuming a project-specific memory scope
        await save_memory({ fact: `User preference: ${key}=${value}`, scope: 'project' });
        logger.info(`Saved user preference: ${key}=${value}`);
    } catch (error) {
        logger.error(`Failed to save user preference ${key}: ${error}`);
    }
};
