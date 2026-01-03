import { useState, useCallback } from 'react';
import { AxiosError } from 'axios';
import { ErrorResponse } from '@/types/common';

interface UseApiState<T> {
  data: T | null;
  error: ErrorResponse | null;
  loading: boolean;
}

export const useApi = <T, Args extends any[]>(
  apiFunc: (...args: Args) => Promise<T>
) => {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    error: null,
    loading: false,
  });

  const execute = useCallback(
    async (...args: Args): Promise<T | null> => {
      setState((prev) => ({ ...prev, loading: true, error: null }));
      try {
        const result = await apiFunc(...args);
        setState({ data: result, error: null, loading: false });
        return result;
      } catch (err) {
        const axiosError = err as AxiosError<ErrorResponse>;
        const errorData: ErrorResponse = axiosError.response?.data || {
          error: 'UnknownError',
          message: axiosError.message || 'An unexpected error occurred',
          timestamp: new Date().toISOString(),
        };
        setState({ data: null, error: errorData, loading: false });
        return null;
      }
    },
    [apiFunc]
  );

  return { ...state, execute };
};
