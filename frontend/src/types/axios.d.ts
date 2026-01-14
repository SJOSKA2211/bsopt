// src/types/axios.d.ts
import { AxiosInstance } from 'axios'; // eslint-disable-line @typescript-eslint/no-unused-vars 
import { NavigateFunction } from 'react-router-dom';

declare module 'axios' {
  export interface AxiosInstance {
    getTokens(): { accessToken: string; refreshToken: string };
    setTokens(accessToken: string, refreshToken: string): void;
    clearTokens(): void;
    logout(navigate: NavigateFunction): void;
  }
}