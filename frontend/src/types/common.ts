export interface DataResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

export interface SuccessResponse {
  success: boolean;
  message: string;
  data?: Record<string, any>;
}

export interface ErrorDetail {
  field?: string;
  message: string;
  code?: string;
}

export interface ErrorResponse {

  error: string;

  message: string;

  details?: ErrorDetail[];

  request_id?: string;

  timestamp: string;

  retryAfter?: number; // Added for 429 handling

}
