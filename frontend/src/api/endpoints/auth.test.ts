/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { login, register, LoginRequest, RegisterRequest } from '@/api/endpoints/auth';

const handlers = [
  http.post('/api/v1/auth/login', async ({ request }) => {
    const body = await request.json() as LoginRequest;
    if (body.email === 'user@example.com' && body.password === 'password') {
      return HttpResponse.json({
        success: true,
        data: {
          access_token: 'fake_access_token',
          refresh_token: 'fake_refresh_token',
          token_type: 'bearer',
          expires_in: 1800,
          user_id: '1',
          email: body.email,
          tier: 'free',
          requires_mfa: false
        }
      });
    }
    return new HttpResponse(JSON.stringify({ error: 'AuthenticationError', message: 'Invalid credentials' }), { status: 401 });
  }),
  
  http.post('/api/v1/auth/register', async ({ request }) => {
    const body = await request.json() as RegisterRequest;
    return HttpResponse.json({
      success: true,
      data: {
        user_id: '2',
        email: body.email,
        message: 'Registration successful',
        verification_required: true
      }
    });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Auth API Endpoints', () => {
  it('should successfully login', async () => {
    const requestData: LoginRequest = {
      email: 'user@example.com',
      password: 'password'
    };
    
    const result = await login(requestData);
    
    expect(result.access_token).toBe('fake_access_token');
    expect(result.user_id).toBe('1');
  });

  it('should fail login with wrong credentials', async () => {
    const requestData: LoginRequest = {
      email: 'wrong@example.com',
      password: 'password'
    };
    
    await expect(login(requestData)).rejects.toThrow();
  });

  it('should successfully register', async () => {
    const requestData: RegisterRequest = {
      email: 'new@example.com',
      password: 'password123',
      password_confirm: 'password123',
      accept_terms: true
    };
    
    const result = await register(requestData);
    
    expect(result.email).toBe('new@example.com');
    expect(result.verification_required).toBe(true);
  });
});
