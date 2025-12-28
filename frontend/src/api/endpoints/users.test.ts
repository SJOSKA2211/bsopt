/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { getCurrentUser, getUserStats } from '@/api/endpoints/users';

const handlers = [
  http.get('/api/v1/users/me', () => {
    return HttpResponse.json({
      id: '1',
      email: 'user@example.com',
      full_name: 'John Doe',
      tier: 'free',
      is_active: true,
      is_verified: true,
      is_mfa_enabled: false,
      created_at: new Date().toISOString(),
      last_login: new Date().toISOString()
    });
  }),
  
  http.get('/api/v1/users/me/stats', () => {
    return HttpResponse.json({
      total_requests: 100,
      requests_today: 10,
      requests_this_month: 50,
      rate_limit_remaining: 90,
      rate_limit_reset: new Date().toISOString()
    });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Users API Endpoints', () => {
  it('should successfully get current user profile', async () => {
    const result = await getCurrentUser();
    
    expect(result.email).toBe('user@example.com');
    expect(result.full_name).toBe('John Doe');
  });

  it('should successfully get user stats', async () => {
    const result = await getUserStats();
    
    expect(result.total_requests).toBe(100);
    expect(result.rate_limit_remaining).toBe(90);
  });
});
