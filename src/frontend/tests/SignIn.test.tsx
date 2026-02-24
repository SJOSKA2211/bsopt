import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SignIn from '../src/components/auth/SignIn';
import React from 'react';

// Mock authClient
const { mockSignInEmail } = vi.hoisted(() => {
  return { mockSignInEmail: vi.fn() }
})

vi.mock('../src/lib/auth-client', () => ({
  authClient: {
    signIn: {
      email: mockSignInEmail,
    },
  },
}));

describe('SignIn Component', () => {
  it('renders sign in form', () => {
    render(<SignIn />);
    expect(screen.getByRole('heading', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('handles submission with loading state', async () => {
    // Mock implementation to trigger callbacks
    mockSignInEmail.mockImplementation(async (data, callbacks) => {
        callbacks.onRequest();
        callbacks.onSuccess();
    });

    render(<SignIn />);

    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
        expect(screen.getByText(/signed in successfully/i)).toBeInTheDocument();
    });

    expect(mockSignInEmail).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123'
    }, expect.any(Object));
  });

  it('handles error state', async () => {
    mockSignInEmail.mockImplementation(async (data, callbacks) => {
        callbacks.onRequest();
        callbacks.onError({ error: { message: 'Invalid credentials' } });
    });

    render(<SignIn />);

    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: 'wrong@example.com' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'wrongpass' } });

    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});
