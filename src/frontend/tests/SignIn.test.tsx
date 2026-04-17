import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import SignIn from '../src/components/auth/SignIn';
import React from 'react';

// Mock useLogin hook
const mockMutateAsync = vi.fn();
vi.mock('../src/api/hooks', () => ({
  useLogin: () => ({
    mutateAsync: mockMutateAsync
  })
}));

describe('SignIn Component', () => {
  it('renders sign in form', () => {
    render(<SignIn />);
    expect(screen.getByText(/BS_OPT/i)).toBeInTheDocument();
    expect(screen.getByText(/QUANT_IDENTITY/i)).toBeInTheDocument();
  });

  it('handles submission with loading state', async () => {
    mockMutateAsync.mockResolvedValueOnce({ data: { access_token: 'test' } });

    render(<SignIn />);

    fireEvent.change(screen.getByPlaceholderText('id@bsopt.pro'), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('••••••••'), { target: { value: 'password123' } });

    fireEvent.click(screen.getByRole('button', { name: /INITIALIZE_ACCESS/i }));

    expect(mockMutateAsync).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123'
    });
  });

  it('handles error state', async () => {
    mockMutateAsync.mockRejectedValueOnce({ response: { data: { detail: 'Invalid credentials' } } });

    render(<SignIn />);

    fireEvent.change(screen.getByPlaceholderText('id@bsopt.pro'), { target: { value: 'wrong@example.com' } });
    fireEvent.change(screen.getByPlaceholderText('••••••••'), { target: { value: 'wrongpass' } });

    fireEvent.click(screen.getByRole('button', { name: /INITIALIZE_ACCESS/i }));

    await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});
