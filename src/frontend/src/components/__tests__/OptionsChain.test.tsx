import { render, screen, waitFor } from '@testing-library/react';
import { OptionsChain } from '../OptionsChain';
import { useWASMPricer } from '../../hooks/useWASMPricer';
import { vi, describe, it, expect, beforeEach } from 'vitest';

// Mock the hook
vi.mock('../../hooks/useWASMPricer');

describe('OptionsChain', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders loading state initially', () => {
    vi.mocked(useWASMPricer).mockReturnValue({
      isInitialized: false,
      error: null,
      calculateGreeks: vi.fn(),
      priceCall: vi.fn(),
    });

    render(<OptionsChain />);
    expect(screen.getByText('Initializing WASM pricing engine...')).toBeInTheDocument();
  });

  it('renders options table when initialized', async () => {
    const mockPriceCall = vi.fn().mockResolvedValue(10.5);
    const mockCalculateGreeks = vi.fn().mockResolvedValue({
      delta: 0.5,
      gamma: 0.1,
      vega: 0.2,
      theta: -0.05,
      rho: 0.01
    });

    vi.mocked(useWASMPricer).mockReturnValue({
      isInitialized: true,
      error: null,
      calculateGreeks: mockCalculateGreeks,
      priceCall: mockPriceCall,
    });

    render(<OptionsChain />);

    // Wait for the effect to run and data to populate
    await waitFor(() => {
        // We expect 21 rows (strikes 90-110)
        // Let's check for one of them
        expect(screen.getByText('90')).toBeInTheDocument();
    });

    // Check for "Rendered 21 options"
    // This will fail if options.len is used instead of options.length
    expect(screen.getByText(/Rendered 21 options/)).toBeInTheDocument();
    
    expect(screen.getAllByText('$10.50')).toHaveLength(21);
  });
});
