import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { expect, describe, it, vi, Mock } from 'vitest';
import '@testing-library/jest-dom';
import Analysis from './Analysis';
import { getOptionPrice } from '../services/pricingService';

vi.mock('../services/pricingService');

const mockedGetOptionPrice = getOptionPrice as Mock;

describe('Analysis Page', () => {
    it('fetches and displays the option price', async () => {
        mockedGetOptionPrice.mockResolvedValue({ price: 5.6789 });

        render(<Analysis />);

        fireEvent.click(screen.getByText('Calculate Price'));

        await waitFor(() => {
            expect(screen.getByText('Calculated Price: 5.6789')).toBeInTheDocument();
        });
    });

    it('handles errors from the api', async () => {
        mockedGetOptionPrice.mockRejectedValue(new Error('API Error'));

        render(<Analysis />);

        fireEvent.click(screen.getByText('Calculate Price'));

        await waitFor(() => {
            expect(screen.getByText('API Error')).toBeInTheDocument();
        });
    });
});