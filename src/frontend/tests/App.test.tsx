import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import App from '../src/App';
import React from 'react';

test('App renders Trading Dashboard', async () => {
  render(<App />);
  expect(await screen.findByText(/BS-Opt Trading Dashboard/i)).toBeDefined();
});