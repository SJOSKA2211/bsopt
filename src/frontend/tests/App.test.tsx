import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import App from '../src/App';

test('App renders Vite + React title', () => {
  render(<App />);
  expect(screen.getByText('Vite + React')).toBeDefined();
});
