import { expect, test } from 'vitest';
import { render } from '@testing-library/react';
import { toHaveNoViolations } from 'vitest-axe';
import { axe } from 'vitest-axe';
import { DashboardPage } from '../src/pages/dashboard/DashboardPage';
import { ThemeProvider } from '@mui/material/styles';
import { theme } from '../src/theme/index';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import React from 'react';

expect.extend({ toHaveNoViolations });

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <BrowserRouter>
          {children}
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

test('DashboardPage should have no accessibility violations', async () => {
  const { container } = render(<DashboardPage />, { wrapper: createWrapper() });
  
  const results = await axe(container);
  
  if (results.violations.length > 0) {
    console.error('Accessibility Violations:', JSON.stringify(results.violations, null, 2));
  }
  
  expect(results.violations).toHaveLength(0);
});
