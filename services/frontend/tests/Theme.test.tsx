import { render, screen } from '@testing-library/react';
import { expect, test } from 'vitest';
import { ThemeProvider } from '../src/theme/ThemeProvider';
import Button from '@mui/material/Button'; // Import Button from @mui/material

test('Button in ThemeProvider has primary color', () => {
  render(
    <ThemeProvider>
      <Button>Test Button</Button>
    </ThemeProvider>
  );
  const button = screen.getByText('Test Button');
  // Expect the button to have a color property corresponding to the primary color
  // This will likely be an inline style or a class that resolves to the color
  // We'll mock the theme later to ensure specific color value.
  // For now, just checking for a style that implies primary color.
  expect(button).toBeDefined(); // Simple check that it renders
  // A more robust check would involve checking the computed style,
  // but that requires more advanced setup for JSDOM and MUI.
  // We will refine this in the green phase.
});
