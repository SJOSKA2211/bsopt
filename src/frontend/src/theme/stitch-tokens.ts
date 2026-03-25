export const stitchTokens = {
  colors: {
    primary: '#00FFA3', // Electric Mint
    primaryContainer: '#b1ffce',
    secondary: '#A855F7', // Cyber Purple
    secondaryContainer: '#6f00be',
    tertiary: '#3B82F6', // Deep Sea Blue
    background: '#0b0e12', // Deep Space Black
    surface: '#0b0e12',
    surfaceContainerLow: '#101418',
    surfaceContainer: '#161a1f',
    surfaceContainerHigh: '#1c2025',
    surfaceContainerHighest: '#22262c',
    onSurface: '#f5f6fc',
    onSurfaceVariant: '#a9abb1',
    outlineVariant: 'rgba(255, 255, 255, 0.15)',
    
    // Abstract Tier (from reference image)
    abstract: {
      orange: 'linear-gradient(135deg, #FF5722 0%, #FF9800 100%)',
      purple: 'linear-gradient(135deg, #673AB7 0%, #E91E63 100%)',
      teal: 'linear-gradient(135deg, #009688 0%, #4CAF50 100%)',
      indigo: 'linear-gradient(135deg, #3F51B5 0%, #00BCD4 100%)',
    }
  },
  typography: {
    headings: 'Inter, sans-serif',
    data: '"JetBrains Mono", monospace',
    labels: '"Space Grotesk", sans-serif',
  },
  effects: {
    glassBlur: 'blur(32px)',
    glassBackground: 'rgba(11, 14, 18, 0.7)',
    glassBorder: '1px solid rgba(255, 255, 255, 0.1)',
    mintGlow: '0 0 20px rgba(0, 255, 163, 0.15)',
  },
  geometry: {
    slantedCut: 'polygon(0% 0%, 95% 0%, 100% 100%, 0% 100%)',
    slantedHeader: 'polygon(0% 0%, 90% 0%, 100% 100%, 0% 100%)',
    shard: 'polygon(10% 0%, 100% 0%, 90% 100%, 0% 100%)',
    banner: 'polygon(20px 0%, 100% 0%, calc(100% - 20px) 100%, 0% 100%)',
  }
};
