# Option Pricing Analytics - Data Visualization Suite

Professional, production-ready data visualization components for option pricing analytics, built with React, TypeScript, D3.js, Plotly.js, and Recharts.

## Overview

This visualization suite provides comprehensive charting components specifically designed for financial derivatives analysis, following the principles of Edward Tufte for clarity, truthfulness, and visual integrity.

## Features

### 1. Volatility Surface (3D)
Interactive 3D visualization of implied volatility across strikes and maturities.

**Use Cases:**
- Identify volatility smiles and skews
- Analyze term structure
- Spot arbitrage opportunities
- Risk management

**Features:**
- Interactive rotation and zoom
- Color-coded volatility levels
- Moneyness vs absolute strike views
- Export to PNG/SVG

### 2. Greeks Dashboard
Comprehensive visualization of option Greeks across strike prices.

**Greeks Included:**
- Delta: Sensitivity to underlying price
- Gamma: Rate of change of delta
- Vega: Sensitivity to volatility
- Theta: Time decay
- Rho: Interest rate sensitivity

**Features:**
- Multiple chart types (line, area, bar)
- ATM reference lines
- Interactive filtering
- Real-time updates

### 3. PnL Charts
Multiple visualization modes for portfolio profit and loss analysis.

**Chart Types:**
- Time Series: PnL over time
- Waterfall: Attribution by category
- Cumulative: Running total
- Breakdown: By position

**Features:**
- Color-coded profit/loss
- Category grouping
- Export functionality
- Responsive design

### 4. Option Payoff Diagrams
Visual representation of option strategy payoffs at expiration.

**Supported Strategies:**
- Single options (calls, puts)
- Vertical spreads (bull/bear)
- Straddles and strangles
- Iron condors
- Butterfly spreads
- Custom multi-leg strategies

**Features:**
- Breakeven point markers
- Max profit/loss indicators
- Individual leg visualization
- Current spot price reference

### 5. Risk Analytics
Comprehensive risk visualization suite.

**Components:**
- VaR Distribution: Histogram with 95%/99% VaR lines
- Scenario Analysis: Stress test results
- Correlation Matrix: Asset correlation heatmap

**Features:**
- Statistical measures (VaR, CVaR)
- Color-coded risk levels
- Interactive tooltips
- Export capabilities

## Installation

```bash
npm install
```

## Usage

### Basic Example

```typescript
import { VolatilitySurface } from './components/charts/VolatilitySurface';
import { useChartTheme } from './hooks/useChartTheme';

function App() {
  const { theme } = useChartTheme();
  const data = [
    { strike: 95, maturity: 0.5, impliedVol: 0.25 },
    { strike: 100, maturity: 0.5, impliedVol: 0.22 },
    // ... more data
  ];

  return (
    <VolatilitySurface
      data={data}
      theme={theme}
      spotPrice={100}
      exportEnabled={true}
    />
  );
}
```

### Greeks Dashboard

```typescript
import { GreeksDashboard } from './components/charts/GreeksDashboard';
import { calculateGreeks } from './utils/chartUtils';

const greeksData = strikes.map(strike => ({
  strike,
  ...calculateGreeks(
    { S: 100, K: strike, T: 0.5, r: 0.05, sigma: 0.25 },
    'call'
  ),
}));

<GreeksDashboard
  data={greeksData}
  theme={theme}
  spotPrice={100}
  selectedGreeks={['delta', 'gamma', 'vega', 'theta']}
/>
```

### PnL Charts

```typescript
import { PnLCharts } from './components/charts/PnLCharts';

const pnlData = [
  { timestamp: '2024-01-01', pnl: 1000, category: 'Delta Hedging' },
  { timestamp: '2024-01-02', pnl: -500, category: 'Gamma Scalping' },
  // ... more data
];

<PnLCharts
  data={pnlData}
  theme={theme}
  chartType="waterfall" // or 'timeseries', 'cumulative', 'breakdown'
/>
```

### Option Payoff Diagram

```typescript
import { OptionPayoffDiagram } from './components/charts/OptionPayoffDiagram';

const ironCondor = {
  name: 'Iron Condor',
  legs: [
    { optionType: 'put', strike: 90, position: 'long', quantity: 1, premium: 1 },
    { optionType: 'put', strike: 95, position: 'short', quantity: 1, premium: 3 },
    { optionType: 'call', strike: 105, position: 'short', quantity: 1, premium: 3 },
    { optionType: 'call', strike: 110, position: 'long', quantity: 1, premium: 1 },
  ],
  breakeven: [93, 107],
  maxProfit: 4,
  maxLoss: -6,
};

<OptionPayoffDiagram
  strategy={ironCondor}
  theme={theme}
  spotPrice={100}
  showComponents={true}
/>
```

## Custom Hooks

### useChartTheme
Manages light/dark theme with system preference detection and persistence.

```typescript
const { theme, mode, toggleTheme, setTheme } = useChartTheme();
```

### useChartExport
Handles exporting charts to various formats.

```typescript
const { exportToPNG, exportToSVG, exportToCSV, exportToJSON } = useChartExport();

await exportToPNG(chartElement, {
  filename: 'volatility-surface',
  format: 'png',
  width: 1200,
  height: 800,
  scale: 2,
});
```

### useChartData
Manages real-time data updates via WebSocket or polling.

```typescript
const { data, isLoading, error, updateData } = useChartData(initialData, {
  dataSource: 'websocket',
  endpoint: 'ws://localhost:8000/stream',
  maxDataPoints: 1000,
});
```

### useChartResize
Responsive chart sizing with debounced resize handling.

```typescript
const containerRef = useRef<HTMLDivElement>(null);
const { width, height } = useChartResize(containerRef, {
  aspectRatio: 16 / 9,
  minWidth: 300,
  debounceMs: 150,
});
```

## Utilities

### Black-Scholes Calculations

```typescript
import { calculateBSMPrice, calculateGreeks } from './utils/chartUtils';

const price = calculateBSMPrice(
  { S: 100, K: 105, T: 0.5, r: 0.05, sigma: 0.25 },
  'call'
);

const greeks = calculateGreeks(
  { S: 100, K: 105, T: 0.5, r: 0.05, sigma: 0.25 },
  'call'
);
```

### Data Transformation

```typescript
import { downsampleData, generateVolatilitySurface } from './utils/chartUtils';

// Downsample for performance
const optimized = downsampleData(largeDataset, 1000, 'lttb');

// Generate synthetic vol surface
const surface = generateVolatilitySurface(
  spotPrice,
  strikes,
  maturities,
  baseVol
);
```

### Statistical Analysis

```typescript
import { calculateStatistics } from './utils/chartUtils';

const stats = calculateStatistics(returns);
// Returns: { mean, stdDev, var95, var99, cvar95, sharpeRatio, maxDrawdown }
```

## Storybook

View all components with interactive examples:

```bash
npm run storybook
```

This will start Storybook at `http://localhost:6006` with:
- All component variations
- Interactive controls
- Dark/light theme switching
- Documentation

## Performance

### Optimization Features

1. **Data Downsampling**: Automatically reduces data points for large datasets
2. **Memoization**: React.useMemo for expensive calculations
3. **WebGL Rendering**: For 3D surfaces with Plotly.js
4. **Debounced Resize**: Prevents excessive re-renders
5. **Code Splitting**: Chunked builds for faster initial load

### Benchmarks

Run performance benchmarks:

```bash
npm run benchmark
```

**Expected Performance:**
- Volatility Surface (1000 points): <100ms render
- Greeks Dashboard (100 strikes): <50ms render
- PnL Charts (1000 data points): <50ms render
- Payoff Diagram: <20ms render

## Accessibility

All charts include:
- ARIA labels
- Keyboard navigation
- High contrast mode support
- Screen reader compatibility
- WCAG 2.1 AA compliance

## Theming

### Light Theme
- High contrast for readability
- Subtle grid lines
- Professional color palette
- Print-friendly

### Dark Theme
- Reduced eye strain
- Optimized for prolonged use
- Enhanced color differentiation
- Modern aesthetic

### Custom Themes

```typescript
import { ChartTheme } from './types/options';

const customTheme: ChartTheme = {
  mode: 'light',
  colors: {
    primary: '#2563eb',
    secondary: '#7c3aed',
    success: '#059669',
    danger: '#dc2626',
    // ... more colors
  },
  fonts: {
    family: 'Inter, sans-serif',
    sizes: { small: 11, medium: 13, large: 15 },
  },
};
```

## Design Principles

This library follows Tuftean principles for data visualization:

1. **Data-Ink Ratio**: Maximize the proportion of ink devoted to data
2. **Chart Junk**: Eliminate unnecessary decorative elements
3. **Truthfulness**: Never distort data with misleading scales or axes
4. **Clarity**: Make every chart immediately understandable
5. **Accessibility**: Ensure visualizations work for all users

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: Latest 2 versions
- Mobile browsers: iOS Safari, Chrome Android

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- TypeScript strict mode compliance
- Unit tests for new features
- Storybook examples
- Documentation updates
- Performance benchmarks

## Support

For issues, questions, or feature requests, please open a GitHub issue.

---

Built with care for the financial analytics community.
