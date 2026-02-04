import React from 'react';

declare module '@mui/material/styles' {
  interface Palette {
    financial: {
      bid: string;
      ask: string;
      positive: string;
      negative: string;
      neutral: string;
      greeks: {
        delta: string;
        gamma: string;
        vega: string;
        theta: string;
        rho: string;
      };
    };
  }
  
  interface PaletteOptions {
    financial?: {
      bid?: string;
      ask?: string;
      positive?: string;
      negative?: string;
      neutral?: string;
      greeks?: {
        delta?: string;
        gamma?: string;
        vega?: string;
        theta?: string;
        rho?: string;
      };
    };
  }
  
  interface TypeBackground {
    elevation1?: string;
    elevation2?: string;
    elevation3?: string;
  }
  
  interface TypographyVariants {
    price: React.CSSProperties;
    percentage: React.CSSProperties;
    ticker: React.CSSProperties;
  }
  
  interface TypographyVariantsOptions {
    price?: React.CSSProperties;
    percentage?: React.CSSProperties;
    ticker?: React.CSSProperties;
  }
}

declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    price: true;
    percentage: true;
    ticker: true;
  }
}

declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    buy: true;
    sell: true;
  }
}
