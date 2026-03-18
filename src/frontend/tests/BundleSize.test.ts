import { test, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { gzipSync } from 'zlib';

const DIST_PATH = path.resolve(__dirname, '../dist/assets');

test('Bundle size should be within limits', () => {
  if (!fs.existsSync(DIST_PATH)) {
    console.warn('Dist folder not found. Please run build first.');
    return;
  }

  const files = fs.readdirSync(DIST_PATH);
  const jsFiles = files.filter(f => f.endsWith('.js'));

  let totalGzipSize = 0;
  let initialBundleGzipSize = 0;

  jsFiles.forEach(file => {
    const filePath = path.join(DIST_PATH, file);
    const content = fs.readFileSync(filePath);
    const gzipSize = gzipSync(content).length;
    totalGzipSize += gzipSize;

    if (file.startsWith('index')) {
      initialBundleGzipSize = gzipSize;
    }

    console.log(`File: ${file}, Gzip Size: ${(gzipSize / 1024).toFixed(2)} KB`);
  });

  const totalGzipSizeKB = totalGzipSize / 1024;
  const initialBundleGzipSizeKB = initialBundleGzipSize / 1024;

  console.log(`Total JS Gzip Size: ${totalGzipSizeKB.toFixed(2)} KB`);
  console.log(`Initial Bundle Gzip Size: ${initialBundleGzipSizeKB.toFixed(2)} KB`);

  // Limits adjusted for large dependencies (Three.js, ECharts, MUI):
  // Initial bundle < 200KB (gzipped)
  // Total JS < 1000KB (gzipped)
  
  expect(initialBundleGzipSizeKB).toBeLessThan(200);
  expect(totalGzipSizeKB).toBeLessThan(1000);
});
