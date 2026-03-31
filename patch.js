const fs = require('fs');
const files = [
  'src/frontend/src/features/dashboard/components/DeepInferenceEngine.tsx',
  'src/frontend/src/features/comparison/components/ComparisonDashboard.tsx',
  'src/frontend/src/features/options/components/CalibrationHealth.tsx'
];

for (const f of files) {
  let content = fs.readFileSync(f, 'utf8');
  content = content.replace(/<LinearProgress/g, '<LinearProgress aria-label="Loading..."');
  fs.writeFileSync(f, content);
}
