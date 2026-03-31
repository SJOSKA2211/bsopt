const fs = require('fs');
const files = [
  'src/frontend/src/pages/dashboard/DashboardPage.tsx'
];

for (const f of files) {
  let content = fs.readFileSync(f, 'utf8');
  content = content.replace(/<CircularProgress/g, '<CircularProgress aria-label="Loading..."');
  fs.writeFileSync(f, content);
}
