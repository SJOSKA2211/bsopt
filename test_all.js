const { execSync } = require('child_process');
try {
  execSync('cd src/frontend && pnpm test', { stdio: 'inherit' });
} catch (e) {
  process.exit(1);
}
