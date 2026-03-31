const fs = require('fs');

let f = 'src/frontend/src/features/charts/components/LivePriceChart.tsx';
let content = fs.readFileSync(f, 'utf8');
content = content.replace(/<IconButton\s*\n\s*size="small"\s*\n\s*onClick=\{\(\) => setShowSMA\(!showSMA\)\}/, '<IconButton \n              aria-label="Toggle Production Trendline"\n              size="small" \n              onClick={() => setShowSMA(!showSMA)}');
fs.writeFileSync(f, content);

f = 'src/frontend/src/components/layout/Layout.tsx';
content = fs.readFileSync(f, 'utf8');
content = content.replace(/<IconButton size="small" sx=\{\{ ml: 'auto', color: 'rgba\\(255,255,255,0.2\\)' \}\}>/g, '<IconButton aria-label="Settings" size="small" sx={{ ml: \'auto\', color: \'rgba(255,255,255,0.2)\' }}>');
content = content.replace(/<IconButton onClick=\{\(\) => setMobileOpen\(true\)\} sx=\{\{ color: '#fff', mr: 1 \}\}>/g, '<IconButton aria-label="Open mobile menu" onClick={() => setMobileOpen(true)} sx={{ color: \'#fff\', mr: 1 }}>');
content = content.replace(/<IconButton sx=\{\{ color: 'rgba\\(255,255,255,0.4\\)' \}\}>\n\s*<NotifIcon fontSize="small" \/>\n\s*<\/IconButton>/g, '<IconButton aria-label="Notifications" sx={{ color: \'rgba(255,255,255,0.4)\' }}>\n                <NotifIcon fontSize="small" />\n             </IconButton>');
content = content.replace(/<IconButton sx=\{\{ color: 'rgba\\(255,255,255,0.4\\)' \}\}>\n\s*<LogoutIcon fontSize="small" \/>\n\s*<\/IconButton>/g, '<IconButton aria-label="Log out" sx={{ color: \'rgba(255,255,255,0.4)\' }}>\n                <LogoutIcon fontSize="small" />\n             </IconButton>');
fs.writeFileSync(f, content);

f = 'src/frontend/src/features/trading/components/OrderTicket.tsx';
content = fs.readFileSync(f, 'utf8');
content = content.replace(/<IconButton size="small"><RemoveIcon fontSize="small" \/><\/IconButton>/g, '<IconButton size="small" aria-label="Decrease value"><RemoveIcon fontSize="small" /></IconButton>');
content = content.replace(/<IconButton size="small"><AddIcon fontSize="small" \/><\/IconButton>/g, '<IconButton size="small" aria-label="Increase value"><AddIcon fontSize="small" /></IconButton>');
fs.writeFileSync(f, content);
