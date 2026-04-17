import React from 'react';
const DashboardPage = () => (
  <div className="bento-grid">
     <div className="col-span-12 lg:col-span-4 bento-card">
        <span className="label-secondary">NET_LIQUIDATION</span>
        <div className="text-3xl font-black mt-2 font-mono">$254,120.42</div>
     </div>
     <div className="col-span-12 lg:col-span-8 bento-card h-[400px]">
        <span className="label-secondary">SIGNAL_ENGINE</span>
     </div>
  </div>
);
export default DashboardPage;
