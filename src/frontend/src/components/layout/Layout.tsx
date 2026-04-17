import React from 'react';
export const Layout = ({ children }: any) => (
  <div className="flex h-screen bg-bento-bg text-white overflow-hidden">
     <aside className="w-[280px] border-r border-white/5 p-8">
        <h1 className="text-2xl font-black text-mint">BS-OPT</h1>
     </aside>
     <div className="flex flex-col flex-grow h-screen overflow-hidden">
        <header className="h-16 border-b border-white/5 flex items-center px-8">
           <span className="text-[10px] font-black opacity-30">TERMINAL_v6.4</span>
        </header>
        <main className="flex-grow overflow-auto p-8">{children}</main>
     </div>
  </div>
);
