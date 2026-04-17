import React from 'react';
import { NavLink } from 'react-router-dom'; // Assuming React Router is used for navigation

const Layout = ({ children }: { children: React.ReactNode }) => (
  <div className="flex h-screen bg-bento-bg text-white overflow-hidden">
     {/* Sidebar */}
     <aside className="w-[280px] border-r border-white/10 p-8 flex flex-col justify-between">
        <div>
          <h1 className="text-3xl font-black text-mint mb-8">BS-OPT</h1>
          <nav>
            <ul className="space-y-4">
              <li>
                <NavLink 
                  to="/dashboard" 
                  className={({ isActive }) => 
                    `flex items-center p-3 rounded-lg transition-colors duration-200 ${isActive ? 'bg-mint text-bento-bg font-semibold' : 'hover:bg-gray-700 hover:bg-opacity-50'}`
                  }
                >
                  Dashboard
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/portfolios" 
                  className={({ isActive }) => 
                    `flex items-center p-3 rounded-lg transition-colors duration-200 ${isActive ? 'bg-mint text-bento-bg font-semibold' : 'hover:bg-gray-700 hover:bg-opacity-50'}`
                  }
                >
                  Portfolios
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/trade" 
                  className={({ isActive }) => 
                    `flex items-center p-3 rounded-lg transition-colors duration-200 ${isActive ? 'bg-mint text-bento-bg font-semibold' : 'hover:bg-gray-700 hover:bg-opacity-50'}`
                  }
                >
                  Trade
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/ml" 
                  className={({ isActive }) => 
                    `flex items-center p-3 rounded-lg transition-colors duration-200 ${isActive ? 'bg-mint text-bento-bg font-semibold' : 'hover:bg-gray-700 hover:bg-opacity-50'}`
                  }
                >
                  ML Pipeline
                </NavLink>
              </li>
              <li>
                <NavLink 
                  to="/market" 
                  className={({ isActive }) => 
                    `flex items-center p-3 rounded-lg transition-colors duration-200 ${isActive ? 'bg-mint text-bento-bg font-semibold' : 'hover:bg-gray-700 hover:bg-opacity-50'}`
                  }
                >
                  Market Data
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>
        {/* Footer or User Info in Sidebar */}
        <div className="mt-auto pt-8">
          <p className="text-xs text-gray-400">User: test@example.com</p> {/* Placeholder */}
          <p className="text-xs text-gray-400">v1.0.0</p>
        </div>
     </aside>

     {/* Main Content Area */}
     <div className="flex flex-col flex-grow h-screen overflow-hidden">
        <header className="h-16 border-b border-white/10 flex items-center px-8 backdrop-blur-md bg-bento-bg bg-opacity-80">
           <span className="text-xl font-semibold">Manifold Dashboard</span>
           {/* Add user profile/logout button here if needed */}
        </header>
        <main className="flex-grow overflow-auto p-8">
          {children}
        </main>
     </div>
  </div>
);

export default Layout;
