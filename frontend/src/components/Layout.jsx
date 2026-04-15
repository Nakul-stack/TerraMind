import React from 'react';
import Navbar from './Navbar';

export default function Layout({ children, modelMode, version }) {
  return (
    <div className="gradient-bg min-h-screen flex flex-col relative">
      <div className="relative z-10 flex flex-col min-h-screen">
        <Navbar modelMode={modelMode} version={version} />
        
        <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
          {children}
        </main>

        <footer className="border-t border-slate-800/50 py-8 bg-slate-900/40 backdrop-blur-md">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4 text-[10px] text-slate-500 font-medium uppercase tracking-widest">
            <div className="flex items-center gap-4">
              <span>TerraMind Agriculture Intel</span>
              <span className="w-1.5 h-1.5 rounded-full bg-slate-700" />
              <span>Hybrid Edge-Enabled Platform</span>
            </div>
            <div className="flex items-center gap-4">
              <span>© 2026 Nakul-stack</span>
              <span className="w-1.5 h-1.5 rounded-full bg-slate-700" />
              <span>Experimental Build</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
