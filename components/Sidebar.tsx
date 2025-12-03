import React from 'react';
import { Layout, PenTool, Image as ImageIcon, Shirt, Sparkles } from 'lucide-react';
import { ModuleType } from '../types';

interface SidebarProps {
  activeModule: ModuleType;
  setActiveModule: (m: ModuleType) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeModule, setActiveModule }) => {
  const navItems = [
    { id: ModuleType.STUDIO, icon: Shirt, label: 'Studio' },
    { id: ModuleType.CAMPAIGN, icon: Layout, label: 'Campaign' },
  ];

  return (
    <div className="w-20 bg-zinc-950 border-r border-zinc-800 flex flex-col items-center py-8 z-20 shrink-0">
      <div className="mb-10 text-pink-500">
        <Sparkles size={28} />
      </div>
      <div className="flex flex-col gap-6 w-full">
        {navItems.map((item) => {
          const isActive = activeModule === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveModule(item.id)}
              className={`flex flex-col items-center gap-1 p-3 w-full transition-all duration-300 relative group
                ${isActive ? 'text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              <div className={`p-2 rounded-xl transition-all ${isActive ? 'bg-zinc-800' : ''}`}>
                <item.icon size={24} strokeWidth={1.5} />
              </div>
              <span className="text-[10px] uppercase tracking-widest font-medium mt-1">
                {item.label}
              </span>
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-pink-500 rounded-r-full" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default Sidebar;