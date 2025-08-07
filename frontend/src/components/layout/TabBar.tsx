'use client';

import { useDebuggerStore } from '@/store/debugger';
import { PanelType } from '@/types';
import { 
  Network, 
  Code2, 
  Terminal, 
  Activity,
  Settings,
  HelpCircle 
} from 'lucide-react';

const tabs = [
  { 
    id: 'network' as PanelType, 
    label: 'Network', 
    icon: Network, 
    description: 'Expert routing visualization' 
  },
  { 
    id: 'elements' as PanelType, 
    label: 'Elements', 
    icon: Code2, 
    description: 'Model architecture inspection' 
  },
  { 
    id: 'console' as PanelType, 
    label: 'Console', 
    icon: Terminal, 
    description: 'Interactive debugging console' 
  },
  { 
    id: 'performance' as PanelType, 
    label: 'Performance', 
    icon: Activity, 
    description: 'Performance metrics and profiling' 
  },
];

export function TabBar() {
  const { activePanel, setActivePanel } = useDebuggerStore();

  return (
    <div className="bg-devtools-surface border-b border-devtools-border">
      <div className="flex items-center justify-between px-4 py-2">
        {/* Main Tabs */}
        <div className="flex items-center space-x-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activePanel === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActivePanel(tab.id)}
                className={`devtools-tab flex items-center space-x-2 ${
                  isActive ? 'active' : ''
                }`}
                title={tab.description}
              >
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
        
        {/* Right Side Actions */}
        <div className="flex items-center space-x-2">
          <button
            className="p-2 rounded hover:bg-devtools-background transition-colors"
            title="Settings"
          >
            <Settings size={16} />
          </button>
          
          <button
            className="p-2 rounded hover:bg-devtools-background transition-colors"
            title="Help"
          >
            <HelpCircle size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}