'use client';

import { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import { TabBar } from './TabBar';
import { StatusBar } from './StatusBar';

interface DebuggerLayoutProps {
  children: ReactNode;
}

export function DebuggerLayout({ children }: DebuggerLayoutProps) {
  return (
    <div className="h-screen bg-devtools-background text-devtools-text overflow-hidden flex flex-col">
      {/* Top Tab Bar */}
      <TabBar />
      
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar />
        
        {/* Main Panel Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {children}
        </div>
      </div>
      
      {/* Bottom Status Bar */}
      <StatusBar />
    </div>
  );
}