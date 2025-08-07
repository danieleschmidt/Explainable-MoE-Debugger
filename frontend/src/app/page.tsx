'use client';

import { useEffect, useState } from 'react';
import { DebuggerLayout } from '@/components/layout/DebuggerLayout';
import { NetworkPanel } from '@/components/panels/NetworkPanel';
import { ElementsPanel } from '@/components/panels/ElementsPanel';
import { ConsolePanel } from '@/components/panels/ConsolePanel';
import { PerformancePanel } from '@/components/panels/PerformancePanel';
import { useDebuggerStore } from '@/store/debugger';
import { useWebSocket } from '@/lib/websocket';
import { LoadingScreen } from '@/components/common/LoadingScreen';

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(true);
  const { activePanel, connectionStatus } = useDebuggerStore();
  const { connect, disconnect } = useWebSocket();

  useEffect(() => {
    // Initialize WebSocket connection
    connect();
    
    // Simulate loading time for better UX
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1500);

    return () => {
      clearTimeout(timer);
      disconnect();
    };
  }, [connect, disconnect]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  const renderActivePanel = () => {
    switch (activePanel) {
      case 'network':
        return <NetworkPanel />;
      case 'elements':
        return <ElementsPanel />;
      case 'console':
        return <ConsolePanel />;
      case 'performance':
        return <PerformancePanel />;
      default:
        return <NetworkPanel />;
    }
  };

  return (
    <DebuggerLayout>
      <div className="flex flex-col h-full">
        {/* Connection Status Bar */}
        <div className="bg-devtools-surface border-b border-devtools-border px-4 py-2">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div 
                className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' 
                    ? 'bg-devtools-success' 
                    : connectionStatus === 'connecting'
                    ? 'bg-devtools-warning animate-pulse'
                    : 'bg-devtools-error'
                }`}
              />
              <span className="text-sm text-devtools-textSecondary">
                {connectionStatus === 'connected' ? 'Connected to MoE Debugger' : 
                 connectionStatus === 'connecting' ? 'Connecting...' : 
                 'Disconnected'}
              </span>
            </div>
            
            <div className="text-sm text-devtools-textSecondary">
              WebSocket: ws://localhost:8000/ws
            </div>
          </div>
        </div>

        {/* Main Panel Content */}
        <div className="flex-1 overflow-hidden">
          {renderActivePanel()}
        </div>
      </div>
    </DebuggerLayout>
  );
}