'use client';

import { useEffect, useRef, useState } from 'react';
import { useDebuggerStore } from '@/store/debugger';
import { RoutingVisualization } from '@/components/visualizations/RoutingVisualization';
import { ControlPanel } from '@/components/visualizations/ControlPanel';
import { MetricsOverlay } from '@/components/visualizations/MetricsOverlay';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Download, 
  Settings,
  Maximize2,
  Filter
} from 'lucide-react';

export function NetworkPanel() {
  const [isPaused, setIsPaused] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const {
    routingEvents,
    expertMetrics,
    loadBalanceMetrics,
    visualizationConfig,
    filterConfig,
    updateVisualizationConfig,
    clearAllData
  } = useDebuggerStore();

  // Get recent routing events for visualization
  const recentEvents = routingEvents.slice(-visualizationConfig.maxTokensDisplay);

  const handlePlayPause = () => {
    setIsPaused(!isPaused);
    updateVisualizationConfig({ 
      animateRouting: !isPaused 
    });
  };

  const handleReset = () => {
    clearAllData();
  };

  const handleExportData = () => {
    const data = {
      routing_events: recentEvents,
      expert_metrics: expertMetrics,
      load_balance_metrics: loadBalanceMetrics,
      timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `moe-debug-data-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full bg-devtools-background">
      {/* Top Toolbar */}
      <div className="bg-devtools-surface border-b border-devtools-border p-3">
        <div className="flex items-center justify-between">
          {/* Left Controls */}
          <div className="flex items-center space-x-2">
            <button
              onClick={handlePlayPause}
              className="devtools-button-secondary flex items-center space-x-2"
              title={isPaused ? 'Resume visualization' : 'Pause visualization'}
            >
              {isPaused ? <Play size={16} /> : <Pause size={16} />}
              <span>{isPaused ? 'Resume' : 'Pause'}</span>
            </button>
            
            <button
              onClick={handleReset}
              className="devtools-button-secondary flex items-center space-x-2"
              title="Clear all data"
            >
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>
            
            <div className="w-px h-6 bg-devtools-border mx-2" />
            
            <button
              onClick={() => setShowControls(!showControls)}
              className={`devtools-button-secondary flex items-center space-x-2 ${
                showControls ? 'bg-devtools-accent text-white' : ''
              }`}
              title="Toggle control panel"
            >
              <Settings size={16} />
              <span>Controls</span>
            </button>
            
            <button
              onClick={handleExportData}
              className="devtools-button-secondary flex items-center space-x-2"
              title="Export debug data"
            >
              <Download size={16} />
              <span>Export</span>
            </button>
          </div>
          
          {/* Right Info */}
          <div className="flex items-center space-x-4 text-sm text-devtools-textSecondary">
            <div>
              Events: <span className="text-devtools-text">{routingEvents.length}</span>
            </div>
            <div>
              Experts: <span className="text-devtools-text">{expertMetrics.length}</span>
            </div>
            {recentEvents.length > 0 && (
              <div>
                Showing: <span className="text-devtools-text">
                  {Math.min(recentEvents.length, visualizationConfig.maxTokensDisplay)}
                </span>
              </div>
            )}
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-1 rounded hover:bg-devtools-background transition-colors"
              title="Toggle fullscreen"
            >
              <Maximize2 size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Control Panel Sidebar */}
        {showControls && (
          <div className="w-80 bg-devtools-surface border-r border-devtools-border">
            <ControlPanel />
          </div>
        )}
        
        {/* Visualization Area */}
        <div className="flex-1 relative overflow-hidden">
          {recentEvents.length > 0 ? (
            <>
              <RoutingVisualization 
                routingEvents={recentEvents}
                expertMetrics={expertMetrics}
                config={visualizationConfig}
                filter={filterConfig}
                isPaused={isPaused}
              />
              
              {/* Metrics Overlay */}
              <MetricsOverlay 
                loadBalanceMetrics={loadBalanceMetrics}
                expertMetrics={expertMetrics}
              />
            </>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 mx-auto bg-devtools-surface rounded-full flex items-center justify-center">
                  <Filter size={32} className="text-devtools-textSecondary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-devtools-text mb-2">
                    No Routing Data
                  </h3>
                  <p className="text-devtools-textSecondary max-w-md">
                    Start a debugging session to see expert routing visualization. 
                    Routing events will appear here in real-time as tokens are processed.
                  </p>
                </div>
                <button className="devtools-button">
                  Start Debug Session
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}