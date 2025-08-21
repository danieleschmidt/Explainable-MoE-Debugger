'use client';

import { useDebuggerStore } from '@/store/debugger';
import { useWebSocket } from '@/lib/websocket';
import { 
  Wifi, 
  WifiOff, 
  Activity, 
  Clock, 
  Database,
  AlertTriangle,
  TrendingUp
} from 'lucide-react';

export function StatusBar() {
  const { 
    connectionStatus, 
    routingEvents, 
    performanceMetrics,
    diagnostics,
    currentSession 
  } = useDebuggerStore();
  
  const { isConnected } = useWebSocket();

  const getConnectionIcon = () => {
    if (isConnected() && connectionStatus === 'connected') {
      return <Wifi size={14} className="text-devtools-success" />;
    }
    return <WifiOff size={14} className="text-devtools-error" />;
  };

  const getLatestPerformance = () => {
    if (performanceMetrics.length === 0) return null;
    return performanceMetrics[performanceMetrics.length - 1];
  };

  const getErrorCount = () => {
    return diagnostics.filter(d => d.severity === 'error').length;
  };

  const getWarningCount = () => {
    return diagnostics.filter(d => d.severity === 'warning').length;
  };

  const latestPerf = getLatestPerformance();

  return (
    <div className="bg-devtools-surface border-t border-devtools-border px-4 py-2">
      <div className="flex items-center justify-between text-xs text-devtools-textSecondary">
        {/* Left Side - Connection and Session Info */}
        <div className="flex items-center space-x-6">
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            {getConnectionIcon()}
            <span className={
              isConnected() && connectionStatus === 'connected' 
                ? 'text-devtools-success' 
                : 'text-devtools-error'
            }>
              {connectionStatus === 'connected' ? 'Connected' : 
               connectionStatus === 'connecting' ? 'Connecting...' : 
               'Disconnected'}
            </span>
          </div>

          {/* Session Info */}
          {currentSession && (
            <div className="flex items-center space-x-2">
              <Database size={14} />
              <span>Session: {currentSession.session_id.slice(0, 8)}...</span>
              <span className={`capitalize ${
                currentSession.status === 'active' ? 'text-devtools-success' : 
                'text-devtools-warning'
              }`}>
                {currentSession.status}
              </span>
            </div>
          )}

          {/* Event Count */}
          <div className="flex items-center space-x-2">
            <Activity size={14} />
            <span>Events: {routingEvents.length.toLocaleString()}</span>
          </div>
        </div>

        {/* Right Side - Performance and Diagnostics */}
        <div className="flex items-center space-x-6">
          {/* Performance Metrics */}
          {latestPerf && (
            <>
              <div className="flex items-center space-x-2">
                <Clock size={14} />
                <span>Latency: {latestPerf.inference_time_ms.toFixed(1)}ms</span>
              </div>

              <div className="flex items-center space-x-2">
                <TrendingUp size={14} />
                <span>
                  Throughput: {latestPerf.throughput_tokens_per_sec.toFixed(1)} tok/s
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <Database size={14} />
                <span>Memory: {latestPerf.memory_usage_mb.toFixed(0)}MB</span>
              </div>
            </>
          )}

          {/* Diagnostics Summary */}
          {(getErrorCount() > 0 || getWarningCount() > 0) && (
            <div className="flex items-center space-x-2">
              <AlertTriangle size={14} className="text-devtools-warning" />
              <span>
                {getErrorCount() > 0 && (
                  <span className="text-devtools-error">
                    {getErrorCount()} errors
                  </span>
                )}
                {getErrorCount() > 0 && getWarningCount() > 0 && ', '}
                {getWarningCount() > 0 && (
                  <span className="text-devtools-warning">
                    {getWarningCount()} warnings
                  </span>
                )}
              </span>
            </div>
          )}

          {/* Timestamp */}
          <div className="flex items-center space-x-2">
            <Clock size={14} />
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}