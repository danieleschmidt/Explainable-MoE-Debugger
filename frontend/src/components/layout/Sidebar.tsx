'use client';

import { useState } from 'react';
import { useDebuggerStore } from '@/store/debugger';
import { 
  ChevronDown, 
  ChevronRight,
  Play,
  Pause,
  Square,
  RefreshCw,
  Database,
  Layers,
  Brain,
  AlertTriangle,
  CheckCircle,
  XCircle
} from 'lucide-react';

export function Sidebar() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['session', 'model', 'diagnostics'])
  );
  
  const {
    currentSession,
    expertMetrics,
    loadBalanceMetrics,
    diagnostics,
    connectionStatus
  } = useDebuggerStore();

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'error':
        return <XCircle size={14} className="text-devtools-error" />;
      case 'warning':
        return <AlertTriangle size={14} className="text-devtools-warning" />;
      default:
        return <CheckCircle size={14} className="text-devtools-success" />;
    }
  };

  return (
    <div className="w-80 bg-devtools-surface border-r border-devtools-border flex flex-col custom-scrollbar">
      <div className="flex-1 overflow-y-auto">
        {/* Session Control Section */}
        <div className="border-b border-devtools-border">
          <button
            onClick={() => toggleSection('session')}
            className="w-full flex items-center justify-between p-3 hover:bg-devtools-background transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Database size={16} />
              <span className="font-semibold">Session</span>
            </div>
            {expandedSections.has('session') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>
          
          {expandedSections.has('session') && (
            <div className="pb-3 px-3 space-y-2">
              {currentSession ? (
                <div className="space-y-2">
                  <div className="text-sm">
                    <div className="text-devtools-textSecondary">Session ID:</div>
                    <div className="font-mono text-xs break-all">
                      {currentSession.session_id}
                    </div>
                  </div>
                  
                  <div className="text-sm">
                    <div className="text-devtools-textSecondary">Status:</div>
                    <div className={`capitalize ${
                      currentSession.status === 'active' ? 'text-devtools-success' :
                      currentSession.status === 'paused' ? 'text-devtools-warning' :
                      'text-devtools-textSecondary'
                    }`}>
                      {currentSession.status}
                    </div>
                  </div>
                  
                  <div className="text-sm">
                    <div className="text-devtools-textSecondary">Tokens Processed:</div>
                    <div>{currentSession.total_tokens_processed.toLocaleString()}</div>
                  </div>
                  
                  <div className="flex space-x-2 pt-2">
                    <button className="devtools-button-secondary flex items-center space-x-1 text-xs px-2 py-1">
                      <Pause size={12} />
                      <span>Pause</span>
                    </button>
                    <button className="devtools-button-secondary flex items-center space-x-1 text-xs px-2 py-1">
                      <RefreshCw size={12} />
                      <span>Reset</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-devtools-textSecondary">
                  No active session
                </div>
              )}
            </div>
          )}
        </div>

        {/* Model Architecture Section */}
        <div className="border-b border-devtools-border">
          <button
            onClick={() => toggleSection('model')}
            className="w-full flex items-center justify-between p-3 hover:bg-devtools-background transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Brain size={16} />
              <span className="font-semibold">Model</span>
            </div>
            {expandedSections.has('model') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>
          
          {expandedSections.has('model') && (
            <div className="pb-3 px-3 space-y-2">
              {currentSession ? (
                <div className="space-y-2">
                  <div className="text-sm">
                    <div className="text-devtools-textSecondary">Model:</div>
                    <div>{currentSession.model_name}</div>
                  </div>
                  
                  <div className="text-sm">
                    <div className="text-devtools-textSecondary">Architecture:</div>
                    <div>{currentSession.model_architecture}</div>
                  </div>
                  
                  <div className="flex space-x-4">
                    <div className="text-sm">
                      <div className="text-devtools-textSecondary">Experts:</div>
                      <div className="text-devtools-accent font-bold">
                        {currentSession.num_experts}
                      </div>
                    </div>
                    
                    <div className="text-sm">
                      <div className="text-devtools-textSecondary">Layers:</div>
                      <div className="text-devtools-accent font-bold">
                        {currentSession.num_layers}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-devtools-textSecondary">
                  No model loaded
                </div>
              )}
            </div>
          )}
        </div>

        {/* Expert Metrics Section */}
        <div className="border-b border-devtools-border">
          <button
            onClick={() => toggleSection('experts')}
            className="w-full flex items-center justify-between p-3 hover:bg-devtools-background transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Layers size={16} />
              <span className="font-semibold">Experts</span>
              <span className="text-xs bg-devtools-background px-2 py-1 rounded">
                {expertMetrics.length}
              </span>
            </div>
            {expandedSections.has('experts') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>
          
          {expandedSections.has('experts') && (
            <div className="pb-3 px-3">
              {expertMetrics.length > 0 ? (
                <div className="space-y-1 max-h-40 overflow-y-auto custom-scrollbar">
                  {expertMetrics.slice(0, 8).map((expert, index) => (
                    <div
                      key={`${expert.layer_idx}-${expert.expert_id}`}
                      className="flex items-center justify-between py-1 px-2 rounded hover:bg-devtools-background transition-colors"
                    >
                      <div className="flex items-center space-x-2">
                        <div 
                          className={`w-3 h-3 rounded-full bg-expert-${expert.expert_id % 8}`}
                        />
                        <span className="text-sm">Expert {expert.expert_id}</span>
                      </div>
                      
                      <div className="text-xs text-devtools-textSecondary">
                        {expert.activation_frequency.toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-devtools-textSecondary">
                  No expert data available
                </div>
              )}
            </div>
          )}
        </div>

        {/* Diagnostics Section */}
        <div className="border-b border-devtools-border">
          <button
            onClick={() => toggleSection('diagnostics')}
            className="w-full flex items-center justify-between p-3 hover:bg-devtools-background transition-colors"
          >
            <div className="flex items-center space-x-2">
              <AlertTriangle size={16} />
              <span className="font-semibold">Diagnostics</span>
              {diagnostics.length > 0 && (
                <span className="text-xs bg-devtools-error px-2 py-1 rounded">
                  {diagnostics.filter(d => d.severity === 'error').length}
                </span>
              )}
            </div>
            {expandedSections.has('diagnostics') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>
          
          {expandedSections.has('diagnostics') && (
            <div className="pb-3 px-3">
              {diagnostics.length > 0 ? (
                <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar">
                  {diagnostics.slice(-10).reverse().map((diagnostic, index) => (
                    <div
                      key={`${diagnostic.timestamp}-${index}`}
                      className="p-2 rounded bg-devtools-background border border-devtools-border"
                    >
                      <div className="flex items-start space-x-2">
                        {getSeverityIcon(diagnostic.severity)}
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">
                            {diagnostic.title}
                          </div>
                          <div className="text-xs text-devtools-textSecondary mt-1">
                            {diagnostic.description}
                          </div>
                          <div className="text-xs text-devtools-textSecondary mt-1">
                            {new Date(diagnostic.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-devtools-textSecondary">
                  No diagnostics available
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}