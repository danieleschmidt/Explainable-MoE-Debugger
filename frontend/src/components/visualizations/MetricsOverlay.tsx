'use client';

import { useState } from 'react';
import { LoadBalanceMetrics, ExpertMetrics } from '@/types';
import { 
  BarChart3, 
  TrendingUp, 
  AlertTriangle, 
  Users,
  X,
  Minimize2,
  Maximize2
} from 'lucide-react';

interface Props {
  loadBalanceMetrics: LoadBalanceMetrics[];
  expertMetrics: ExpertMetrics[];
}

export function MetricsOverlay({ loadBalanceMetrics, expertMetrics }: Props) {
  const [isVisible, setIsVisible] = useState(true);
  const [isMinimized, setIsMinimized] = useState(false);

  if (!isVisible) return null;

  const latestLoadBalance = loadBalanceMetrics[loadBalanceMetrics.length - 1];
  const totalExperts = expertMetrics.length;
  const activeExperts = expertMetrics.filter(e => e.activation_frequency > 0).length;
  const deadExperts = latestLoadBalance?.dead_experts.length || 0;

  return (
    <div className="absolute top-4 left-4 bg-devtools-surface border border-devtools-border rounded-lg shadow-lg z-20">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-devtools-border">
        <div className="flex items-center space-x-2">
          <BarChart3 size={16} />
          <span className="font-semibold text-devtools-text text-sm">
            Live Metrics
          </span>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setIsMinimized(!isMinimized)}
            className="p-1 rounded hover:bg-devtools-background transition-colors"
            title={isMinimized ? 'Expand' : 'Minimize'}
          >
            {isMinimized ? <Maximize2 size={14} /> : <Minimize2 size={14} />}
          </button>
          <button
            onClick={() => setIsVisible(false)}
            className="p-1 rounded hover:bg-devtools-background transition-colors"
            title="Close"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      {/* Content */}
      {!isMinimized && (
        <div className="p-3 space-y-3 min-w-[250px]">
          {/* Expert Status */}
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-devtools-textSecondary uppercase">
              Expert Status
            </h4>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="metric-card">
                <div className="flex items-center space-x-2">
                  <Users size={14} className="text-devtools-success" />
                  <div>
                    <div className="text-lg font-bold text-devtools-text">
                      {activeExperts}
                    </div>
                    <div className="text-xs text-devtools-textSecondary">
                      Active
                    </div>
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center space-x-2">
                  <AlertTriangle size={14} className="text-devtools-error" />
                  <div>
                    <div className="text-lg font-bold text-devtools-text">
                      {deadExperts}
                    </div>
                    <div className="text-xs text-devtools-textSecondary">
                      Dead
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Load Balance */}
          {latestLoadBalance && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-devtools-textSecondary uppercase">
                Load Balance
              </h4>
              
              <div className="metric-card">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-devtools-textSecondary">
                      Fairness Index
                    </span>
                    <span className={`text-sm font-semibold ${
                      latestLoadBalance.fairness_index > 0.8 
                        ? 'text-devtools-success'
                        : latestLoadBalance.fairness_index > 0.6
                        ? 'text-devtools-warning'
                        : 'text-devtools-error'
                    }`}>
                      {latestLoadBalance.fairness_index.toFixed(3)}
                    </span>
                  </div>
                  
                  <div className="w-full bg-devtools-background rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-300 ${
                        latestLoadBalance.fairness_index > 0.8 
                          ? 'bg-devtools-success'
                          : latestLoadBalance.fairness_index > 0.6
                          ? 'bg-devtools-warning'
                          : 'bg-devtools-error'
                      }`}
                      style={{ 
                        width: `${latestLoadBalance.fairness_index * 100}%` 
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="metric-card">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-devtools-textSecondary">
                    Coefficient of Variation
                  </span>
                  <span className="text-sm font-semibold text-devtools-text">
                    {latestLoadBalance.coefficient_of_variation.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Expert Load Distribution */}
          {latestLoadBalance && latestLoadBalance.expert_loads.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-devtools-textSecondary uppercase">
                Load Distribution
              </h4>
              
              <div className="space-y-1">
                {latestLoadBalance.expert_loads.map((load, expertId) => (
                  <div key={expertId} className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full bg-expert-${expertId % 8}`} />
                    <span className="text-xs text-devtools-textSecondary w-8">
                      E{expertId}
                    </span>
                    <div className="flex-1 bg-devtools-background rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-300 bg-expert-${expertId % 8}`}
                        style={{ 
                          width: `${Math.min(100, (load / Math.max(...latestLoadBalance.expert_loads)) * 100)}%`
                        }}
                      />
                    </div>
                    <span className="text-xs text-devtools-text w-8 text-right">
                      {load}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top Performing Experts */}
          {expertMetrics.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-devtools-textSecondary uppercase">
                Top Experts
              </h4>
              
              <div className="space-y-1">
                {expertMetrics
                  .sort((a, b) => b.activation_frequency - a.activation_frequency)
                  .slice(0, 3)
                  .map((expert) => (
                    <div key={`${expert.layer_idx}-${expert.expert_id}`} className="flex items-center justify-between text-xs">
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full bg-expert-${expert.expert_id % 8}`} />
                        <span className="text-devtools-textSecondary">
                          L{expert.layer_idx} E{expert.expert_id}
                        </span>
                      </div>
                      <div className="text-devtools-text">
                        {expert.activation_frequency.toFixed(1)}%
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Show button when minimized */}
      {isMinimized && (
        <div className="p-2">
          <div className="flex items-center space-x-2 text-sm">
            <BarChart3 size={14} />
            <span>{activeExperts}/{totalExperts} active</span>
            {deadExperts > 0 && (
              <span className="text-devtools-error">
                {deadExperts} dead
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}