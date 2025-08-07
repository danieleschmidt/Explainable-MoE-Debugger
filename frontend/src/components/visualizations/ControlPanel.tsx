'use client';

import { useState } from 'react';
import { useDebuggerStore } from '@/store/debugger';
import { 
  Settings, 
  Eye, 
  EyeOff, 
  Palette, 
  Sliders,
  Filter,
  Clock,
  Hash,
  ChevronDown,
  ChevronRight
} from 'lucide-react';

export function ControlPanel() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['display', 'filters', 'animation'])
  );
  
  const {
    visualizationConfig,
    filterConfig,
    updateVisualizationConfig,
    updateFilterConfig,
    currentSession
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

  return (
    <div className="h-full bg-devtools-surface custom-scrollbar overflow-y-auto">
      <div className="p-4">
        <div className="flex items-center space-x-2 mb-6">
          <Settings size={18} />
          <h3 className="text-lg font-semibold text-devtools-text">Controls</h3>
        </div>

        {/* Display Settings */}
        <div className="mb-6">
          <button
            onClick={() => toggleSection('display')}
            className="w-full flex items-center justify-between p-2 hover:bg-devtools-background rounded transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Eye size={16} />
              <span className="font-medium">Display</span>
            </div>
            {expandedSections.has('display') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>

          {expandedSections.has('display') && (
            <div className="mt-3 space-y-4 pl-4">
              {/* Show Token Labels */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-devtools-text">Token Labels</label>
                <button
                  onClick={() => updateVisualizationConfig({
                    showTokenLabels: !visualizationConfig.showTokenLabels
                  })}
                  className={`p-2 rounded transition-colors ${
                    visualizationConfig.showTokenLabels 
                      ? 'bg-devtools-accent text-white' 
                      : 'bg-devtools-background'
                  }`}
                >
                  {visualizationConfig.showTokenLabels ? 
                    <Eye size={14} /> : 
                    <EyeOff size={14} />
                  }
                </button>
              </div>

              {/* Show Confidence Scores */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-devtools-text">Confidence Scores</label>
                <button
                  onClick={() => updateVisualizationConfig({
                    showConfidenceScores: !visualizationConfig.showConfidenceScores
                  })}
                  className={`p-2 rounded transition-colors ${
                    visualizationConfig.showConfidenceScores 
                      ? 'bg-devtools-accent text-white' 
                      : 'bg-devtools-background'
                  }`}
                >
                  {visualizationConfig.showConfidenceScores ? 
                    <Hash size={14} /> : 
                    <Hash size={14} className="opacity-50" />
                  }
                </button>
              </div>

              {/* Expert Color Scheme */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">Color Scheme</label>
                <select
                  value={visualizationConfig.expertColorScheme}
                  onChange={(e) => updateVisualizationConfig({
                    expertColorScheme: e.target.value as any
                  })}
                  className="devtools-input w-full text-sm"
                >
                  <option value="categorical">Categorical</option>
                  <option value="heatmap">Heatmap</option>
                  <option value="default">Default</option>
                </select>
              </div>

              {/* Max Tokens Display */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">
                  Max Tokens: {visualizationConfig.maxTokensDisplay}
                </label>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="10"
                  value={visualizationConfig.maxTokensDisplay}
                  onChange={(e) => updateVisualizationConfig({
                    maxTokensDisplay: parseInt(e.target.value)
                  })}
                  className="w-full accent-devtools-accent"
                />
              </div>
            </div>
          )}
        </div>

        {/* Filter Settings */}
        <div className="mb-6">
          <button
            onClick={() => toggleSection('filters')}
            className="w-full flex items-center justify-between p-2 hover:bg-devtools-background rounded transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Filter size={16} />
              <span className="font-medium">Filters</span>
            </div>
            {expandedSections.has('filters') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>

          {expandedSections.has('filters') && (
            <div className="mt-3 space-y-4 pl-4">
              {/* Layer Range */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">
                  Layer Range: {filterConfig.layerRange[0]} - {filterConfig.layerRange[1]}
                </label>
                <div className="space-y-2">
                  <div className="flex space-x-2">
                    <input
                      type="number"
                      min="0"
                      max={currentSession?.num_layers || 32}
                      value={filterConfig.layerRange[0]}
                      onChange={(e) => updateFilterConfig({
                        layerRange: [parseInt(e.target.value), filterConfig.layerRange[1]]
                      })}
                      className="devtools-input flex-1 text-sm"
                      placeholder="Min"
                    />
                    <input
                      type="number"
                      min="0"
                      max={currentSession?.num_layers || 32}
                      value={filterConfig.layerRange[1]}
                      onChange={(e) => updateFilterConfig({
                        layerRange: [filterConfig.layerRange[0], parseInt(e.target.value)]
                      })}
                      className="devtools-input flex-1 text-sm"
                      placeholder="Max"
                    />
                  </div>
                </div>
              </div>

              {/* Expert Range */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">
                  Expert Range: {filterConfig.expertRange[0]} - {filterConfig.expertRange[1]}
                </label>
                <div className="flex space-x-2">
                  <input
                    type="number"
                    min="0"
                    max={currentSession?.num_experts || 8}
                    value={filterConfig.expertRange[0]}
                    onChange={(e) => updateFilterConfig({
                      expertRange: [parseInt(e.target.value), filterConfig.expertRange[1]]
                    })}
                    className="devtools-input flex-1 text-sm"
                    placeholder="Min"
                  />
                  <input
                    type="number"
                    min="0"
                    max={currentSession?.num_experts || 8}
                    value={filterConfig.expertRange[1]}
                    onChange={(e) => updateFilterConfig({
                      expertRange: [filterConfig.expertRange[0], parseInt(e.target.value)]
                    })}
                    className="devtools-input flex-1 text-sm"
                    placeholder="Max"
                  />
                </div>
              </div>

              {/* Confidence Threshold */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">
                  Confidence Threshold: {filterConfig.confidenceThreshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={filterConfig.confidenceThreshold}
                  onChange={(e) => updateFilterConfig({
                    confidenceThreshold: parseFloat(e.target.value)
                  })}
                  className="w-full accent-devtools-accent"
                />
              </div>

              {/* Show Only Active Experts */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-devtools-text">Active Experts Only</label>
                <button
                  onClick={() => updateFilterConfig({
                    showOnlyActiveExperts: !filterConfig.showOnlyActiveExperts
                  })}
                  className={`p-2 rounded transition-colors ${
                    filterConfig.showOnlyActiveExperts 
                      ? 'bg-devtools-accent text-white' 
                      : 'bg-devtools-background'
                  }`}
                >
                  {filterConfig.showOnlyActiveExperts ? 
                    <Eye size={14} /> : 
                    <EyeOff size={14} />
                  }
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Animation Settings */}
        <div className="mb-6">
          <button
            onClick={() => toggleSection('animation')}
            className="w-full flex items-center justify-between p-2 hover:bg-devtools-background rounded transition-colors"
          >
            <div className="flex items-center space-x-2">
              <Clock size={16} />
              <span className="font-medium">Animation</span>
            </div>
            {expandedSections.has('animation') ? 
              <ChevronDown size={16} /> : 
              <ChevronRight size={16} />
            }
          </button>

          {expandedSections.has('animation') && (
            <div className="mt-3 space-y-4 pl-4">
              {/* Animate Routing */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-devtools-text">Animate Routing</label>
                <button
                  onClick={() => updateVisualizationConfig({
                    animateRouting: !visualizationConfig.animateRouting
                  })}
                  className={`p-2 rounded transition-colors ${
                    visualizationConfig.animateRouting 
                      ? 'bg-devtools-accent text-white' 
                      : 'bg-devtools-background'
                  }`}
                >
                  <Clock size={14} />
                </button>
              </div>

              {/* Update Interval */}
              <div>
                <label className="text-sm text-devtools-text mb-2 block">
                  Update Interval: {visualizationConfig.updateInterval}ms
                </label>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={visualizationConfig.updateInterval}
                  onChange={(e) => updateVisualizationConfig({
                    updateInterval: parseInt(e.target.value)
                  })}
                  className="w-full accent-devtools-accent"
                />
              </div>
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-devtools-text">Quick Actions</h4>
          
          <button
            onClick={() => {
              updateFilterConfig({
                layerRange: [0, currentSession?.num_layers || 32],
                expertRange: [0, currentSession?.num_experts || 8],
                confidenceThreshold: 0.0,
                showOnlyActiveExperts: false,
                selectedTokens: []
              });
            }}
            className="devtools-button-secondary w-full text-sm"
          >
            Reset Filters
          </button>
          
          <button
            onClick={() => {
              updateVisualizationConfig({
                showTokenLabels: true,
                showConfidenceScores: true,
                animateRouting: true,
                expertColorScheme: 'categorical',
                maxTokensDisplay: 50,
                updateInterval: 100
              });
            }}
            className="devtools-button-secondary w-full text-sm"
          >
            Reset Display
          </button>
        </div>
      </div>
    </div>
  );
}