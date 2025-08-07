'use client';

import { useState } from 'react';
import { useDebuggerStore } from '@/store/debugger';
import { 
  Code2, 
  Layers, 
  Zap, 
  Search,
  ChevronRight,
  ChevronDown,
  Box,
  Hash,
  Activity
} from 'lucide-react';

export function ElementsPanel() {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['model']));
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  
  const { currentSession, expertMetrics } = useDebuggerStore();

  const toggleNode = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  // Mock model structure - in real implementation this would come from the backend
  const modelStructure = {
    id: 'model',
    name: currentSession?.model_name || 'Mixtral-8x7B',
    type: 'MixtralForCausalLM',
    children: [
      {
        id: 'embed_tokens',
        name: 'embed_tokens',
        type: 'Embedding',
        params: '32,000 × 4,096',
        children: []
      },
      {
        id: 'layers',
        name: 'layers',
        type: 'ModuleList',
        children: Array.from({ length: currentSession?.num_layers || 32 }, (_, i) => ({
          id: `layer_${i}`,
          name: `layers.${i}`,
          type: 'MixtralDecoderLayer',
          children: [
            {
              id: `layer_${i}_self_attn`,
              name: 'self_attn',
              type: 'MixtralAttention',
              params: '4,096 → 4,096',
              children: []
            },
            {
              id: `layer_${i}_block_sparse_moe`,
              name: 'block_sparse_moe',
              type: 'MixtralSparseMoeBlock',
              children: [
                {
                  id: `layer_${i}_gate`,
                  name: 'gate',
                  type: 'Linear',
                  params: '4,096 → 8',
                  children: []
                },
                {
                  id: `layer_${i}_experts`,
                  name: 'experts',
                  type: 'ModuleList',
                  children: Array.from({ length: currentSession?.num_experts || 8 }, (_, j) => ({
                    id: `layer_${i}_expert_${j}`,
                    name: `experts.${j}`,
                    type: 'MixtralMLP',
                    params: '4,096 → 14,336 → 4,096',
                    children: []
                  }))
                }
              ]
            }
          ]
        }))
      },
      {
        id: 'norm',
        name: 'norm',
        type: 'MixtralRMSNorm',
        params: '4,096',
        children: []
      },
      {
        id: 'lm_head',
        name: 'lm_head',
        type: 'Linear',
        params: '4,096 → 32,000',
        children: []
      }
    ]
  };

  const getExpertMetrics = (layerIdx: number, expertId: number) => {
    return expertMetrics.find(m => m.layer_idx === layerIdx && m.expert_id === expertId);
  };

  const renderTreeNode = (node: any, depth = 0) => {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedNodes.has(node.id);
    const isSelected = selectedNode === node.id;
    
    // Filter based on search
    if (searchQuery && !node.name.toLowerCase().includes(searchQuery.toLowerCase())) {
      return null;
    }

    // Get expert metrics if this is an expert node
    const expertMatch = node.id.match(/layer_(\d+)_expert_(\d+)/);
    const expertMetric = expertMatch 
      ? getExpertMetrics(parseInt(expertMatch[1]), parseInt(expertMatch[2]))
      : null;

    return (
      <div key={node.id}>
        <div
          className={`flex items-center space-x-2 py-1 px-2 rounded cursor-pointer transition-colors ${
            isSelected 
              ? 'bg-devtools-accent text-white' 
              : 'hover:bg-devtools-background'
          }`}
          style={{ paddingLeft: `${depth * 20 + 8}px` }}
          onClick={() => setSelectedNode(isSelected ? null : node.id)}
        >
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleNode(node.id);
              }}
              className="p-0.5 rounded hover:bg-devtools-border transition-colors"
            >
              {isExpanded ? 
                <ChevronDown size={14} /> : 
                <ChevronRight size={14} />
              }
            </button>
          )}
          
          {!hasChildren && <div className="w-5" />}
          
          <div className="flex items-center space-x-2 flex-1 min-w-0">
            {node.type === 'MixtralSparseMoeBlock' ? (
              <Zap size={14} className="text-devtools-accent flex-shrink-0" />
            ) : node.type === 'MixtralMLP' ? (
              <Box size={14} className="text-devtools-success flex-shrink-0" />
            ) : node.type.includes('Attention') ? (
              <Activity size={14} className="text-devtools-warning flex-shrink-0" />
            ) : (
              <Layers size={14} className="text-devtools-textSecondary flex-shrink-0" />
            )}
            
            <span className="font-mono text-sm truncate">
              {node.name}
            </span>
            
            {node.type && (
              <span className="text-xs text-devtools-textSecondary">
                {node.type}
              </span>
            )}
          </div>
          
          {expertMetric && (
            <div className="flex items-center space-x-2 flex-shrink-0">
              <div className={`w-2 h-2 rounded-full ${
                expertMetric.activation_frequency > 10 
                  ? 'bg-devtools-success'
                  : expertMetric.activation_frequency > 1
                  ? 'bg-devtools-warning'
                  : 'bg-devtools-error'
              }`} />
              <span className="text-xs text-devtools-textSecondary">
                {expertMetric.activation_frequency.toFixed(1)}%
              </span>
            </div>
          )}
        </div>
        
        {hasChildren && isExpanded && (
          <div>
            {node.children.map((child: any) => renderTreeNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const renderDetails = () => {
    if (!selectedNode) {
      return (
        <div className="flex items-center justify-center h-full text-devtools-textSecondary">
          <div className="text-center space-y-2">
            <Code2 size={32} />
            <p>Select a component to view details</p>
          </div>
        </div>
      );
    }

    // Find the selected node in the structure
    const findNode = (node: any): any => {
      if (node.id === selectedNode) return node;
      if (node.children) {
        for (const child of node.children) {
          const found = findNode(child);
          if (found) return found;
        }
      }
      return null;
    };

    const node = findNode(modelStructure);
    if (!node) return null;

    // Get expert metrics if this is an expert node
    const expertMatch = node.id.match(/layer_(\d+)_expert_(\d+)/);
    const expertMetric = expertMatch 
      ? getExpertMetrics(parseInt(expertMatch[1]), parseInt(expertMatch[2]))
      : null;

    return (
      <div className="p-4 space-y-4">
        <div className="border-b border-devtools-border pb-4">
          <h3 className="text-lg font-semibold text-devtools-text mb-2">
            {node.name}
          </h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-2 text-sm">
              <span className="text-devtools-textSecondary">Type:</span>
              <span className="font-mono text-devtools-text">{node.type}</span>
            </div>
            {node.params && (
              <div className="flex items-center space-x-2 text-sm">
                <span className="text-devtools-textSecondary">Parameters:</span>
                <span className="font-mono text-devtools-text">{node.params}</span>
              </div>
            )}
          </div>
        </div>

        {expertMetric && (
          <div className="space-y-3">
            <h4 className="font-semibold text-devtools-text">Expert Metrics</h4>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="metric-card">
                <div className="text-xs text-devtools-textSecondary mb-1">
                  Activation Frequency
                </div>
                <div className="text-lg font-bold text-devtools-text">
                  {expertMetric.activation_frequency.toFixed(2)}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="text-xs text-devtools-textSecondary mb-1">
                  Tokens Processed
                </div>
                <div className="text-lg font-bold text-devtools-text">
                  {expertMetric.total_tokens_processed.toLocaleString()}
                </div>
              </div>
              
              <div className="metric-card">
                <div className="text-xs text-devtools-textSecondary mb-1">
                  Avg Confidence
                </div>
                <div className="text-lg font-bold text-devtools-text">
                  {(expertMetric.average_confidence * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="metric-card">
                <div className="text-xs text-devtools-textSecondary mb-1">
                  Weight Magnitude
                </div>
                <div className="text-lg font-bold text-devtools-text">
                  {expertMetric.weight_magnitude.toFixed(4)}
                </div>
              </div>
            </div>

            <div className="text-xs text-devtools-textSecondary">
              Last active: {new Date(expertMetric.last_active_timestamp).toLocaleString()}
            </div>
          </div>
        )}

        {node.children && node.children.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-semibold text-devtools-text">Children</h4>
            <div className="text-sm text-devtools-textSecondary">
              {node.children.length} child components
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-full bg-devtools-background">
      {/* Tree View */}
      <div className="w-1/2 border-r border-devtools-border bg-devtools-surface">
        {/* Search Bar */}
        <div className="p-3 border-b border-devtools-border">
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-devtools-textSecondary" />
            <input
              type="text"
              placeholder="Search components..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="devtools-input w-full pl-10"
            />
          </div>
        </div>
        
        {/* Tree */}
        <div className="overflow-y-auto custom-scrollbar h-full p-2">
          {renderTreeNode(modelStructure)}
        </div>
      </div>
      
      {/* Details View */}
      <div className="w-1/2 bg-devtools-background overflow-y-auto custom-scrollbar">
        {renderDetails()}
      </div>
    </div>
  );
}