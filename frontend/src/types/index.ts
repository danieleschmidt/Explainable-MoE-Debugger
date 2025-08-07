// Core types for MoE Debugger frontend

export interface RoutingEvent {
  id: string;
  timestamp: string;
  session_id: string;
  layer_idx: number;
  token_idx: number;
  token_text: string;
  selected_experts: number[];
  routing_weights: number[];
  confidence_scores: number[];
  metadata: Record<string, any>;
}

export interface ExpertMetrics {
  expert_id: number;
  layer_idx: number;
  total_tokens_processed: number;
  average_confidence: number;
  activation_frequency: number;
  weight_magnitude: number;
  gradient_norm?: number;
  last_active_timestamp: string;
}

export interface LoadBalanceMetrics {
  layer_idx: number;
  expert_loads: number[];
  load_variance: number;
  coefficient_of_variation: number;
  fairness_index: number;
  dead_experts: number[];
  overloaded_experts: number[];
  timestamp: string;
}

export interface TokenAttribution {
  token_text: string;
  token_idx: number;
  expert_attributions: Record<number, number>;
  total_attribution: number;
  layer_attributions: Record<number, number>;
}

export interface DiagnosticResult {
  diagnostic_type: string;
  severity: 'info' | 'warning' | 'error';
  title: string;
  description: string;
  affected_layers: number[];
  affected_experts: number[];
  recommendations: string[];
  metadata: Record<string, any>;
  timestamp: string;
}

export interface PerformanceMetrics {
  session_id: string;
  timestamp: string;
  inference_time_ms: number;
  routing_overhead_ms: number;
  memory_usage_mb: number;
  cache_hit_rate: number;
  experts_activated: number;
  tokens_processed: number;
  throughput_tokens_per_sec: number;
}

export interface SessionInfo {
  session_id: string;
  model_name: string;
  model_architecture: string;
  num_experts: number;
  num_layers: number;
  created_at: string;
  last_active: string;
  total_tokens_processed: number;
  status: 'active' | 'paused' | 'completed';
}

export interface WebSocketMessage {
  type: 'routing_event' | 'performance_metrics' | 'diagnostic_result' | 'session_update';
  data: any;
  timestamp: string;
}

export type PanelType = 'network' | 'elements' | 'console' | 'performance';

export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';

export interface VisualizationConfig {
  showTokenLabels: boolean;
  showConfidenceScores: boolean;
  animateRouting: boolean;
  expertColorScheme: 'default' | 'heatmap' | 'categorical';
  maxTokensDisplay: number;
  updateInterval: number;
}

export interface FilterConfig {
  layerRange: [number, number];
  expertRange: [number, number];
  confidenceThreshold: number;
  showOnlyActiveExperts: boolean;
  selectedTokens: string[];
}

// D3.js visualization data structures
export interface NetworkNode {
  id: string;
  type: 'token' | 'expert';
  label: string;
  layer?: number;
  expert_id?: number;
  token_idx?: number;
  x?: number;
  y?: number;
  weight?: number;
  confidence?: number;
  color?: string;
}

export interface NetworkLink {
  id: string;
  source: string;
  target: string;
  weight: number;
  confidence: number;
  animated?: boolean;
}

export interface HeatmapCell {
  row: number;
  col: number;
  value: number;
  label: string;
  tooltip: string;
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  series: string;
}

// API Response types
export interface APIResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: string;
}

export interface SessionListResponse {
  sessions: SessionInfo[];
  total: number;
  page: number;
  page_size: number;
}

export interface AnalysisResponse {
  routing_events: RoutingEvent[];
  load_balance_metrics: LoadBalanceMetrics[];
  expert_metrics: ExpertMetrics[];
  diagnostics: DiagnosticResult[];
  performance_metrics: PerformanceMetrics[];
}