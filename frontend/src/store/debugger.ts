import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  PanelType, 
  ConnectionStatus, 
  RoutingEvent, 
  ExpertMetrics, 
  LoadBalanceMetrics, 
  DiagnosticResult,
  PerformanceMetrics,
  SessionInfo,
  VisualizationConfig,
  FilterConfig,
  WebSocketMessage
} from '@/types';

interface DebuggerState {
  // UI State
  activePanel: PanelType;
  connectionStatus: ConnectionStatus;
  isLoading: boolean;
  error: string | null;

  // Session State
  currentSession: SessionInfo | null;
  sessions: SessionInfo[];

  // Data State
  routingEvents: RoutingEvent[];
  expertMetrics: ExpertMetrics[];
  loadBalanceMetrics: LoadBalanceMetrics[];
  diagnostics: DiagnosticResult[];
  performanceMetrics: PerformanceMetrics[];

  // Configuration
  visualizationConfig: VisualizationConfig;
  filterConfig: FilterConfig;

  // Real-time Data Buffer
  realtimeBuffer: WebSocketMessage[];
  maxBufferSize: number;

  // Actions
  setActivePanel: (panel: PanelType) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  setCurrentSession: (session: SessionInfo | null) => void;
  addSession: (session: SessionInfo) => void;
  updateSession: (sessionId: string, updates: Partial<SessionInfo>) => void;
  
  addRoutingEvent: (event: RoutingEvent) => void;
  addRoutingEvents: (events: RoutingEvent[]) => void;
  updateExpertMetrics: (metrics: ExpertMetrics[]) => void;
  updateLoadBalanceMetrics: (metrics: LoadBalanceMetrics[]) => void;
  addDiagnostic: (diagnostic: DiagnosticResult) => void;
  addPerformanceMetric: (metric: PerformanceMetrics) => void;
  
  updateVisualizationConfig: (config: Partial<VisualizationConfig>) => void;
  updateFilterConfig: (config: Partial<FilterConfig>) => void;
  
  addRealtimeMessage: (message: WebSocketMessage) => void;
  clearRealtimeBuffer: () => void;
  
  clearAllData: () => void;
  resetStore: () => void;
}

const initialVisualizationConfig: VisualizationConfig = {
  showTokenLabels: true,
  showConfidenceScores: true,
  animateRouting: true,
  expertColorScheme: 'categorical',
  maxTokensDisplay: 50,
  updateInterval: 100,
};

const initialFilterConfig: FilterConfig = {
  layerRange: [0, 32],
  expertRange: [0, 8],
  confidenceThreshold: 0.1,
  showOnlyActiveExperts: false,
  selectedTokens: [],
};

export const useDebuggerStore = create<DebuggerState>()(
  devtools(
    (set, get) => ({
      // Initial State
      activePanel: 'network',
      connectionStatus: 'disconnected',
      isLoading: false,
      error: null,

      currentSession: null,
      sessions: [],

      routingEvents: [],
      expertMetrics: [],
      loadBalanceMetrics: [],
      diagnostics: [],
      performanceMetrics: [],

      visualizationConfig: initialVisualizationConfig,
      filterConfig: initialFilterConfig,

      realtimeBuffer: [],
      maxBufferSize: 1000,

      // Actions
      setActivePanel: (panel) => set({ activePanel: panel }),
      
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      setError: (error) => set({ error }),

      setCurrentSession: (session) => set({ currentSession: session }),
      
      addSession: (session) => set((state) => ({
        sessions: [...state.sessions, session]
      })),
      
      updateSession: (sessionId, updates) => set((state) => ({
        sessions: state.sessions.map(session => 
          session.session_id === sessionId 
            ? { ...session, ...updates }
            : session
        ),
        currentSession: state.currentSession?.session_id === sessionId
          ? { ...state.currentSession, ...updates }
          : state.currentSession
      })),

      addRoutingEvent: (event) => set((state) => {
        const events = [...state.routingEvents, event];
        // Keep only the last 1000 events for performance
        if (events.length > 1000) {
          events.shift();
        }
        return { routingEvents: events };
      }),

      addRoutingEvents: (events) => set((state) => {
        const allEvents = [...state.routingEvents, ...events];
        // Keep only the last 1000 events for performance
        const trimmedEvents = allEvents.slice(-1000);
        return { routingEvents: trimmedEvents };
      }),

      updateExpertMetrics: (metrics) => set({ expertMetrics: metrics }),
      
      updateLoadBalanceMetrics: (metrics) => set({ loadBalanceMetrics: metrics }),
      
      addDiagnostic: (diagnostic) => set((state) => {
        const diagnostics = [...state.diagnostics, diagnostic];
        // Keep only the last 100 diagnostics
        if (diagnostics.length > 100) {
          diagnostics.shift();
        }
        return { diagnostics };
      }),

      addPerformanceMetric: (metric) => set((state) => {
        const metrics = [...state.performanceMetrics, metric];
        // Keep only the last 200 metrics for charts
        if (metrics.length > 200) {
          metrics.shift();
        }
        return { performanceMetrics: metrics };
      }),

      updateVisualizationConfig: (config) => set((state) => ({
        visualizationConfig: { ...state.visualizationConfig, ...config }
      })),

      updateFilterConfig: (config) => set((state) => ({
        filterConfig: { ...state.filterConfig, ...config }
      })),

      addRealtimeMessage: (message) => set((state) => {
        const buffer = [...state.realtimeBuffer, message];
        // Trim buffer to max size
        if (buffer.length > state.maxBufferSize) {
          buffer.shift();
        }
        return { realtimeBuffer: buffer };
      }),

      clearRealtimeBuffer: () => set({ realtimeBuffer: [] }),

      clearAllData: () => set({
        routingEvents: [],
        expertMetrics: [],
        loadBalanceMetrics: [],
        diagnostics: [],
        performanceMetrics: [],
        realtimeBuffer: [],
        error: null,
      }),

      resetStore: () => set({
        activePanel: 'network',
        connectionStatus: 'disconnected',
        isLoading: false,
        error: null,
        currentSession: null,
        sessions: [],
        routingEvents: [],
        expertMetrics: [],
        loadBalanceMetrics: [],
        diagnostics: [],
        performanceMetrics: [],
        visualizationConfig: initialVisualizationConfig,
        filterConfig: initialFilterConfig,
        realtimeBuffer: [],
      }),
    }),
    {
      name: 'moe-debugger-store',
      partialize: (state: DebuggerState) => ({
        // Only persist UI preferences and config
        visualizationConfig: state.visualizationConfig,
        filterConfig: state.filterConfig,
        activePanel: state.activePanel,
      }),
    }
  )
);