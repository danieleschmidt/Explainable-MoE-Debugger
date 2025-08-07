import { useDebuggerStore } from '@/store/debugger'
import { RoutingEvent, ExpertMetrics, DiagnosticResult } from '@/types'

// Mock routing event for testing
const mockRoutingEvent: RoutingEvent = {
  id: 'test-event-1',
  timestamp: new Date().toISOString(),
  session_id: 'test-session',
  layer_idx: 0,
  token_idx: 0,
  token_text: 'hello',
  selected_experts: [0, 2],
  routing_weights: [0.7, 0.0, 0.3, 0.0],
  confidence_scores: [0.9, 0.8],
  metadata: {}
}

const mockExpertMetrics: ExpertMetrics = {
  expert_id: 0,
  layer_idx: 0,
  total_tokens_processed: 100,
  average_confidence: 0.85,
  activation_frequency: 25.5,
  weight_magnitude: 1.2,
  last_active_timestamp: new Date().toISOString()
}

const mockDiagnostic: DiagnosticResult = {
  diagnostic_type: 'dead_expert',
  severity: 'warning',
  title: 'Dead Expert Detected',
  description: 'Expert 3 has not been activated in the last 1000 tokens',
  affected_layers: [0, 1],
  affected_experts: [3],
  recommendations: ['Consider reducing the number of experts'],
  metadata: {},
  timestamp: new Date().toISOString()
}

describe('useDebuggerStore', () => {
  beforeEach(() => {
    // Reset store state
    useDebuggerStore.getState().resetStore()
  })

  it('initializes with correct default state', () => {
    const state = useDebuggerStore.getState()
    
    expect(state.activePanel).toBe('network')
    expect(state.connectionStatus).toBe('disconnected')
    expect(state.isLoading).toBe(false)
    expect(state.routingEvents).toEqual([])
    expect(state.expertMetrics).toEqual([])
    expect(state.diagnostics).toEqual([])
  })

  it('updates active panel', () => {
    const { setActivePanel } = useDebuggerStore.getState()
    
    setActivePanel('console')
    
    expect(useDebuggerStore.getState().activePanel).toBe('console')
  })

  it('updates connection status', () => {
    const { setConnectionStatus } = useDebuggerStore.getState()
    
    setConnectionStatus('connected')
    
    expect(useDebuggerStore.getState().connectionStatus).toBe('connected')
  })

  it('adds routing events', () => {
    const { addRoutingEvent } = useDebuggerStore.getState()
    
    addRoutingEvent(mockRoutingEvent)
    
    const state = useDebuggerStore.getState()
    expect(state.routingEvents).toHaveLength(1)
    expect(state.routingEvents[0]).toEqual(mockRoutingEvent)
  })

  it('limits routing events to 1000', () => {
    const { addRoutingEvents } = useDebuggerStore.getState()
    
    // Add 1100 events
    const events = Array.from({ length: 1100 }, (_, i) => ({
      ...mockRoutingEvent,
      id: `event-${i}`
    }))
    
    addRoutingEvents(events)
    
    const state = useDebuggerStore.getState()
    expect(state.routingEvents).toHaveLength(1000)
    expect(state.routingEvents[0].id).toBe('event-100') // First 100 should be trimmed
  })

  it('updates expert metrics', () => {
    const { updateExpertMetrics } = useDebuggerStore.getState()
    
    updateExpertMetrics([mockExpertMetrics])
    
    const state = useDebuggerStore.getState()
    expect(state.expertMetrics).toHaveLength(1)
    expect(state.expertMetrics[0]).toEqual(mockExpertMetrics)
  })

  it('adds diagnostics', () => {
    const { addDiagnostic } = useDebuggerStore.getState()
    
    addDiagnostic(mockDiagnostic)
    
    const state = useDebuggerStore.getState()
    expect(state.diagnostics).toHaveLength(1)
    expect(state.diagnostics[0]).toEqual(mockDiagnostic)
  })

  it('limits diagnostics to 100', () => {
    const { addDiagnostic } = useDebuggerStore.getState()
    
    // Add 150 diagnostics
    for (let i = 0; i < 150; i++) {
      addDiagnostic({
        ...mockDiagnostic,
        title: `Diagnostic ${i}`
      })
    }
    
    const state = useDebuggerStore.getState()
    expect(state.diagnostics).toHaveLength(100)
  })

  it('updates visualization config', () => {
    const { updateVisualizationConfig } = useDebuggerStore.getState()
    
    updateVisualizationConfig({
      showTokenLabels: false,
      maxTokensDisplay: 100
    })
    
    const state = useDebuggerStore.getState()
    expect(state.visualizationConfig.showTokenLabels).toBe(false)
    expect(state.visualizationConfig.maxTokensDisplay).toBe(100)
    expect(state.visualizationConfig.animateRouting).toBe(true) // Unchanged
  })

  it('clears all data', () => {
    const store = useDebuggerStore.getState()
    
    // Add some data
    store.addRoutingEvent(mockRoutingEvent)
    store.updateExpertMetrics([mockExpertMetrics])
    store.addDiagnostic(mockDiagnostic)
    
    // Clear data
    store.clearAllData()
    
    const state = useDebuggerStore.getState()
    expect(state.routingEvents).toEqual([])
    expect(state.expertMetrics).toEqual([])
    expect(state.diagnostics).toEqual([])
  })

  it('resets store completely', () => {
    const store = useDebuggerStore.getState()
    
    // Modify state
    store.setActivePanel('console')
    store.setConnectionStatus('connected')
    store.addRoutingEvent(mockRoutingEvent)
    
    // Reset
    store.resetStore()
    
    const state = useDebuggerStore.getState()
    expect(state.activePanel).toBe('network')
    expect(state.connectionStatus).toBe('disconnected')
    expect(state.routingEvents).toEqual([])
  })
})