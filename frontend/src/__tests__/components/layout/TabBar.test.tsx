import { render, screen, fireEvent } from '@testing-library/react'
import { TabBar } from '@/components/layout/TabBar'
import { useDebuggerStore } from '@/store/debugger'

// Mock the store
const mockStore = {
  activePanel: 'network' as const,
  setActivePanel: jest.fn(),
}

jest.mock('@/store/debugger', () => ({
  useDebuggerStore: () => mockStore,
}))

beforeEach(() => {
  jest.clearAllMocks()
})

describe('TabBar', () => {
  it('renders all navigation tabs', () => {
    render(<TabBar />)
    
    expect(screen.getByText('Network')).toBeInTheDocument()
    expect(screen.getByText('Elements')).toBeInTheDocument()
    expect(screen.getByText('Console')).toBeInTheDocument()
    expect(screen.getByText('Performance')).toBeInTheDocument()
  })

  it('highlights the active panel', () => {
    render(<TabBar />)
    
    const networkTab = screen.getByText('Network').closest('button')
    expect(networkTab).toHaveClass('active')
  })

  it('calls setActivePanel when tab is clicked', () => {
    render(<TabBar />)
    
    const elementsTab = screen.getByText('Elements')
    fireEvent.click(elementsTab)
    
    expect(mockStore.setActivePanel).toHaveBeenCalledWith('elements')
  })

  it('shows correct tooltips', () => {
    render(<TabBar />)
    
    const networkTab = screen.getByText('Network').closest('button')
    expect(networkTab).toHaveAttribute('title', 'Expert routing visualization')
  })

  it('renders settings and help buttons', () => {
    render(<TabBar />)
    
    expect(screen.getByTitle('Settings')).toBeInTheDocument()
    expect(screen.getByTitle('Help')).toBeInTheDocument()
  })
})