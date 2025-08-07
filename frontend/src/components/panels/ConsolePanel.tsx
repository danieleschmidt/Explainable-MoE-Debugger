'use client';

import { useState, useRef, useEffect } from 'react';
import { useDebuggerStore } from '@/store/debugger';
import { useWebSocket } from '@/lib/websocket';
import { 
  Terminal, 
  Send, 
  Trash2,
  History,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react';

interface ConsoleMessage {
  id: string;
  type: 'input' | 'output' | 'error' | 'info';
  content: string;
  timestamp: Date;
}

const EXAMPLE_COMMANDS = [
  'moe.get_expert_stats()',
  'moe.analyze_routing_patterns()',
  'moe.detect_dead_experts()',
  'moe.get_load_balance_metrics()',
  'moe.trace_token_routing("hello")',
  'moe.set_expert_capacity(2.0)',
  'moe.force_expert([0, 3, 7])',
  'session.export_data()',
];

export function ConsolePanel() {
  const [messages, setMessages] = useState<ConsoleMessage[]>([
    {
      id: '0',
      type: 'info',
      content: 'MoE Debugger Console v0.1.0-alpha\nType help() for available commands.',
      timestamp: new Date(),
    }
  ]);
  
  const [currentInput, setCurrentInput] = useState('');
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [isExecuting, setIsExecuting] = useState(false);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { send } = useWebSocket();

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const addMessage = (type: ConsoleMessage['type'], content: string) => {
    const message: ConsoleMessage = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, message]);
  };

  const executeCommand = async (command: string) => {
    if (!command.trim()) return;

    // Add input to messages
    addMessage('input', command);
    
    // Add to history
    setCommandHistory(prev => [...prev, command]);
    setHistoryIndex(-1);
    setCurrentInput('');
    setIsExecuting(true);

    try {
      // Handle built-in commands
      if (command === 'help()' || command === 'help') {
        const helpText = `Available Commands:
        
• moe.get_expert_stats() - Get expert utilization statistics
• moe.analyze_routing_patterns() - Analyze token routing patterns  
• moe.detect_dead_experts() - Find unused experts
• moe.get_load_balance_metrics() - Get load balancing metrics
• moe.trace_token_routing(token) - Trace specific token routing
• moe.set_expert_capacity(capacity) - Set expert capacity
• moe.force_expert(expert_ids) - Force specific experts
• session.export_data() - Export session data
• clear - Clear console
• help() - Show this help`;
        
        addMessage('output', helpText);
      } else if (command === 'clear') {
        setMessages([{
          id: Date.now().toString(),
          type: 'info',
          content: 'Console cleared.',
          timestamp: new Date(),
        }]);
      } else {
        // Send command to backend
        send({
          type: 'console_command',
          data: {
            command,
            timestamp: new Date().toISOString(),
          }
        });

        // Mock response for demo - in real implementation this would come from WebSocket
        setTimeout(() => {
          if (command.startsWith('moe.get_expert_stats')) {
            const mockStats = {
              total_experts: 8,
              active_experts: 6,
              dead_experts: [1, 7],
              avg_utilization: 12.5,
              max_utilization: 25.3,
              min_utilization: 0.0
            };
            addMessage('output', JSON.stringify(mockStats, null, 2));
          } else if (command.startsWith('moe.detect_dead_experts')) {
            const mockResult = {
              dead_experts: [1, 7],
              suggestions: [
                'Consider reducing number of experts',
                'Check routing temperature',
                'Verify training data diversity'
              ]
            };
            addMessage('output', JSON.stringify(mockResult, null, 2));
          } else if (command.startsWith('moe.trace_token_routing')) {
            const token = command.match(/["'](.*?)["']/)?.[1] || 'token';
            const mockTrace = {
              token,
              layer_0: { experts: [2, 5], weights: [0.73, 0.27] },
              layer_1: { experts: [0, 4], weights: [0.61, 0.39] },
              layer_2: { experts: [3, 6], weights: [0.84, 0.16] }
            };
            addMessage('output', JSON.stringify(mockTrace, null, 2));
          } else if (command.includes('export_data')) {
            addMessage('output', 'Data exported successfully to moe-debug-data.json');
          } else {
            // Generic response for unknown commands
            addMessage('output', `Executed: ${command}\nResult: OK`);
          }
          setIsExecuting(false);
        }, 500);
      }
    } catch (error) {
      addMessage('error', `Error: ${error}`);
      setIsExecuting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      executeCommand(currentInput);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0) {
        const newIndex = historyIndex === -1 
          ? commandHistory.length - 1 
          : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setCurrentInput(commandHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex >= 0) {
        const newIndex = historyIndex + 1;
        if (newIndex >= commandHistory.length) {
          setHistoryIndex(-1);
          setCurrentInput('');
        } else {
          setHistoryIndex(newIndex);
          setCurrentInput(commandHistory[newIndex]);
        }
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      // Simple autocomplete - find matching commands
      const matches = EXAMPLE_COMMANDS.filter(cmd => 
        cmd.startsWith(currentInput.toLowerCase())
      );
      if (matches.length === 1) {
        setCurrentInput(matches[0]);
      }
    }
  };

  const getMessageIcon = (type: ConsoleMessage['type']) => {
    switch (type) {
      case 'input':
        return <ChevronRight size={14} className="text-devtools-accent flex-shrink-0" />;
      case 'error':
        return <AlertCircle size={14} className="text-devtools-error flex-shrink-0" />;
      case 'output':
        return <CheckCircle size={14} className="text-devtools-success flex-shrink-0" />;
      case 'info':
        return <Info size={14} className="text-devtools-textSecondary flex-shrink-0" />;
      default:
        return null;
    }
  };

  const getMessageStyle = (type: ConsoleMessage['type']) => {
    switch (type) {
      case 'input':
        return 'text-devtools-text font-semibold';
      case 'error':
        return 'text-devtools-error';
      case 'output':
        return 'text-devtools-text';
      case 'info':
        return 'text-devtools-textSecondary';
      default:
        return 'text-devtools-text';
    }
  };

  return (
    <div className="flex flex-col h-full bg-devtools-background">
      {/* Toolbar */}
      <div className="bg-devtools-surface border-b border-devtools-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Terminal size={16} />
            <span className="font-semibold text-devtools-text">Console</span>
            <span className="text-xs bg-devtools-background px-2 py-1 rounded">
              Interactive
            </span>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setMessages(prev => prev.slice(0, 1))}
              className="devtools-button-secondary flex items-center space-x-2 text-sm"
              title="Clear console"
            >
              <Trash2 size={14} />
              <span>Clear</span>
            </button>
            
            <button
              className="devtools-button-secondary flex items-center space-x-2 text-sm"
              title="Command history"
            >
              <History size={14} />
              <span>History ({commandHistory.length})</span>
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
        <div className="space-y-3">
          {messages.map((message) => (
            <div key={message.id} className="flex space-x-3">
              <div className="pt-0.5">
                {getMessageIcon(message.type)}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className={`font-mono text-sm whitespace-pre-wrap ${getMessageStyle(message.type)}`}>
                  {message.content}
                </div>
                <div className="text-xs text-devtools-textSecondary mt-1">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          
          {isExecuting && (
            <div className="flex space-x-3">
              <div className="pt-0.5">
                <div className="w-3 h-3 border-2 border-devtools-accent border-t-transparent rounded-full animate-spin" />
              </div>
              <div className="text-sm text-devtools-textSecondary">
                Executing command...
              </div>
            </div>
          )}
        </div>
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-devtools-surface border-t border-devtools-border p-3">
        <div className="flex items-center space-x-3">
          <ChevronRight size={16} className="text-devtools-accent" />
          
          <input
            ref={inputRef}
            type="text"
            value={currentInput}
            onChange={(e) => setCurrentInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter command... (Tab for autocomplete, ↑↓ for history)"
            className="flex-1 bg-transparent border-none outline-none text-devtools-text font-mono text-sm placeholder-devtools-textSecondary"
            disabled={isExecuting}
          />
          
          <button
            onClick={() => executeCommand(currentInput)}
            disabled={isExecuting || !currentInput.trim()}
            className="devtools-button-secondary flex items-center space-x-2 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={14} />
            <span>Execute</span>
          </button>
        </div>
        
        {/* Quick Commands */}
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLE_COMMANDS.slice(0, 4).map((cmd) => (
            <button
              key={cmd}
              onClick={() => setCurrentInput(cmd)}
              className="text-xs bg-devtools-background hover:bg-devtools-border px-2 py-1 rounded transition-colors text-devtools-textSecondary hover:text-devtools-text"
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}