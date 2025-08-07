'use client';

import { Brain, Loader2 } from 'lucide-react';

export function LoadingScreen() {
  return (
    <div className="h-screen bg-devtools-background flex items-center justify-center">
      <div className="text-center space-y-6">
        {/* Logo Animation */}
        <div className="relative">
          <div className="w-20 h-20 mx-auto bg-devtools-accent rounded-lg flex items-center justify-center">
            <Brain size={40} className="text-white" />
          </div>
          <div className="absolute -inset-2 border-2 border-devtools-accent rounded-lg animate-ping opacity-30"></div>
        </div>

        {/* Loading Text */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-devtools-text">
            MoE Debugger
          </h1>
          <p className="text-devtools-textSecondary">
            Explainable Mixture of Experts
          </p>
        </div>

        {/* Loading Animation */}
        <div className="flex items-center justify-center space-x-2">
          <Loader2 size={20} className="animate-spin text-devtools-accent" />
          <span className="text-sm text-devtools-textSecondary">
            Initializing debugger...
          </span>
        </div>

        {/* Loading Steps */}
        <div className="space-y-2 text-xs text-devtools-textSecondary">
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-devtools-success animate-pulse"></div>
            <span>Backend connection established</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-devtools-success animate-pulse delay-200"></div>
            <span>WebSocket initialized</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-devtools-accent animate-pulse delay-400"></div>
            <span>Loading visualization engine...</span>
          </div>
        </div>
      </div>
    </div>
  );
}