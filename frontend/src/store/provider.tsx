'use client';

import { ReactNode } from 'react';

interface StoreProviderProps {
  children: ReactNode;
}

export function StoreProvider({ children }: StoreProviderProps) {
  // The Zustand store is already created and accessible via the hook
  // This provider is mainly for future extensibility or additional providers
  return <>{children}</>;
}