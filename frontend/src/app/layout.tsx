import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';
import { Toaster } from 'react-hot-toast';
import { StoreProvider } from '@/store/provider';

const inter = Inter({ subsets: ['latin'] });
const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-mono',
});

export const metadata: Metadata = {
  title: 'MoE Debugger - Explainable Mixture of Experts',
  description: 'Chrome DevTools-style GUI for debugging Mixture of Experts models',
  keywords: ['machine learning', 'mixture of experts', 'debugging', 'visualization'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.className} ${jetbrainsMono.variable}`}>
      <body className="bg-devtools-background text-devtools-text">
        <StoreProvider>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                background: '#2d2d2d',
                color: '#cccccc',
                border: '1px solid #3c3c3c',
              },
              success: {
                iconTheme: {
                  primary: '#00d084',
                  secondary: '#2d2d2d',
                },
              },
              error: {
                iconTheme: {
                  primary: '#f85149',
                  secondary: '#2d2d2d',
                },
              },
            }}
          />
        </StoreProvider>
      </body>
    </html>
  );
}