/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Chrome DevTools color scheme
        devtools: {
          background: '#1e1e1e',
          surface: '#2d2d2d',
          border: '#3c3c3c',
          text: '#cccccc',
          textSecondary: '#969696',
          accent: '#0078d4',
          success: '#00d084',
          warning: '#ffb000',
          error: '#f85149',
        },
        expert: {
          0: '#ff6b6b',
          1: '#4ecdc4',
          2: '#45b7d1',
          3: '#96ceb4',
          4: '#feca57',
          5: '#ff9ff3',
          6: '#54a0ff',
          7: '#5f27cd',
        },
      },
      fontFamily: {
        mono: ['Monaco', 'Menlo', 'Ubuntu Mono', 'monospace'],
      },
      animation: {
        'pulse-routing': 'pulse-routing 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slide-up 0.2s ease-out',
        'fade-in': 'fade-in 0.3s ease-in-out',
      },
      keyframes: {
        'pulse-routing': {
          '0%, 100%': {
            opacity: '1',
          },
          '50%': {
            opacity: '.3',
          },
        },
        'slide-up': {
          '0%': {
            transform: 'translateY(10px)',
            opacity: '0',
          },
          '100%': {
            transform: 'translateY(0)',
            opacity: '1',
          },
        },
        'fade-in': {
          '0%': {
            opacity: '0',
          },
          '100%': {
            opacity: '1',
          },
        },
      },
    },
  },
  plugins: [],
};