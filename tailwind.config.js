/** @type {import('tailwindcss').Config} */
export default {
    content: ['./src/**/*.{html,js}'],
    theme: {
        extend: {
            colors: {
                'deep-space': '#0b0f1a',
                'deep-space-light': '#111827',
                'neon-cyan': '#00f0ff',
                'neon-blue': '#3b82f6',
                'neon-violet': '#8b5cf6',
                'neon-pink': '#ec4899',
                'glass-white': 'rgba(255, 255, 255, 0.05)',
                'glass-border': 'rgba(255, 255, 255, 0.1)',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                display: ['"Space Grotesk"', 'system-ui', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
            },
            boxShadow: {
                'neon': '0 0 5px rgba(0, 240, 255, 0.3), 0 0 20px rgba(0, 240, 255, 0.1)',
                'neon-strong': '0 0 5px rgba(0, 240, 255, 0.5), 0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(0, 240, 255, 0.15)',
                'neon-violet': '0 0 5px rgba(139, 92, 246, 0.3), 0 0 20px rgba(139, 92, 246, 0.1)',
                'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
                'glass-lg': '0 16px 48px rgba(0, 0, 0, 0.4)',
                'depth': '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
            },
            borderRadius: {
                '2xl': '16px',
                '3xl': '24px',
            },
            animation: {
                'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
                'float': 'float 6s ease-in-out infinite',
                'gradient-shift': 'gradient-shift 8s ease infinite',
                'typing-dot': 'typing-dot 1.4s ease-in-out infinite',
                'slide-up': 'slide-up 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
                'fade-in': 'fade-in 0.6s ease-out forwards',
                'scan-line': 'scan-line 3s linear infinite',
            },
            keyframes: {
                'glow-pulse': {
                    '0%, 100%': { boxShadow: '0 0 5px rgba(0,240,255,0.4), 0 0 20px rgba(0,240,255,0.2)' },
                    '50%': { boxShadow: '0 0 10px rgba(0,240,255,0.6), 0 0 40px rgba(0,240,255,0.3), 0 0 60px rgba(0,240,255,0.15)' },
                },
                'float': {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                },
                'gradient-shift': {
                    '0%, 100%': { backgroundPosition: '0% 50%' },
                    '50%': { backgroundPosition: '100% 50%' },
                },
                'typing-dot': {
                    '0%, 60%, 100%': { opacity: '0.2', transform: 'scale(0.8)' },
                    '30%': { opacity: '1', transform: 'scale(1)' },
                },
                'slide-up': {
                    '0%': { transform: 'translateY(20px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                'fade-in': {
                    '0%': { opacity: '0', transform: 'translateY(20px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                'scan-line': {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(200%)' },
                },
            },
            backgroundImage: {
                'grid-pattern': 'linear-gradient(rgba(0,240,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,240,255,0.03) 1px, transparent 1px)',
            },
            backgroundSize: {
                'grid': '60px 60px',
            },
        },
    },
    plugins: [],
};
