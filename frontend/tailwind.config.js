/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Core Space Palette
        void: 'rgb(var(--color-void) / <alpha-value>)',
        space: 'rgb(var(--color-space) / <alpha-value>)',
        nebula: 'rgb(var(--color-nebula) / <alpha-value>)',
        cosmic: 'rgb(var(--color-cosmic) / <alpha-value>)',
        
        // Stellar Accents
        stellar: {
          DEFAULT: 'rgb(var(--color-stellar) / <alpha-value>)',
          light: 'rgb(var(--color-nova) / <alpha-value>)',
        },
        nova: 'rgb(var(--color-nova) / <alpha-value>)',
        aurora: 'rgb(var(--color-aurora) / <alpha-value>)',
        plasma: 'rgb(var(--color-plasma) / <alpha-value>)',
        
        // Dark theme palette
        slate: {
          850: '#1a1f2e',
          900: '#0f1219',
          950: '#080b10',
        },
        
        // Status colors
        success: 'rgb(var(--color-success) / <alpha-value>)',
        warning: 'rgb(var(--color-warning) / <alpha-value>)',
        danger: 'rgb(var(--color-danger) / <alpha-value>)',
        info: 'rgb(var(--color-info) / <alpha-value>)',
        
        // Severity colors (unified with stellar theme)
        severity: {
          critical: 'rgb(var(--color-danger) / <alpha-value>)',
          high: 'rgb(var(--color-warning) / <alpha-value>)',
          medium: 'rgb(245 158 11 / <alpha-value>)', // amber-500
          low: 'rgb(var(--color-aurora) / <alpha-value>)',
        },
        
        // Text hierarchy
        'text-primary': 'rgb(var(--color-text-primary) / <alpha-value>)',
        'text-secondary': 'rgb(var(--color-text-secondary) / <alpha-value>)',
        'text-tertiary': 'rgb(var(--color-text-tertiary) / <alpha-value>)',
        
        // Borders
        border: 'rgb(var(--color-border) / <alpha-value>)',
        'border-subtle': 'rgb(var(--color-border-subtle) / <alpha-value>)',
      },
      
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        display: ['Outfit', 'system-ui', 'sans-serif'],
      },
      
      // Spacing scale extensions
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      
      // Border radius
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      
      // Box shadows
      boxShadow: {
        'stellar': '0 0 20px rgba(245, 166, 35, 0.2), 0 0 40px rgba(245, 166, 35, 0.1)',
        'stellar-lg': '0 0 30px rgba(245, 166, 35, 0.3), 0 0 60px rgba(245, 166, 35, 0.15)',
        'aurora': '0 0 20px rgba(0, 217, 192, 0.2), 0 0 40px rgba(0, 217, 192, 0.1)',
        'danger': '0 0 20px rgba(255, 71, 87, 0.3), 0 0 40px rgba(255, 71, 87, 0.15)',
        'glow': '0 4px 24px -4px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03)',
        'card': '0 8px 32px -8px rgba(0, 0, 0, 0.5)',
        'card-hover': '0 12px 48px -12px rgba(245, 166, 35, 0.2)',
      },
      
      // Background gradients
      backgroundImage: {
        'stellar-gradient': 'linear-gradient(135deg, rgb(var(--color-stellar)), rgb(var(--color-nova)))',
        'aurora-gradient': 'linear-gradient(135deg, rgb(var(--color-aurora)), rgb(var(--color-success)))',
        'space-gradient': 'linear-gradient(180deg, rgb(var(--color-void)), rgb(var(--color-space)))',
        'card-gradient': 'linear-gradient(135deg, rgba(22, 26, 35, 0.9), rgba(14, 17, 23, 0.95))',
        'grid-pattern': `
          linear-gradient(rgba(245, 166, 35, 0.02) 1px, transparent 1px),
          linear-gradient(90deg, rgba(245, 166, 35, 0.02) 1px, transparent 1px)
        `,
        'gradient-radial': 'radial-gradient(ellipse at center, var(--tw-gradient-stops))',
      },
      
      backgroundSize: {
        'grid': '50px 50px',
      },
      
      // Animations
      animation: {
        'fade-in': 'fade-in 0.4s ease-out forwards',
        'slide-up': 'slide-up 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'slide-in-right': 'slide-in-right 0.3s ease-out forwards',
        'scale-in': 'scale-in 0.25s ease-out forwards',
        'float': 'float 4s ease-in-out infinite',
        'shimmer': 'shimmer 1.5s infinite linear',
        'beacon': 'beacon-pulse 2s ease-out infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scan': 'scan 8s linear infinite',
      },
      
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(245, 166, 35, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(245, 166, 35, 0.8), 0 0 30px rgba(245, 166, 35, 0.4)' },
        },
        scan: {
          '0%, 100%': { transform: 'translateY(-100%)' },
          '50%': { transform: 'translateY(100%)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },
      
      // Z-index scale
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
      
      // Transition timing functions
      transitionTimingFunction: {
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
        'in-out-expo': 'cubic-bezier(0.87, 0, 0.13, 1)',
      },
    },
  },
  plugins: [],
}
