/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bento-bg': 'var(--color-bento-bg)',
        'bento-card': 'var(--color-bento-card)',
        'bento-border': 'var(--color-bento-border)',
        mint: 'var(--color-mint)',
        teal: 'var(--color-teal)',
        quantum: 'var(--color-quantum)',
        nebula: 'var(--color-nebula)',
        amber: 'var(--color-amber)',
        electrum: 'var(--color-electrum)',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      animation: {
         shake: 'shake 0.5s cubic-bezier(.36,.07,.19,.97) both',
         shimmer: 'shimmer 2s infinite linear',
      },
      keyframes: {
         shake: {
           '10%, 90%': { transform: 'translate3d(-1px, 0, 0)' },
           '20%, 80%': { transform: 'translate3d(2px, 0, 0)' },
           '30%, 50%, 70%': { transform: 'translate3d(-4px, 0, 0)' },
           '40%, 60%': { transform: 'translate3d(4px, 0, 0)' }
         },
         shimmer: {
           '0%': { transform: 'translateX(-100%)' },
           '100%': { transform: 'translateX(100%)' }
         }
      }
    },
  },
  plugins: [],
}
