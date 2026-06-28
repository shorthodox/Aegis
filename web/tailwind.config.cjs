/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/pages/**/*.html',
    './src/scripts/**/*.js',
    './src/components/**/*.js',
  ],
  theme: {
    extend: {
      colors: {
        deep: '#0C0F13',
        void: '#0C0F13',
        cyan: { DEFAULT: '#4A8FAB' },
        orange: { DEFAULT: '#B8966A' },
        gold: { DEFAULT: '#B8966A' }
      }
    }
  },
  plugins: []
}
