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
        deep: '#050505',
        void: '#0a0a0c',
        cyan: { DEFAULT: '#00f2ff' },
        orange: { DEFAULT: '#ff8c00' }
      }
    }
  },
  plugins: []
}
