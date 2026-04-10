import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: { port: 3054, proxy: { '/api': { target: 'http://localhost:8054', rewrite: p => p.replace(/^\/api/, '') } } }
})
