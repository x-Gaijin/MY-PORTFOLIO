import { defineConfig } from 'vite';
import { resolve } from 'path';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  plugins: [basicSsl()],
  root: 'src',
  envDir: resolve(__dirname, '.'),   // load .env files from project root, not src/
  publicDir: '../public',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'src/index.html'),
        fpga: resolve(__dirname, 'src/portfolio-fpga.html'),
        arduino: resolve(__dirname, 'src/portfolio-arduino.html'),
        ai: resolve(__dirname, 'src/portfolio-ai.html'),
        cybersecurity: resolve(__dirname, 'src/portfolio-cybersecurity.html'),
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@modules': resolve(__dirname, 'src/modules'),
      '@styles': resolve(__dirname, 'src/styles'),
      '@assets': resolve(__dirname, 'src/assets'),
    },
  },
  server: {
    port: 5173,
    open: true,
    host: '0.0.0.0',
    https: true,
    proxy: {
      '/ws': {
        target: 'ws://localhost:8888',
        ws: true,
        rewrite: (path) => path.replace(/^\/ws/, ''),
      },
    },
  },
});