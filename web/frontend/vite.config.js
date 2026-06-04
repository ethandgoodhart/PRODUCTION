import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/state': 'http://127.0.0.1:5050',
      '/cam': 'http://127.0.0.1:5050',
      '/gps': 'http://127.0.0.1:5050',
      '/mbtile': 'http://127.0.0.1:5050',
      '/route': 'http://127.0.0.1:5050',
      '/nav_route': 'http://127.0.0.1:5050',
      '/quit': 'http://127.0.0.1:5050',
      '/offline-video-control': 'http://127.0.0.1:5050',
      '/live_cameras': 'http://127.0.0.1:5050',
      '/ego-trace': 'http://127.0.0.1:5050',
      '/browser-log': 'http://127.0.0.1:5050',
    },
  },
  build: {
    outDir: '../static/dist',
    emptyOutDir: true,
  },
});
