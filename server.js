/**
 * Combined server for Render deployment:
 *   1. Serves the built Vite static site (dist/)
 *   2. Proxies WebSocket at /ws → ngrok → Python backend
 *
 * Flow:
 *   Phone (mobile data)
 *     ↓  wss://my-app.onrender.com/ws   ← never blocked by ISP
 *   Render (this server)
 *     ↓  wss://driving-dominant-bobcat.ngrok-free.app  ← server-to-server, fine
 *   ngrok → Python backend ✅
 */

import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_WS_URL || 'wss://driving-dominant-bobcat.ngrok-free.app';

// ─── Static file server ───────────────────────────────────────────────────────
const app = express();
app.use(express.static(join(__dirname, 'dist')));

// Serve the correct HTML for each portfolio page
app.get('*', (req, res) => {
    const page = req.path.replace(/^\//, '') || 'index.html';
    const file = join(__dirname, 'dist', page.endsWith('.html') ? page : 'index.html');
    res.sendFile(file, () => {
        // Fallback to index.html if not found
        res.sendFile(join(__dirname, 'dist', 'index.html'));
    });
});

// ─── HTTP server ──────────────────────────────────────────────────────────────
const server = createServer(app);

// ─── WebSocket relay ──────────────────────────────────────────────────────────
const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (request, socket, head) => {
    const url = request.url || '';
    if (url === '/ws' || url.startsWith('/ws?')) {
        wss.handleUpgrade(request, socket, head, (ws) => {
            wss.emit('connection', ws, request);
        });
    } else {
        socket.destroy();
    }
});

wss.on('connection', (clientWs, request) => {
    console.log(`[RELAY] Browser connected — relaying to ${BACKEND_URL}`);

    const backendWs = new WebSocket(BACKEND_URL, {
        headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    backendWs.binaryType = 'nodebuffer';

    // ── Browser → Backend ────────────────────────────────
    clientWs.on('message', (data, isBinary) => {
        if (backendWs.readyState === WebSocket.OPEN) {
            backendWs.send(data, { binary: isBinary });
        }
    });

    // ── Backend → Browser ────────────────────────────────
    backendWs.on('message', (data, isBinary) => {
        if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(data, { binary: isBinary });
        }
    });

    // ── Cleanup ──────────────────────────────────────────
    clientWs.on('close', (code) => {
        console.log(`[RELAY] Browser closed (${code})`);
        if (backendWs.readyState === WebSocket.OPEN) backendWs.close();
    });

    backendWs.on('close', (code) => {
        console.log(`[RELAY] Backend closed (${code})`);
        if (clientWs.readyState === WebSocket.OPEN) clientWs.close();
    });

    clientWs.on('error', (e) => console.error('[RELAY] Client error:', e.message));
    backendWs.on('error', (e) => {
        console.error('[RELAY] Backend error:', e.message);
        if (clientWs.readyState === WebSocket.OPEN) clientWs.close(1011, 'Backend unreachable');
    });

    backendWs.on('open', () => console.log('[RELAY] ✅ Connected to backend'));
});

// ─── Start ────────────────────────────────────────────────────────────────────
server.listen(PORT, () => {
    console.log(`🚀 Portfolio server on port ${PORT}`);
    console.log(`🔗 WS relay → ${BACKEND_URL}`);
});
