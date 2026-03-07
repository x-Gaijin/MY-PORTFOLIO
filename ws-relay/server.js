/**
 * WebSocket Relay Server
 * Runs on Render (free tier) — proxies browser WebSocket connections
 * through to the ngrok tunnel, bypassing mobile ISP blocking.
 *
 * Phone → wss://relay.onrender.com → wss://driving-dominant-bobcat.ngrok-free.app → Python backend
 */

const { WebSocketServer, WebSocket } = require('ws');
const http = require('http');

const PORT = process.env.PORT || 8080;
const BACKEND_URL = process.env.BACKEND_WS_URL || 'wss://driving-dominant-bobcat.ngrok-free.app';

const server = http.createServer((req, res) => {
    // Health check endpoint for Render
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('WebSocket relay is running\n');
});

const wss = new WebSocketServer({ server });

wss.on('connection', (clientWs, req) => {
    console.log(`[RELAY] Browser connected from ${req.socket.remoteAddress}`);

    // Connect to the actual Python backend via ngrok
    const backendWs = new WebSocket(BACKEND_URL, {
        headers: {
            'ngrok-skip-browser-warning': 'true',
        },
    });
    backendWs.binaryType = 'nodebuffer';

    // ─── Browser → Backend ───────────────────────────────
    clientWs.on('message', (data, isBinary) => {
        if (backendWs.readyState === WebSocket.OPEN) {
            backendWs.send(data, { binary: isBinary });
        }
    });

    // ─── Backend → Browser ───────────────────────────────
    backendWs.on('message', (data, isBinary) => {
        if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.send(data, { binary: isBinary });
        }
    });

    // ─── Error / Close handling ───────────────────────────
    clientWs.on('close', (code, reason) => {
        console.log(`[RELAY] Browser disconnected (code=${code})`);
        if (backendWs.readyState === WebSocket.OPEN) backendWs.close();
    });

    backendWs.on('close', (code, reason) => {
        console.log(`[RELAY] Backend disconnected (code=${code})`);
        if (clientWs.readyState === WebSocket.OPEN) clientWs.close();
    });

    clientWs.on('error', (err) => console.error('[RELAY] Client error:', err.message));
    backendWs.on('error', (err) => {
        console.error('[RELAY] Backend error:', err.message);
        if (clientWs.readyState === WebSocket.OPEN) {
            clientWs.close(1011, 'Backend unreachable');
        }
    });

    backendWs.on('open', () => {
        console.log('[RELAY] ✅ Connected to backend via ngrok');
    });
});

server.listen(PORT, () => {
    console.log(`[RELAY] 🚀 Listening on port ${PORT}`);
    console.log(`[RELAY] 🔗 Forwarding to: ${BACKEND_URL}`);
});
