/**
 * WebSocketClient — Configurable WebSocket with exponential backoff reconnect.
 *
 * Usage:
 *   const ws = new WebSocketClient(url, { onMessage, onStatusChange });
 *   ws.connect();
 *   ws.send('Hello');
 *   ws.disconnect();
 */

export const ConnectionStatus = Object.freeze({
    CONNECTED: 'connected',
    RECONNECTING: 'reconnecting',
    OFFLINE: 'offline',
});

export class WebSocketClient {
    /**
     * @param {string} url
     * @param {Object} opts
     * @param {(data: string) => void} opts.onMessage
     * @param {(status: string) => void} opts.onStatusChange
     * @param {number} [opts.maxRetries=10]
     * @param {number} [opts.baseDelay=1000]
     */
    constructor(url, { onMessage, onStatusChange, maxRetries = 10, baseDelay = 1000 } = {}) {
        this.url = url;
        this.onMessage = onMessage || (() => { });
        this.onStatusChange = onStatusChange || (() => { });
        this.maxRetries = maxRetries;
        this.baseDelay = baseDelay;

        /** @type {WebSocket|null} */
        this.ws = null;
        this.retryCount = 0;
        this.retryTimeout = null;
        this.intentionalClose = false;
        this._status = ConnectionStatus.OFFLINE;
    }

    get status() {
        return this._status;
    }

    set status(value) {
        if (this._status !== value) {
            this._status = value;
            this.onStatusChange(value);
        }
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

        this.intentionalClose = false;

        try {
            this.ws = new WebSocket(this.url);
        } catch (err) {
            console.warn('[WebSocketClient] Failed to create WebSocket:', err);
            this.status = ConnectionStatus.OFFLINE;
            return;
        }

        this.ws.onopen = () => {
            this.retryCount = 0;
            this.status = ConnectionStatus.CONNECTED;
        };

        this.ws.onmessage = (event) => {
            this.onMessage(event.data);
        };

        this.ws.onerror = () => {
            /* errors trigger onclose, handled there */
        };

        this.ws.onclose = () => {
            if (this.intentionalClose) {
                this.status = ConnectionStatus.OFFLINE;
                return;
            }
            this._reconnect();
        };
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
        }
    }

    disconnect() {
        this.intentionalClose = true;
        clearTimeout(this.retryTimeout);
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.status = ConnectionStatus.OFFLINE;
    }

    _reconnect() {
        if (this.retryCount >= this.maxRetries) {
            this.status = ConnectionStatus.OFFLINE;
            return;
        }

        this.status = ConnectionStatus.RECONNECTING;
        const delay = Math.min(this.baseDelay * 2 ** this.retryCount, 30_000);
        this.retryCount++;

        this.retryTimeout = setTimeout(() => {
            this.connect();
        }, delay);
    }
}
