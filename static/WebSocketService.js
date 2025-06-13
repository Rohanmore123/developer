/**
 * WebSocketService.js
 * A service to manage WebSocket connections for the Prasha Doctor Chat
 */

class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.listeners = {
      message: [],
      open: [],
      close: [],
      error: [],
      transcription: [],
      aiResponse: []
    };
  }

  /**
   * Connect to the WebSocket server
   * @param {string} patientId - The patient ID
   * @param {string} token - JWT token for authentication
   * @param {string} serverUrl - The server URL
   * @returns {Promise} - Resolves when connected, rejects on error
   */
  connect(patientId, token, serverUrl = 'http://44.221.59.250:8000') {
    return new Promise((resolve, reject) => {
      // Close existing connection if any
      if (this.socket) {
        this.socket.close();
      }

      try {
        // Create WebSocket URL with token as query parameter
        const protocol = serverUrl.startsWith('https') ? 'wss:' : 'ws:';
        const host = serverUrl.replace(/^https?:\/\//, '');
        const wsUrl = `${protocol}//${host}/smart-chat/${patientId}?token=${token}`;
        
        console.log(`Connecting to WebSocket: ${wsUrl}`);
        
        // Create WebSocket
        this.socket = new WebSocket(wsUrl);
        
        // WebSocket event handlers
        this.socket.onopen = (event) => {
          console.log('WebSocket connection opened');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this._notifyListeners('open', event);
          resolve();
        };
        
        this.socket.onmessage = (event) => {
          console.log('Received message:', event.data);
          try {
            const data = JSON.parse(event.data);
            this._notifyListeners('message', data);
            
            if (data.transcription) {
              // This is a transcription response
              this._notifyListeners('transcription', data.transcription);
            } else if (data.response) {
              // This is a normal response
              this._notifyListeners('aiResponse', {
                text: data.response,
                keywords: data.extracted_keywords || []
              });
            } else if (data.error) {
              // This is an error
              console.error('Server error:', data.error);
              this._notifyListeners('error', new Error(data.error));
            }
          } catch (error) {
            console.error('Error parsing message:', error);
            this._notifyListeners('error', error);
          }
        };
        
        this.socket.onclose = (event) => {
          console.log(`WebSocket closed: Code ${event.code}, Reason: ${event.reason}`);
          this.isConnected = false;
          this._notifyListeners('close', event);
          
          // Attempt to reconnect if not a normal closure
          if (event.code !== 1000 && event.code !== 1001) {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
              this.reconnectAttempts++;
              const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
              console.log(`Reconnecting in ${delay/1000} seconds...`);
              setTimeout(() => this.connect(patientId, token, serverUrl), delay);
            }
          }
        };
        
        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this._notifyListeners('error', error);
          reject(error);
        };
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        this._notifyListeners('error', error);
        reject(error);
      }
    });
  }

  /**
   * Send a text message
   * @param {string} text - The message text
   * @returns {boolean} - True if sent successfully, false otherwise
   */
  sendTextMessage(text) {
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return false;
    }
    
    try {
      this.socket.send(JSON.stringify({ text }));
      console.log('Text message sent:', text);
      return true;
    } catch (error) {
      console.error('Error sending text message:', error);
      this._notifyListeners('error', error);
      return false;
    }
  }

  /**
   * Send audio data
   * @param {string} base64Audio - Base64 encoded audio data
   * @returns {boolean} - True if sent successfully, false otherwise
   */
  sendAudioMessage(base64Audio) {
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return false;
    }
    
    try {
      this.socket.send(JSON.stringify({ audio: base64Audio }));
      console.log('Audio message sent');
      return true;
    } catch (error) {
      console.error('Error sending audio message:', error);
      this._notifyListeners('error', error);
      return false;
    }
  }

  /**
   * Close the WebSocket connection
   */
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  /**
   * Add an event listener
   * @param {string} event - Event name ('message', 'open', 'close', 'error', 'transcription', 'aiResponse')
   * @param {Function} callback - Callback function
   */
  addEventListener(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  /**
   * Remove an event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function to remove
   */
  removeEventListener(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  /**
   * Notify all listeners of an event
   * @param {string} event - Event name
   * @param {any} data - Event data
   * @private
   */
  _notifyListeners(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Check if WebSocket is connected
   * @returns {boolean} - True if connected, false otherwise
   */
  isSocketConnected() {
    return this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN;
  }
}

// Create a singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;
