import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { FaArrowLeft, FaMicrophone, FaStop, FaArrowCircleRight } from "react-icons/fa";

const PrashaDoctorChat = () => {
  // State variables
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Refs
  const wsRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  // Navigation
  const navigate = useNavigate();
  
  // Constants
  const JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzdGF0aWMtdXNlciIsImVtYWlsIjoic3RhdGljQGV4YW1wbGUuY29tIiwicm9sZXMiOiJhZG1pbixkb2N0b3IscGF0aWVudCJ9.3oZ2Ubh5rLBdHvQHd5Qr9GJczA5MXcxaVx5H5xLwvZ4';
  const REMOTE_SERVER_URL = 'http://44.221.59.250:8000';
  const PATIENT_ID = 'a07b24f2-5eda-4dd7-b5c4-ddef1226b8a2'; // In a real app, this would come from auth context
  
  // Connect to WebSocket
  useEffect(() => {
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);
  
  // Connect to WebSocket
  const connectWebSocket = () => {
    // Close existing connection if any
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    try {
      // Create WebSocket URL with token as query parameter
      const protocol = REMOTE_SERVER_URL.startsWith('https') ? 'wss:' : 'ws:';
      const host = REMOTE_SERVER_URL.replace(/^https?:\/\//, '');
      const wsUrl = `${protocol}//${host}/smart-chat/${PATIENT_ID}?token=${JWT_TOKEN}`;
      
      console.log(`Connecting to WebSocket: ${wsUrl}`);
      
      // Create WebSocket
      wsRef.current = new WebSocket(wsUrl);
      
      // WebSocket event handlers
      wsRef.current.onopen = () => {
        console.log('WebSocket connection opened');
        setIsConnected(true);
        
        // Add welcome message if no messages
        if (messages.length === 0) {
          setMessages([
            {
              id: Date.now(),
              text: 'Connected to chat server! How can I help you today?',
              sender: 'ai',
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }
          ]);
        }
      };
      
      wsRef.current.onmessage = (event) => {
        console.log('Received message:', event.data);
        try {
          const data = JSON.parse(event.data);
          
          // Hide typing indicator
          setIsTyping(false);
          
          if (data.transcription) {
            // This is a transcription response
            console.log('Received transcription:', data.transcription);
            
            // Update the last user message if it was a recording
            setMessages(prevMessages => {
              const newMessages = [...prevMessages];
              const lastUserMessageIndex = newMessages
                .map((msg, index) => ({ ...msg, index }))
                .filter(msg => msg.sender === 'user')
                .pop();
              
              if (lastUserMessageIndex && lastUserMessageIndex.text.includes('🎤 Recording...')) {
                newMessages[lastUserMessageIndex.index].text = data.transcription;
              } else {
                newMessages.push({
                  id: Date.now(),
                  text: data.transcription,
                  sender: 'user',
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                });
              }
              
              return newMessages;
            });
          } else if (data.response) {
            // This is a normal response
            setMessages(prevMessages => [
              ...prevMessages,
              {
                id: Date.now(),
                text: data.response,
                sender: 'ai',
                keywords: data.extracted_keywords || [],
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              }
            ]);
          } else if (data.error) {
            // This is an error
            console.error('Server error:', data.error);
            alert(`Error: ${data.error}`);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };
      
      wsRef.current.onclose = (event) => {
        console.log(`WebSocket closed: Code ${event.code}, Reason: ${event.reason}`);
        setIsConnected(false);
        
        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && event.code !== 1001) {
          setTimeout(connectWebSocket, 3000);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      alert(`Error connecting to server: ${error.message}`);
    }
  };
  
  // Handle input change
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };
  
  // Send text message
  const sendMessage = () => {
    const message = inputValue.trim();
    
    if (!message) {
      return;
    }
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert('Not connected to server. Please wait or refresh the page.');
      return;
    }
    
    // Add message to chat
    setMessages(prevMessages => [
      ...prevMessages,
      {
        id: Date.now(),
        text: message,
        sender: 'user',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    
    // Show typing indicator
    setIsTyping(true);
    
    // Send message to server
    try {
      wsRef.current.send(JSON.stringify({ text: message }));
      console.log('Message sent:', message);
    } catch (error) {
      console.error('Error sending message:', error);
      alert('Error sending message. Please try again.');
      setIsTyping(false);
    }
    
    // Clear input
    setInputValue("");
  };
  
  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };
  
  // Toggle recording
  const toggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };
  
  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorderRef.current.onstop = sendAudio;
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      // Add a placeholder message
      setMessages(prevMessages => [
        ...prevMessages,
        {
          id: Date.now(),
          text: '🎤 Recording...',
          sender: 'user',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } catch (error) {
      console.error('Microphone error:', error);
      alert(`Microphone error: ${error.message}`);
    }
  };
  
  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };
  
  // Send audio
  const sendAudio = () => {
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const reader = new FileReader();
    
    // Show typing indicator
    setIsTyping(true);
    
    reader.onload = () => {
      const base64Audio = reader.result.split(',')[1];
      
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ audio: base64Audio }));
        console.log('Audio sent to server');
      } else {
        console.error('WebSocket not connected');
        alert('WebSocket not connected. Please refresh the page.');
        setIsTyping(false);
      }
    };
    
    reader.readAsDataURL(audioBlob);
  };
  
  // Handle back button click
  const handleBack = () => {
    navigate(-1); // Go back to previous page
  };
  
  return (
    <div className="flex flex-col h-screen">
      <div className="flex items-center justify-between p-4 border-b">
        <button onClick={handleBack} className="text-gray-700">
          <FaArrowLeft />
        </button>
        <h2 className="text-lg font-semibold">Prasha Doctor</h2>
        <div className="w-6"></div> {/* Spacer to balance layout */}
      </div>
      
      <div 
        ref={chatMessagesRef}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {messages.length > 0 ? (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`rounded-xl px-4 py-2 max-w-xs
                  ${
                    msg.sender === 'user'
                      ? 'bg-purple-600 text-white'
                      : 'bg-orange-200 text-black'
                  }`}
              >
                <div>{msg.text}</div>
                {msg.keywords && msg.keywords.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {msg.keywords.map((keyword, index) => (
                      <span 
                        key={index}
                        className="text-xs bg-white bg-opacity-20 px-2 py-0.5 rounded-full"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
                <p className="text-xs text-right mt-1 text-gray-500">
                  {msg.timestamp}
                </p>
              </div>
            </div>
          ))
        ) : (
          <div className="flex justify-center items-center h-full">
            <h1 className="text-gray-500">
              Start a conversation with Prasha Doctor
            </h1>
          </div>
        )}
        
        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-orange-200 rounded-xl px-4 py-2 flex items-center space-x-1">
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
            </div>
          </div>
        )}
      </div>
      
      <div className="flex items-center p-3 border-t bg-white">
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Message..."
          className="flex-1 p-2 rounded-xl bg-gray-100"
          disabled={!isConnected}
        />
        <button 
          onClick={toggleRecording} 
          className="ml-2 text-gray-700"
          disabled={!isConnected}
        >
          {isRecording ? <FaStop className="text-red-500 animate-pulse" /> : <FaMicrophone />}
        </button>
        <button
          onClick={sendMessage}
          className="ml-2 bg-purple-600 p-2 rounded-full text-white disabled:bg-gray-400"
          disabled={!isConnected || !inputValue.trim()}
        >
          <FaArrowCircleRight />
        </button>
      </div>
    </div>
  );
};

export default PrashaDoctorChat;
