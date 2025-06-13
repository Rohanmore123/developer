import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { FaArrowLeft, FaMicrophone, FaStop, FaArrowCircleRight } from "react-icons/fa";
import webSocketService from "./WebSocketService";

const PrashaDoctorChatWithService = () => {
  // State variables
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  
  // Refs
  const chatMessagesRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  // Navigation
  const navigate = useNavigate();
  
  // Constants
  const JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzdGF0aWMtdXNlciIsImVtYWlsIjoic3RhdGljQGV4YW1wbGUuY29tIiwicm9sZXMiOiJhZG1pbixkb2N0b3IscGF0aWVudCJ9.3oZ2Ubh5rLBdHvQHd5Qr9GJczA5MXcxaVx5H5xLwvZ4';
  const REMOTE_SERVER_URL = 'http://44.221.59.250:8000';
  const PATIENT_ID = 'a07b24f2-5eda-4dd7-b5c4-ddef1226b8a2'; // In a real app, this would come from auth context
  
  // Initialize WebSocket connection and event listeners
  useEffect(() => {
    // Set up event listeners
    const handleOpen = () => {
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
    
    const handleClose = () => {
      setIsConnected(false);
    };
    
    const handleError = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    const handleTranscription = (transcription) => {
      // Update the last user message if it was a recording
      setMessages(prevMessages => {
        const newMessages = [...prevMessages];
        const lastUserMessageIndex = newMessages
          .map((msg, index) => ({ ...msg, index }))
          .filter(msg => msg.sender === 'user')
          .pop();
        
        if (lastUserMessageIndex && lastUserMessageIndex.text.includes('🎤 Recording...')) {
          newMessages[lastUserMessageIndex.index].text = transcription;
        } else {
          newMessages.push({
            id: Date.now(),
            text: transcription,
            sender: 'user',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
          });
        }
        
        return newMessages;
      });
      
      setIsTyping(true);
    };
    
    const handleAiResponse = (response) => {
      setIsTyping(false);
      
      setMessages(prevMessages => [
        ...prevMessages,
        {
          id: Date.now(),
          text: response.text,
          sender: 'ai',
          keywords: response.keywords || [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    };
    
    // Add event listeners
    webSocketService.addEventListener('open', handleOpen);
    webSocketService.addEventListener('close', handleClose);
    webSocketService.addEventListener('error', handleError);
    webSocketService.addEventListener('transcription', handleTranscription);
    webSocketService.addEventListener('aiResponse', handleAiResponse);
    
    // Connect to WebSocket
    webSocketService.connect(PATIENT_ID, JWT_TOKEN, REMOTE_SERVER_URL)
      .catch(error => {
        console.error('Failed to connect:', error);
        alert(`Failed to connect: ${error.message}`);
      });
    
    // Cleanup on unmount
    return () => {
      webSocketService.removeEventListener('open', handleOpen);
      webSocketService.removeEventListener('close', handleClose);
      webSocketService.removeEventListener('error', handleError);
      webSocketService.removeEventListener('transcription', handleTranscription);
      webSocketService.removeEventListener('aiResponse', handleAiResponse);
      webSocketService.disconnect();
    };
  }, []);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  }, [messages]);
  
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
    
    if (!webSocketService.isSocketConnected()) {
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
    const success = webSocketService.sendTextMessage(message);
    if (!success) {
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
    
    reader.onload = () => {
      const base64Audio = reader.result.split(',')[1];
      
      if (webSocketService.isSocketConnected()) {
        const success = webSocketService.sendAudioMessage(base64Audio);
        if (!success) {
          alert('Error sending audio. Please try again.');
        }
      } else {
        console.error('WebSocket not connected');
        alert('WebSocket not connected. Please refresh the page.');
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

export default PrashaDoctorChatWithService;
