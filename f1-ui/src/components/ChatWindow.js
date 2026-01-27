import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import './ChatWindow.css';

const ChatWindow = ({ initialMessage }) => {
  const [messages, setMessages] = useState(
    initialMessage
      ? [{ id: Date.now(), text: initialMessage, sender: 'user' }]
      : []
  );
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false); // New state for loading
  const messagesEndRef = useRef(null);

  const sendMessage = async () => {
    if (newMessage.trim() === '') return;

    const userMessage = { id: Date.now(), text: newMessage, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setNewMessage('');
    setIsLoading(true); // Start loading

    try {
      const response = await fetch('http://localhost:5003/llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: newMessage }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const llmResponse = {
        id: Date.now(),
        text: data.response || 'No response received',
        sender: 'llm',
        image: data.image || null,
      };
      setMessages((prevMessages) => [...prevMessages, llmResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now(),
        text: 'Error sending message. Please try again.',
        sender: 'llm',
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false); // Stop loading regardless of success or failure
    }
  };

  const handleInputChange = (event) => {
    setNewMessage(event.target.value);
  };

  // Scroll to the bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="w-[900px] h-[700px] flex flex-col rounded-2xl border border-gray-700">
      <div className="chat-window-body p-4 overflow-y-auto" style={{ flexGrow: 1 }}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={`mb-4 ${
              message.sender === 'user' ? 'text-right' : 'text-left'
            }`}
          >
            <div
              className={`inline-block p-3 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-red-500 text-white'
                  : 'bg-gray-200 text-gray-800'
              }`}
            >
              {message.text}
              {message.image && (
                <img
                  src={message.image}
                  alt="Response"
                  className="mt-2 max-w-full rounded"
                />
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="mb-4 text-left">
            <div className="inline-block p-3 rounded-lg bg-gray-200 text-gray-800">
              Searching...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-window-footer p-4 border-t border-gray-700 flex items-center">
        <input
          type="text"
          placeholder="What do you want to know about Formula 1?"
          value={newMessage}
          onChange={handleInputChange}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          className="flex-grow px-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:border-red-500 text-gray-800 bg-gray-200"
          aria-label="Type your message"
        />
        <button
          className="ml-4 px-4 py-2 bg-red-600 text-white rounded-lg font-semibold hover:bg-red-700 transition-colors"
          onClick={sendMessage}
          aria-label="Send message"
        >
          ↑
        </button>
      </div>
    </div>
  );
};

ChatWindow.propTypes = {
  initialMessage: PropTypes.string,
};

export default ChatWindow;