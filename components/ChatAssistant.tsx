import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Bot } from 'lucide-react';
import { ChatMessage } from '../types';
import { createChatSession } from '../services/geminiService';
import { Chat, GenerateContentResponse } from "@google/genai";

const ChatAssistant: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'model', text: 'Hello! I am your AI Design Assistant. How can I help you today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatSessionRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize chat session once
    if (!chatSessionRef.current) {
      chatSessionRef.current = createChatSession();
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim() || !chatSessionRef.current) return;

    const userMsg: ChatMessage = { role: 'user', text: inputValue };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsTyping(true);

    try {
      const result = await chatSessionRef.current.sendMessage({ message: userMsg.text });
      const text = result.text;
      setMessages(prev => [...prev, { role: 'model', text: text || "I didn't catch that." }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'model', text: "I'm having trouble connecting to the network right now." }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <>
      {/* Floating Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed bottom-6 right-6 p-4 rounded-full shadow-2xl transition-all duration-300 z-50 flex items-center justify-center
          ${isOpen ? 'bg-zinc-800 text-zinc-400 rotate-90' : 'bg-pink-600 text-white hover:bg-pink-700'}`}
      >
        {isOpen ? <X size={24} /> : <MessageSquare size={24} />}
      </button>

      {/* Chat Window */}
      <div className={`fixed bottom-24 right-6 w-80 md:w-96 bg-zinc-900 border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden transition-all duration-300 z-40 transform origin-bottom-right flex flex-col
        ${isOpen ? 'scale-100 opacity-100' : 'scale-90 opacity-0 pointer-events-none'}`}
        style={{ height: '500px' }}
      >
        {/* Header */}
        <div className="bg-zinc-950 p-4 border-b border-zinc-800 flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-pink-500 to-violet-600 flex items-center justify-center">
            <Bot size={16} className="text-white" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Design Assistant</h3>
            <p className="text-xs text-zinc-500">Powered by Gemini 3 Pro</p>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed
                ${msg.role === 'user' 
                  ? 'bg-zinc-800 text-white rounded-br-none' 
                  : 'bg-zinc-950 border border-zinc-800 text-zinc-300 rounded-bl-none'}`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-zinc-950 border border-zinc-800 rounded-2xl rounded-bl-none px-4 py-3 flex gap-1">
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce" />
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce delay-75" />
                <span className="w-1.5 h-1.5 bg-zinc-500 rounded-full animate-bounce delay-150" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-3 bg-zinc-950 border-t border-zinc-800">
          <div className="relative">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask for ideas or help..."
              className="w-full bg-zinc-900 border border-zinc-800 text-zinc-200 text-sm rounded-xl py-3 pl-4 pr-10 focus:outline-none focus:border-zinc-700"
            />
            <button 
              onClick={handleSend}
              disabled={!inputValue.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-pink-600 hover:bg-pink-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default ChatAssistant;