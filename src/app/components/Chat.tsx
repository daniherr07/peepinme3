'use client';

import type React from 'react';
import { useState, useRef, useEffect } from 'react';
import { processQuery, type StoreGroup } from '../lib/chatbot';
import StoreCard from './StoreCard';
import styles from './Chat.module.css';

// The message interface remains the same
interface Message {
  id: number;
  sender: 'user' | 'bot';
  text: string;
  storeGroups?: StoreGroup[];
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages([
      {
        id: Date.now(),
        sender: 'bot',
        text: "¡Pura Vida! I'm PeepInMe. Ask me where to find things in Costa Rica, and I'll find the 5 best matches for you!",
      },
    ]);
  }, []);

  useEffect(() => {
    // Scroll to the bottom whenever messages update or loading state changes
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { id: Date.now(), sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const botResponse = await processQuery(input);
    const botMessage: Message = {
      id: Date.now() + 1,
      sender: 'bot',
      text: botResponse.introMessage,
      storeGroups: botResponse.storeGroups,
    };
    
    // Update state in two steps to ensure smooth UI transition
    setIsLoading(false);
    setMessages((prev) => [...prev, botMessage]);
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.messagesArea}>
        {messages.map((msg) => (
          <div key={msg.id} className={`${styles.messageWrapper} ${styles[msg.sender]}`}>
            <div className={styles.message}>
              {/* Message text is now rendered directly */}
              <p>{msg.text}</p>

              {/* Render the grouped store results */}
              {msg.storeGroups && (
                <div className={styles.storeGroupsContainer}>
                  {msg.storeGroups.map((group) => (
                    <div key={group.category}>
                      <h4 className={styles.storeGroupHeader}>{group.category}</h4>
                      <div className={styles.storesContainer}>
                        {group.stores.map((store) => (
                          <StoreCard key={store.id} store={store} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Reverted to the static loading message using your CSS styles */}
        {isLoading && (
          <div className={`${styles.messageWrapper} ${styles.bot}`}>
            <div className={styles.message}>
              <div className={styles.loading}>
                <span />
                <span />
                <span />
                <p>PeepInMe is thinking...</p>
              </div>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className={styles.inputArea}>
        <form onSubmit={handleSend} className={styles.inputForm}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask for a product or store..."
            className={styles.input}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()} className={styles.sendButton}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}