'use client';

import { useEffect } from 'react';
import { processQuery } from '../lib/chatbot.jsx';
import styles from './Chat.module.css';


export default function Chat() {

  const handleSend = async () => {
    const botResponse = await processQuery("bababoe");
  };
  useEffect(() => {
    handleSend()
  }, []);


  return (
    <div className={styles.chatContainer}>
    </div>
  );
}