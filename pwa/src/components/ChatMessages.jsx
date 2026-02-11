import React, { useRef, useEffect } from 'react';
import { MessageBubble } from './MessageBubble.jsx';

export function ChatMessages({ messages, streaming, model }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="messages">
        <Welcome model={model} />
      </div>
    );
  }

  return (
    <div className="messages">
      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          message={msg}
          isStreaming={streaming && i === messages.length - 1 && msg.role === 'assistant'}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

function Welcome({ model }) {
  return (
    <div className="welcome">
      <img src="/favicon.svg" alt="" className="welcome-logo" />
      <h2>MORNINGSTAR Vision AI</h2>
      <p>
        AI-powered code generation, review, and visual analysis.
        Currently using <strong>{model.name}</strong>.
      </p>
      <div className="quick-actions">
        <QuickChip text="Write a React hook" />
        <QuickChip text="Explain this code" />
        <QuickChip text="Debug my function" />
        <QuickChip text="Generate an API" />
      </div>
    </div>
  );
}

function QuickChip({ text }) {
  return <span className="quick-action">{text}</span>;
}
