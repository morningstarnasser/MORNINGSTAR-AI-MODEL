import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessages } from './components/ChatMessages.jsx';
import { ChatInput } from './components/ChatInput.jsx';
import { Header } from './components/Header.jsx';
import { SettingsPanel } from './components/SettingsPanel.jsx';
import { InstallBanner } from './components/InstallBanner.jsx';
import { ConnectionBar } from './components/ConnectionBar.jsx';
import { sendMessage, checkConnection } from './api.js';

const DEFAULT_API_URL = 'http://localhost:11434';

const MODELS = [
  { id: 'morningstar', name: 'MORNINGSTAR 14B', desc: 'Fast, daily coding', vision: false },
  { id: 'morningstar-32b', name: 'MORNINGSTAR 32B', desc: 'Maximum quality', vision: false },
  { id: 'morningstar-vision', name: 'MORNINGSTAR Vision', desc: 'Image analysis + code', vision: true },
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [model, setModel] = useState(MODELS[0]);
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem('ms-api-url') || DEFAULT_API_URL);
  const [connected, setConnected] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const abortRef = useRef(null);

  // Persist API URL
  useEffect(() => {
    localStorage.setItem('ms-api-url', apiUrl);
  }, [apiUrl]);

  // Check connection on mount and URL change
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await checkConnection(apiUrl);
      if (!cancelled) setConnected(ok);
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [apiUrl]);

  const handleSend = useCallback(async (text, images) => {
    if (!text.trim() && images.length === 0) return;
    if (streaming) return;

    const userMsg = {
      role: 'user',
      content: text,
      images: images.length > 0 ? images : undefined,
      timestamp: Date.now(),
    };

    const assistantMsg = {
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const allMsgs = [...messages, userMsg].map(m => ({
        role: m.role,
        content: m.content,
        ...(m.images ? { images: m.images } : {}),
      }));

      await sendMessage({
        apiUrl,
        model: model.id,
        messages: allMsgs,
        signal: controller.signal,
        onToken: (token) => {
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = { ...last, content: last.content + token };
            return updated;
          });
        },
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = {
            ...last,
            content: last.content || `Error: ${err.message}`,
            error: true,
          };
          return updated;
        });
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [messages, model, apiUrl, streaming]);

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);

  const handleClear = useCallback(() => {
    setMessages([]);
  }, []);

  return (
    <div className="app">
      <Header
        model={model}
        models={MODELS}
        onModelChange={setModel}
        onSettings={() => setShowSettings(true)}
        onClear={handleClear}
        hasMessages={messages.length > 0}
      />
      <InstallBanner />
      {connected === false && <ConnectionBar connected={false} apiUrl={apiUrl} />}
      <ChatMessages
        messages={messages}
        streaming={streaming}
        model={model}
      />
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        streaming={streaming}
        supportsVision={model.vision}
      />
      {showSettings && (
        <SettingsPanel
          apiUrl={apiUrl}
          onApiUrlChange={setApiUrl}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
