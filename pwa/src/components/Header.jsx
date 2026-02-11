import React, { useState, useRef, useEffect } from 'react';

export function Header({ model, models, onModelChange, onSettings, onClear, hasMessages }) {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('pointerdown', handler);
    return () => document.removeEventListener('pointerdown', handler);
  }, [open]);

  return (
    <header className="header">
      <div className="header-brand">
        <img src="/favicon.svg" alt="MORNINGSTAR" className="header-logo" />
        <span className="header-title">MORNINGSTAR</span>
      </div>
      <div className="header-actions">
        {hasMessages && (
          <button className="settings-btn" onClick={onClear} title="New chat" aria-label="New chat">
            +
          </button>
        )}
        <div className="model-selector" ref={dropdownRef}>
          <button className="model-btn" onClick={() => setOpen(!open)}>
            <span className="dot" />
            <span>{model.name.replace('MORNINGSTAR ', '')}</span>
            <span className="chevron">{open ? '\u25B2' : '\u25BC'}</span>
          </button>
          {open && (
            <div className="model-dropdown">
              {models.map((m) => (
                <button
                  key={m.id}
                  className={`model-option ${m.id === model.id ? 'active' : ''}`}
                  onClick={() => { onModelChange(m); setOpen(false); }}
                >
                  <span className="model-option-name">{m.name}</span>
                  <span className="model-option-desc">{m.desc}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="settings-btn" onClick={onSettings} title="Settings" aria-label="Settings">
          &#9881;
        </button>
      </div>
    </header>
  );
}
