import React from 'react';

export function SettingsPanel({ apiUrl, onApiUrlChange, onClose }) {
  return (
    <>
      <div className="settings-overlay" onClick={onClose} />
      <div className="settings-panel">
        <div className="settings-handle" />
        <div className="settings-title">Settings</div>

        <div className="settings-group">
          <div className="settings-label">Ollama API URL</div>
          <input
            className="settings-input"
            type="url"
            value={apiUrl}
            onChange={(e) => onApiUrlChange(e.target.value)}
            placeholder="http://localhost:11434"
          />
          <div className="settings-hint">
            The URL of your Ollama server. Default: http://localhost:11434
          </div>
        </div>

        <div className="settings-group">
          <div className="settings-label">About</div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <strong>MORNINGSTAR Vision AI</strong> is an open-source AI system for
            code generation, code review, and visual analysis. This PWA connects
            to your local Ollama instance to chat with MORNINGSTAR models.
          </p>
        </div>

        <div className="settings-group">
          <div className="settings-label">Models</div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <strong>14B</strong> — Fast daily coding (16GB+ RAM)<br />
            <strong>32B</strong> — Maximum quality (24GB+ RAM)<br />
            <strong>Vision</strong> — Image analysis, UI-to-code (16GB+ RAM)
          </p>
        </div>
      </div>
    </>
  );
}
