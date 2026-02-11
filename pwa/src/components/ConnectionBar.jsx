import React from 'react';

export function ConnectionBar({ connected, apiUrl }) {
  if (connected !== false) return null;

  return (
    <div className="connection-bar">
      <span className="connection-dot" />
      <span>Cannot reach Ollama at {apiUrl}</span>
    </div>
  );
}
