import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';

export function MessageBubble({ message, isStreaming }) {
  const { role, content, images, error } = message;

  return (
    <div className={`message ${role}`}>
      <div className="message-role">{role === 'user' ? 'You' : 'MORNINGSTAR'}</div>
      <div className={`message-bubble ${error ? 'error' : ''}`}>
        {images && images.length > 0 && (
          <div>
            {images.map((img, i) => (
              <img
                key={i}
                src={`data:image/jpeg;base64,${img}`}
                alt="Attached"
                className="message-image"
              />
            ))}
          </div>
        )}
        {content ? (
          <ReactMarkdown components={{ pre: PreBlock, code: CodeInline }}>
            {content}
          </ReactMarkdown>
        ) : isStreaming ? (
          <div className="typing-indicator">
            <span /><span /><span />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function PreBlock({ children }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(() => {
    const code = extractText(children);
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [children]);

  return (
    <pre>
      {children}
      <button className="copy-btn" onClick={handleCopy}>
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </pre>
  );
}

function CodeInline({ inline, className, children, ...props }) {
  if (inline) {
    return <code className={className} {...props}>{children}</code>;
  }
  return <code className={className} {...props}>{children}</code>;
}

function extractText(node) {
  if (typeof node === 'string') return node;
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (node?.props?.children) return extractText(node.props.children);
  return '';
}
