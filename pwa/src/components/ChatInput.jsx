import React, { useState, useRef, useCallback } from 'react';

export function ChatInput({ onSend, onStop, streaming, supportsVision }) {
  const [text, setText] = useState('');
  const [images, setImages] = useState([]);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleSubmit = useCallback(() => {
    if (streaming) return;
    if (!text.trim() && images.length === 0) return;
    onSend(text, images.map(i => i.base64));
    setText('');
    setImages([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, images, streaming, onSend]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleInput = useCallback((e) => {
    setText(e.target.value);
    // Auto-resize
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }, []);

  const handleAttach = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFiles = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    files.forEach((file) => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        setImages(prev => [...prev, {
          base64,
          preview: reader.result,
          name: file.name,
        }]);
      };
      reader.readAsDataURL(file);
    });
    e.target.value = '';
  }, []);

  const removeImage = useCallback((index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  }, []);

  return (
    <div className="input-area">
      {images.length > 0 && (
        <div className="image-preview-strip">
          {images.map((img, i) => (
            <div key={i} className="image-preview-item">
              <img src={img.preview} alt={img.name} className="image-preview-thumb" />
              <button className="image-preview-remove" onClick={() => removeImage(i)}>
                &times;
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="input-row">
        {supportsVision && (
          <>
            <button className="attach-btn" onClick={handleAttach} title="Attach image" aria-label="Attach image">
              &#128247;
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={handleFiles}
            />
          </>
        )}
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="input-field"
            placeholder={supportsVision ? 'Message or attach an image...' : 'Type a message...'}
            value={text}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
          />
        </div>
        {streaming ? (
          <button className="send-btn stop-btn" onClick={onStop} title="Stop" aria-label="Stop generation">
            &#9632;
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={handleSubmit}
            disabled={!text.trim() && images.length === 0}
            title="Send"
            aria-label="Send message"
          >
            &#9654;
          </button>
        )}
      </div>
    </div>
  );
}
