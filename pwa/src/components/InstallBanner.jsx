import React, { useState, useEffect } from 'react';

export function InstallBanner() {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [dismissed, setDismissed] = useState(() => sessionStorage.getItem('ms-install-dismissed') === '1');

  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  if (!deferredPrompt || dismissed) return null;

  const handleInstall = async () => {
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setDeferredPrompt(null);
    }
  };

  const handleDismiss = () => {
    setDismissed(true);
    sessionStorage.setItem('ms-install-dismissed', '1');
  };

  return (
    <div className="install-banner">
      <span className="install-banner-text">Install MORNINGSTAR for quick access</span>
      <div className="install-banner-actions">
        <button className="install-banner-btn dismiss" onClick={handleDismiss}>Later</button>
        <button className="install-banner-btn primary" onClick={handleInstall}>Install</button>
      </div>
    </div>
  );
}
