/**
 * Ollama API integration for MORNINGSTAR Vision AI PWA.
 * Streams chat completions from the local Ollama server.
 */

export async function checkConnection(apiUrl) {
  try {
    const res = await fetch(`${apiUrl}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function sendMessage({ apiUrl, model, messages, signal, onToken }) {
  const res = await fetch(`${apiUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
    }),
    signal,
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => 'Unknown error');
    throw new Error(`Ollama API error (${res.status}): ${errText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (data.message?.content) {
          onToken(data.message.content);
        }
      } catch {
        // Skip malformed lines
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer);
      if (data.message?.content) {
        onToken(data.message.content);
      }
    } catch {
      // Skip
    }
  }
}
