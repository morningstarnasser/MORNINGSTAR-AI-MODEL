# Local Ollama model inventory

This snapshot records the models installed on the development machine. We never commit model
weights to Git; run `ollama list` for the live inventory on any machine.

| Ollama name | Role | Size at snapshot |
|---|---|---:|
| `qwen3.5:27b-int4` | General-purpose local base | 16 GB |
| `creativesync/hydra-ai:latest` | Morningstar Hydra identity/model wrapper | 16 GB |
| `hf.co/huihui-ai/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF:BF16` | Local research model | 19 GB |

The Morningstar CLI discovers Ollama models dynamically, so newly installed models appear in
its provider picker without a code change. The repository contains only Modelfiles,
configuration and evaluation code; weights remain in the local Ollama store.

```bash
ollama list
ollama run creativesync/hydra-ai:latest
```
