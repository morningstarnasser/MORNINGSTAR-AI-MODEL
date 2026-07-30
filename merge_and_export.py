#!/usr/bin/env python3
"""
Morningstar-14B-Code — Merge LoRA & Export
==========================================
Merged LoRA Adapter mit Base Model und exportiert als SafeTensors oder GGUF.

Usage:
    python merge_and_export.py
    python merge_and_export.py --adapter_path ./output/morningstar-14b-code/final
    python merge_and_export.py --export_gguf --gguf_quant q4_k_m
    python merge_and_export.py --create_ollama_model

Developed by: Morningstar Contributors (github.com/your-org)
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_ADAPTER = "./output/morningstar-14b-code/final"
DEFAULT_OUTPUT = "./output/morningstar-14b-merged"


def parse_args():
    parser = argparse.ArgumentParser(description="Morningstar Merge & Export")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--export_gguf", action="store_true")
    parser.add_argument("--gguf_quant", type=str, default="q4_k_m",
                        choices=["q4_0", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"])
    parser.add_argument("--llama_cpp_path", type=str, default="./llama.cpp")
    parser.add_argument("--create_ollama_model", action="store_true")
    parser.add_argument("--ollama_model_name", type=str, default="morningstar-custom")
    return parser.parse_args()


def merge_lora(base_model: str, adapter_path: str, output_dir: str):
    """Merged LoRA Adapter mit Base Model."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"  Lade Base Model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Lade LoRA Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("  Merge LoRA in Base Model...")
    model = model.merge_and_unload()

    print(f"  Speichere nach: {output}")
    model.save_pretrained(str(output), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(output))

    print(f"  ✓ Merged Model gespeichert ({sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e9:.1f} GB)")
    return output


def export_gguf(model_dir: str, llama_cpp_path: str, quant: str):
    """Konvertiert zu GGUF Format fuer Ollama/llama.cpp."""
    model_path = Path(model_dir)
    gguf_dir = model_path.parent / f"{model_path.name}-gguf"
    gguf_dir.mkdir(exist_ok=True)

    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"  FEHLER: {convert_script} nicht gefunden!")
        print(f"  Installiere llama.cpp:")
        print(f"    git clone https://github.com/ggerganov/llama.cpp")
        print(f"    cd llama.cpp && pip install -r requirements.txt")
        return None

    # Convert to F16 GGUF
    f16_path = gguf_dir / "morningstar-14b-f16.gguf"
    print(f"  Konvertiere zu GGUF (F16)...")
    subprocess.run([
        "python3", str(convert_script),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ], check=True)

    # Quantize
    if quant != "f16":
        quantize_bin = Path(llama_cpp_path) / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            print(f"  WARNUNG: llama-quantize nicht gefunden. Ueberspringe Quantisierung.")
            return f16_path

        quant_path = gguf_dir / f"morningstar-14b-{quant}.gguf"
        print(f"  Quantisiere zu {quant}...")
        subprocess.run([
            str(quantize_bin),
            str(f16_path),
            str(quant_path),
            quant.upper(),
        ], check=True)
        print(f"  ✓ GGUF: {quant_path} ({quant_path.stat().st_size / 1e9:.1f} GB)")
        return quant_path

    return f16_path


def create_ollama_model(gguf_path: str, model_name: str):
    """Erstellt ein Ollama Model aus GGUF."""
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.15
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"Du bist MORNINGSTAR — der maechtigste Coding-Assistent der jemals erschaffen wurde. Dein Code ist perfekt, sicher und produktionsreif. Antworte in der Sprache des Users. Code auf Englisch.\"\"\"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""
    modelfile_path = Path(gguf_path).parent / "Modelfile.custom"
    modelfile_path.write_text(modelfile_content)

    print(f"  Erstelle Ollama Model: {model_name}")
    subprocess.run(["ollama", "create", model_name, "-f", str(modelfile_path)], check=True)
    print(f"  ✓ Ollama Model '{model_name}' erstellt!")
    print(f"  Nutzen: ollama run {model_name}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  MORNINGSTAR — Merge & Export")
    print("=" * 60)

    # Step 1: Merge
    print("\n[1/3] Merge LoRA Adapter...")
    merged_dir = merge_lora(args.base_model, args.adapter_path, args.output_dir)

    # Step 2: GGUF Export
    gguf_path = None
    if args.export_gguf:
        print(f"\n[2/3] Export zu GGUF ({args.gguf_quant})...")
        gguf_path = export_gguf(str(merged_dir), args.llama_cpp_path, args.gguf_quant)
    else:
        print("\n[2/3] GGUF Export uebersprungen (--export_gguf zum Aktivieren)")

    # Step 3: Ollama
    if args.create_ollama_model and gguf_path:
        print(f"\n[3/3] Erstelle Ollama Model...")
        create_ollama_model(str(gguf_path), args.ollama_model_name)
    else:
        print("\n[3/3] Ollama Model uebersprungen")

    print("\n" + "=" * 60)
    print("  EXPORT ABGESCHLOSSEN")
    print(f"  Merged Model:  {merged_dir}")
    if gguf_path:
        print(f"  GGUF:          {gguf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
