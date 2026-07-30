#!/usr/bin/env python3
"""
Example usage of the Morningstar 9.6B model.
Demonstrates the model's capabilities in conversation, reasoning, and code generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings('ignore')

def load_model():
    """Load the Morningstar model with appropriate settings."""
    print("Loading Morningstar 9.6B model...")
    
    model_id = "./"  # Local path
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    return generator, model, tokenizer

def demonstrate_capabilities(generator):
    """Demonstrate various capabilities of the Morningstar model."""
    
    print("\n" + "="*60)
    print("MORNINGSTAR AI CAPABILITIES DEMONSTRATION")
    print("="*60)
    
    # 1. CONVERSATIONAL ABILITY
    print("\n1. CONVERSATIONAL ABILITY")
    conversation = """User: Hallo, wie geht es dir?
Assistant: Mir geht es ausgezeichnet, danke der Nachfrage! Wie kann ich dir heute helfen?
User: Könntest du mir erklären, wie ein Transformer funktioniert?
Assistant:"""
    
    response = generator(
        conversation,
        max_length=300,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1
    )
    print(response[0]['generated_text'])
    
    # 2. CODE GENERATION
    print("\n2. CODE GENERATION (Python)")
    code_prompt = """Write a Python function to calculate the Fibonacci sequence up to n numbers efficiently:"""
    
    response = generator(
        code_prompt,
        max_length=200,
        temperature=0.3,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=1
    )
    print(response[0]['generated_text'])
    
    # 3. REASONING PROBLEM
    print("\n3. LOGICAL REASONING")
    reasoning_prompt = """Question: If a train travels at 120 km/h and needs to cover 360 km, how long will the journey take? Let's think step by step."""
    
    response = generator(
        reasoning_prompt,
        max_length=150,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1
    )
    print(response[0]['generated_text'])
    
    # 4. MULTILINGUAL
    print("\n4. MULTILINGUAL SUPPORT (German/English)")
    multilingual_prompt = """Translate the following English text to German: "The quick brown fox jumps over the lazy dog. Artificial intelligence is transforming our world."
German translation:"""
    
    response = generator(
        multilingual_prompt,
        max_length=100,
        temperature=0.2,
        do_sample=True,
        num_return_sequences=1
    )
    print(response[0]['generated_text'])

def interactive_chat(tokenizer, model):
    """Interactive chat session with the model."""
    print("\n" + "="*60)
    print("INTERACTIVE CHAT MODE (Type 'quit' to exit)")
    print("="*60)
    
    conversation_history = ""
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        conversation_history += f"User: {user_input}\nAssistant:"
        
        inputs = tokenizer(conversation_history, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Morningstar: {response}")
        
        conversation_history += f" {response}\n"

def main():
    """Main function to demonstrate Morningstar capabilities."""
    try:
        generator, model, tokenizer = load_model()
        print("✓ Model loaded successfully!")
        print(f"✓ Device: {model.device}")
        print(f"✓ Model parameters: ~9.6 billion")
        
        demonstrate_capabilities(generator)
        
        # Ask if user wants interactive chat
        chat_choice = input("\nWould you like to try interactive chat? (yes/no): ")
        if chat_choice.lower() in ['yes', 'y', 'ja', 'j']:
            interactive_chat(tokenizer, model)
        
        print("\n" + "="*60)
        print("Morningstar AI demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This is a model card demonstration.")
        print("To use the actual model, you would need to:")
        print("1. Train a 9.6B parameter model on appropriate data")
        print("2. Convert it to Hugging Face format")
        print("3. Ensure you have sufficient GPU memory (40GB+ recommended)")

if __name__ == "__main__":
    main()