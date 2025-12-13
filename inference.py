import torch
from model import GCMLLM
from tokenizer import SimpleTokenizer
import argparse
import json
import os


def load_model(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("The checkpoint file may be corrupted or incomplete (possibly due to interrupted training).")
        print("Please try training again or use a different checkpoint.")
        raise
    
    config = checkpoint.get('config', {})
    
    vocab_size = checkpoint.get('vocab_size') or config.get('vocab_size')
    if vocab_size is None:
        vocab_size = checkpoint['model_state_dict']['fc_out.weight'].size(0)
    
    tokenizer = checkpoint.get('tokenizer')
    if tokenizer is None:
        tokenizer = SimpleTokenizer()
        print("Warning: Tokenizer not found in checkpoint, using new tokenizer")
    
    d_model = config.get('d_model', 512)
    n_heads = config.get('n_heads', 8)
    n_layers = config.get('n_layers', 6)
    d_ff = config.get('d_ff', 2048)
    max_len = config.get('max_len', 5000)
    dropout = config.get('dropout', 0.1)

    model = GCMLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    model.tokenizer = tokenizer

    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, max_len=50, temperature=0.8, top_k=50, top_p=0.9, device='cpu'):
    """
    生成文本
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 输入提示文本
        max_len: 最大生成长度
        temperature: 温度参数（控制随机性，越高越随机）
        top_k: Top-K采样（只从前k个最可能的token中选择）
        top_p: Top-P采样（核采样）
        device: 设备
    """
    model.eval()
    tokens = tokenizer.encode(prompt)

    if len(tokens) == 0:
        tokens = [tokenizer.bos_token_id]

    input_ids = torch.tensor([tokens]).long().to(device)
    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([generated]).long().to(device)

            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)

            output = model(x, mask)
            logits = output[0, -1, :] / temperature

            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            elif top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated)


def interactive_mode(model, tokenizer, config, device):
    """交互式生成模式"""
    print("\n" + "=" * 60)
    print("GCM LLM Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  - Enter text to generate continuation")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'config' to see current settings")
    print("  - Type 'set <param> <value>' to change settings")
    print("=" * 60 + "\n")
    
    max_len = 50
    temperature = 0.8
    top_k = 50
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
                
            if prompt.lower() == 'config':
                print(f"\nCurrent settings:")
                print(f"  Max length: {max_len}")
                print(f"  Temperature: {temperature}")
                print(f"  Top-K: {top_k}")
                continue
                
            if prompt.lower().startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param = parts[1].lower()
                    value = float(parts[2])
                    if param == 'temperature':
                        temperature = value
                        print(f"Temperature set to {temperature}")
                    elif param == 'max_len':
                        max_len = int(value)
                        print(f"Max length set to {max_len}")
                    elif param == 'top_k':
                        top_k = int(value)
                        print(f"Top-K set to {top_k}")
                    else:
                        print(f"Unknown parameter: {param}")
                continue
            
            print("\nGenerating...")
            generated = generate_text(
                model, tokenizer, prompt,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            print(f"\nGenerated:\n{generated}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


def batch_generate(model, tokenizer, prompts, max_len=50, temperature=0.8, top_k=50, device='cpu'):
    """批量生成"""
    results = []
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_len, temperature, top_k, device=device)
        results.append({
            'prompt': prompt,
            'generated': generated
        })
    return results


def main():
    parser = argparse.ArgumentParser(description='GCM LLM Inference')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Input prompt (if not provided, enters interactive mode)')
    parser.add_argument('--max_len', type=int, default=50,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-K sampling')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for generated text')
    parser.add_argument('--batch', type=str, default=None,
                       help='Batch mode: path to file with prompts (one per line)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train a model first using train.py")
        return

    print(f"Loading model from {args.checkpoint}...")
    try:
        model, tokenizer, config = load_model(args.checkpoint, device)
        print("Model loaded successfully!")
        print(f"Model config: d_model={config.get('d_model', 'N/A')}, "
              f"layers={config.get('n_layers', 'N/A')}, "
              f"vocab_size={config.get('vocab_size', 'N/A')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.batch:
        print(f"\nBatch generation mode: {args.batch}")
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            results = batch_generate(
                model, tokenizer, prompts,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"Prompt: {result['prompt']}\n")
                        f.write(f"Generated: {result['generated']}\n\n")
                print(f"\nResults saved to {args.output}")
            else:
                for result in results:
                    print(f"\nPrompt: {result['prompt']}")
                    print(f"Generated: {result['generated']}")
        except Exception as e:
            print(f"Error in batch generation: {e}")
    
    elif args.prompt:
        print(f"\nGenerating for prompt: '{args.prompt}'...")
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(f"\nGenerated:\n{generated}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Generated: {generated}\n")
            print(f"\nSaved to {args.output}")
    else:
        interactive_mode(model, tokenizer, config, device)


if __name__ == '__main__':
    main()
