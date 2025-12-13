"""
超高效训练脚本 - 用1万行数据达到100万行效果
Ultra Efficient Training Script

使用方法:
python train_ultra.py --data_path data/1w.txt --epochs 100 --augmentation_factor 15
"""

import torch
import argparse
import os
import json
from datetime import datetime
from model import GCMLLM
from tokenizer import SimpleTokenizer
from data_loader import TextDataLoader
from ultra_efficient_trainer import ultra_efficient_train


def main():
    parser = argparse.ArgumentParser(description='Ultra Efficient Training - 1万行达到100万行效果')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (file or directory)')
    parser.add_argument('--data_format', type=str, default='auto',
                       choices=['auto', 'txt', 'json', 'jsonl'])
    parser.add_argument('--text_key', type=str, default='text')
    parser.add_argument('--line_mode', action='store_true',
                       help='Treat each line as a separate sample')
    parser.add_argument('--min_length', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=2000)
    
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    parser.add_argument('--augmentation_factor', type=int, default=15,
                       help='数据增强倍数（每个样本生成多少个增强版本）')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--use_rope', action='store_true', default=True,
                       help='Use RoPE position encoding')
    parser.add_argument('--use_flash', action='store_true',
                       help='Use Flash Attention')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=50)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 超高效训练系统")
    print("=" * 80)
    print(f"目标: 用 {args.data_path} 的1万行数据达到100万行数据的效果")
    print(f"数据增强倍数: {args.augmentation_factor}x")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n📂 加载数据...")
    
    use_line_mode = args.line_mode
    data_loader = TextDataLoader(
        min_length=args.min_length,
        max_length=args.max_length,
        line_mode=use_line_mode
    )
    
    if os.path.isfile(args.data_path):
        if args.data_format == 'auto':
            texts = data_loader.load_file(args.data_path, text_key=args.text_key)
        elif args.data_format == 'txt':
            texts = data_loader.load_txt(args.data_path)
        elif args.data_format == 'json':
            texts = data_loader.load_json(args.data_path, text_key=args.text_key)
        elif args.data_format == 'jsonl':
            texts = data_loader.load_jsonl(args.data_path, text_key=args.text_key)
        else:
            raise ValueError(f"Unsupported format: {args.data_format}")
    elif os.path.isdir(args.data_path):
        texts = data_loader.load_directory(args.data_path)
    else:
        raise ValueError(f"Data path not found: {args.data_path}")
    
    if len(texts) == 0 and not use_line_mode and os.path.isfile(args.data_path):
        print("\n⚠️  段落模式加载失败，自动切换到行模式...")
        use_line_mode = True
        data_loader = TextDataLoader(
            min_length=args.min_length,
            max_length=args.max_length,
            line_mode=True
        )
        
        if args.data_format == 'auto' or args.data_format == 'txt':
            texts = data_loader.load_txt(args.data_path)
        elif args.data_format == 'json':
            texts = data_loader.load_json(args.data_path, text_key=args.text_key)
        elif args.data_format == 'jsonl':
            texts = data_loader.load_jsonl(args.data_path, text_key=args.text_key)
        
        if len(texts) > 0:
            print(f"✓ 行模式成功加载 {len(texts)} 个样本")
        else:
            print("\n⚠️  尝试放宽长度限制...")
            data_loader = TextDataLoader(
                min_length=1,
                max_length=10000,
                line_mode=True
            )
            
            if args.data_format == 'auto' or args.data_format == 'txt':
                texts = data_loader.load_txt(args.data_path)
            elif args.data_format == 'json':
                texts = data_loader.load_json(args.data_path, text_key=args.text_key)
            elif args.data_format == 'jsonl':
                texts = data_loader.load_jsonl(args.data_path, text_key=args.text_key)
    
    if len(texts) == 0:
        print("\n❌ 错误: 无法加载任何有效数据!")
        print("\n可能的解决方案:")
        print("1. 检查数据文件格式是否正确")
        print("2. 使用 --line_mode 参数（如果每行是一个样本）")
        print("3. 调整 --min_length 和 --max_length 参数")
        print(f"   当前设置: min_length={args.min_length}, max_length={args.max_length}")
        print("4. 检查数据文件编码是否为 UTF-8")
        raise ValueError("No valid texts loaded!")
    
    print(f"✓ 加载了 {len(texts)} 个训练样本")
    if len(texts) > 0:
        avg_len = sum(len(t) for t in texts) / len(texts)
        min_len = min(len(t) for t in texts)
        max_len = max(len(t) for t in texts)
        print(f"  平均长度: {avg_len:.1f} 字符")
        print(f"  长度范围: {min_len} - {max_len} 字符")
        print(f"  使用模式: {'行模式' if use_line_mode else '段落模式'}")
    
    val_size = int(len(texts) * args.val_split)
    train_texts = texts[:-val_size] if val_size > 0 else texts
    val_texts = texts[-val_size:] if val_size > 0 else []
    
    print(f"✓ 训练集: {len(train_texts)} 样本")
    print(f"✓ 验证集: {len(val_texts)} 样本")
    
    print("\n📚 构建词汇表...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(train_texts, min_freq=1)
    print(f"✓ 词汇表大小: {tokenizer.vocab_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    print("\n🏗️  创建模型...")
    model = GCMLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        use_rope=args.use_rope,
        use_flash=args.use_flash,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    model.tokenizer = tokenizer
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型参数量: {num_params:,}")
    print(f"  模型大小: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    config = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'max_len': args.max_len,
        'dropout': args.dropout,
        'vocab_size': tokenizer.vocab_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'augmentation_factor': args.augmentation_factor,
        'use_amp': args.use_amp,
        'save_dir': args.save_dir,
        'save_interval': args.save_interval,
        'log_interval': args.log_interval
    }
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    config_path = os.path.join(args.save_dir, 'ultra_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置已保存到: {config_path}")
    
    print("\n" + "=" * 80)
    model, history = ultra_efficient_train(
        model=model,
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        device=device,
        config=config,
        num_original_samples=len(train_texts)
    )
    
    print("\n💾 保存模型...")
    final_path = os.path.join(args.save_dir, 'ultra_final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.vocab_size,
        'config': config,
        'history': history
    }, final_path)
    print(f"✓ 模型已保存到: {final_path}")
    
    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最终验证困惑度: {history['val_ppls'][-1]:.2f}" if history['val_ppls'] else "N/A")
    print("=" * 80)


if __name__ == '__main__':
    main()

