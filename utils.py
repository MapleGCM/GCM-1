import torch
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


def plot_training_history(checkpoint_path, save_path=None):
    """绘制训练历史曲线"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("The checkpoint file may be corrupted or incomplete.")
        return
    
    train_losses = checkpoint.get('train_losses', [])
    val_ppls = checkpoint.get('val_ppls', [])
    
    if not train_losses and not val_ppls:
        print("No training history found in checkpoint")
        return
    
    has_val = len(val_ppls) > 0
    
    if has_val:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_train, ax_val = axes
    else:
        fig, ax_train = plt.subplots(1, 1, figsize=(10, 5))
        ax_val = None
    
    epochs = range(1, len(train_losses) + 1)
    
    ax_train.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1.5)
    ax_train.set_xlabel('Epoch', fontsize=11)
    ax_train.set_ylabel('Loss', fontsize=11)
    ax_train.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)
    
    if has_val and ax_val is not None:
        val_epochs = range(1, len(val_ppls) + 1)
        ax_val.plot(val_epochs, val_ppls, 'r-', label='Validation Perplexity', linewidth=1.5)
        ax_val.set_xlabel('Epoch', fontsize=11)
        ax_val.set_ylabel('Perplexity', fontsize=11)
        ax_val.set_title('Validation Perplexity', fontsize=12, fontweight='bold')
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def analyze_model(checkpoint_path):
    """分析模型信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("The checkpoint file may be corrupted or incomplete.")
        return
    
    print("=" * 60)
    print("Model Analysis")
    print("=" * 60)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    
    config = checkpoint.get('config', {})
    print(f"\nModel Configuration:")
    print(f"  d_model: {config.get('d_model', 'N/A')}")
    print(f"  n_heads: {config.get('n_heads', 'N/A')}")
    print(f"  n_layers: {config.get('n_layers', 'N/A')}")
    print(f"  d_ff: {config.get('d_ff', 'N/A')}")
    print(f"  vocab_size: {config.get('vocab_size', 'N/A')}")
    
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"\nTraining Info:")
    print(f"  Epoch: {epoch}")
    
    if 'train_losses' in checkpoint:
        losses = checkpoint['train_losses']
        print(f"  Final training loss: {losses[-1]:.4f}" if losses else "  N/A")
    
    if 'val_ppls' in checkpoint:
        ppls = checkpoint['val_ppls']
        print(f"  Best validation PPL: {min(ppls):.2f}" if ppls else "  N/A")
    
    state_dict = checkpoint.get('model_state_dict', {})
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024*1024):.2f} MB (float32)")
    
    print("=" * 60)


def export_model_onnx(checkpoint_path, output_path, sample_input_len=128):
    """导出模型为 ONNX 格式（实验性）"""
    try:
        import onnx
        from onnxruntime import InferenceSession
    except ImportError:
        print("ONNX export requires: pip install onnx onnxruntime")
        return
    
    from model import GCMLLM
    from inference import load_model
    
    try:
        device = torch.device('cpu')
        model, tokenizer, config = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    vocab_size = config.get('vocab_size')
    d_model = config.get('d_model', 512)
    
    model.eval()
    
    sample_input = torch.randint(0, vocab_size, (1, sample_input_len)).long()
    sample_mask = torch.tril(torch.ones(sample_input_len, sample_input_len))
    
    try:
        torch.onnx.export(
            model,
            (sample_input, sample_mask),
            output_path,
            input_names=['input_ids', 'mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'mask': {0: 'seq_len', 1: 'seq_len'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=11
        )
        print(f"Model exported to {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python utils.py analyze <checkpoint_path>")
        print("  python utils.py plot <checkpoint_path> [output_path]")
        print("  python utils.py export_onnx <checkpoint_path> <output_path>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'analyze':
        if len(sys.argv) < 3:
            print("Error: checkpoint path required")
            sys.exit(1)
        analyze_model(sys.argv[2])
    
    elif command == 'plot':
        if len(sys.argv) < 3:
            print("Error: checkpoint path required")
            sys.exit(1)
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_training_history(sys.argv[2], save_path)
    
    elif command == 'export_onnx':
        if len(sys.argv) < 4:
            print("Error: checkpoint path and output path required")
            sys.exit(1)
        export_model_onnx(sys.argv[2], sys.argv[3])
    
    else:
        print(f"Unknown command: {command}")

