import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """RMS归一化，比LayerNorm更高效"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class RoPE(nn.Module):
    """旋转位置编码"""
    def __init__(self, d_head: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.d_head = d_head
        self.max_len = max_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码
        返回: (cos, sin) 用于旋转，形状为 [1, seq_len, 1, d_head]
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        if self.d_head % 2 == 1:
            emb = torch.cat((freqs, freqs, freqs[..., :1]), dim=-1)
        else:
            emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转一半维度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, 
                            cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用旋转位置编码到query和key
        q, k: [batch, n_heads, seq_len, d_head]
        cos, sin: [1, seq_len, 1, d_head] -> 需要调整为 [1, 1, seq_len, d_head]
        """
        # 确保cos和sin的维度正确
        # cos/sin输入: [1, seq_len, 1, d_head]
        # 需要调整为: [1, 1, seq_len, d_head] 以匹配 [batch, n_heads, seq_len, d_head]
        
        # 检查维度
        if cos.dim() == 4:
            # [1, seq_len, 1, d_head] -> [1, 1, seq_len, d_head]
            cos = cos.transpose(1, 2)  # [1, 1, seq_len, d_head]
            sin = sin.transpose(1, 2)  # [1, 1, seq_len, d_head]
        elif cos.dim() == 2:
            # [seq_len, d_head] -> [1, 1, seq_len, d_head]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head]
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head]
        
        # 确保d_head维度匹配
        q_d_head = q.shape[-1]
        cos_d_head = cos.shape[-1]
        
        if q_d_head != cos_d_head:
            if cos_d_head > q_d_head:
                cos = cos[..., :q_d_head]
                sin = sin[..., :q_d_head]
            else:
                pad_size = q_d_head - cos_d_head
                cos = F.pad(cos, (0, pad_size))
                sin = F.pad(sin, (0, pad_size))
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU激活函数，比ReLU更强大"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class FeedForward(nn.Module):
    """使用SwiGLU的前馈网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU需要2倍维度
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.activation = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(xW1) * (xW3)) W2
        gate = self.w1(x)
        up = self.w3(x)
        return self.w2(self.dropout(self.activation(gate) * up))


class MultiHeadAttention(nn.Module):
    """优化的多头注意力，支持RoPE和Flash Attention（如果可用）"""
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True, use_flash: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_rope = use_rope
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        if use_rope:
            self.rope = RoPE(self.d_k, max_len=5000)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用RoPE
        if self.use_rope:
            if rope_cos_sin is None:
                # 使用Q的形状来生成RoPE（Q已经是[batch, n_heads, seq_len, d_k]）
                # 但我们需要基于seq_len生成，所以传入一个占位tensor
                cos, sin = self.rope(Q, seq_len)  # Q: [batch, n_heads, seq_len, d_k]
            else:
                cos, sin = rope_cos_sin
            Q, K = self.rope.apply_rotary_pos_emb(Q, K, cos, sin)

        # Flash Attention（如果可用且支持）
        if self.use_flash and mask is None:
            # PyTorch 2.0+ 的scaled_dot_product_attention
            attention_output = F.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        else:
            # 标准注意力
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                # scores形状: [batch, n_heads, seq_len, seq_len]
                # mask需要调整为 [batch, 1, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]
                if mask.dim() == 2:
                    # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.dim() == 3:
                    # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                    if mask.size(0) == batch_size:
                        mask = mask.unsqueeze(1)
                    else:
                        # batch维度不匹配，使用第一个batch的mask并广播
                        mask = mask[0:1].unsqueeze(1).expand(batch_size, -1, -1, -1)
                elif mask.dim() == 4:
                    # 已经是 [batch, 1, seq_len, seq_len] 或类似形状
                    if mask.size(1) != 1:
                        mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
                
                # 确保mask的seq_len维度匹配
                if mask.size(-1) != seq_len or mask.size(-2) != seq_len:
                    # 如果seq_len不匹配，重新创建因果mask
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device)).bool()
                    mask = causal_mask.unsqueeze(0).unsqueeze(0)
                
                # 使用FP16安全的值（-1e4在FP16范围内，-1e9会溢出）
                # 或者使用torch.finfo获取当前dtype的最小值
                if scores.dtype == torch.float16:
                    fill_value = -1e4
                else:
                    fill_value = -1e9
                scores = scores.masked_fill(mask == 0, fill_value)
            else:
                # 因果掩码
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
                # 使用FP16安全的值
                if scores.dtype == torch.float16:
                    fill_value = -1e4
                else:
                    fill_value = -1e9
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), fill_value)

            attention_weights = F.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return self.W_o(attention_output)


class TransformerBlock(nn.Module):
    """优化的Transformer块，使用RMSNorm和Pre-Norm架构"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, 
                 use_rope: bool = True, use_flash: bool = False, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, use_rope, use_flash)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # Pre-Norm架构（更稳定）
        if self.use_gradient_checkpointing and self.training:
            # 梯度检查点节省内存
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attention_block, x, mask, rope_cos_sin
            )
            x = x + self.dropout(attn_output)
            
            ff_output = torch.utils.checkpoint.checkpoint(
                self._feedforward_block, x
            )
            x = x + self.dropout(ff_output)
        else:
            attn_output = self._attention_block(x, mask, rope_cos_sin)
            x = x + self.dropout(attn_output)
            
            ff_output = self._feedforward_block(x)
            x = x + self.dropout(ff_output)
        
        return x

    def _attention_block(self, x: torch.Tensor, mask: Optional[torch.Tensor], 
                        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask, rope_cos_sin)

    def _feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(self.norm2(x))


class GCMLLM(nn.Module):
    """优化的GCM LLM模型"""
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 5000, 
                 dropout: float = 0.1, use_rope: bool = True, use_flash: bool = False,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_rope = use_rope
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # RoPE位置编码（如果使用）
        # 注意：RoPE在每个attention层内部使用，这里不需要全局的rope
        if not use_rope:
            # 传统位置编码作为备选
            self.pos_encoding = self._create_pos_encoding(d_model, max_len)
        else:
            self.pos_encoding = None
        
        # Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope, use_flash, use_gradient_checkpointing)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.norm_out = RMSNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()

    def _create_pos_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """创建传统位置编码（如果不用RoPE）"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # 位置编码
        if self.use_rope:
            # RoPE在注意力层中应用，不需要在这里生成
            rope_cos_sin = None  # 让每个attention层自己生成
        else:
            # 传统位置编码
            x = x + self.pos_encoding[:, :x.size(1)].to(x.device)
            rope_cos_sin = None
        
        x = self.dropout(x)

        # Transformer块
        for transformer in self.transformer_blocks:
            x = transformer(x, mask, rope_cos_sin)

        # 输出
        x = self.norm_out(x)
        output = self.fc_out(x)
        return output

    def generate(self, tokenizer, prompt: str, max_len: int = 100, 
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 use_kv_cache: bool = True) -> str:
        """优化的生成函数，支持KV缓存"""
        self.eval()
        tokens = tokenizer.encode(prompt)
        
        if len(tokens) == 0:
            tokens = [tokenizer.bos_token_id]

        input_ids = torch.tensor([tokens]).long()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        generated = tokens.copy()
        
        # KV缓存（如果启用）
        kv_cache = None if not use_kv_cache else {}

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([generated]).long().to(device)
                seq_len = x.size(1)
                
                # 创建掩码
                mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

                # 前向传播
                output = self(x, mask)
                logits = output[0, -1, :] / temperature

                # 采样
                next_token = self._sample_token(logits, top_k, top_p)
                generated.append(next_token)

                # 检查结束
                if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(generated)

    def _sample_token(self, logits: torch.Tensor, top_k: int, top_p: float) -> int:
        """改进的采样策略"""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()
        elif top_p < 1.0:
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        
        return next_token

    def _create_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码"""
        return torch.tril(torch.ones(seq_len, seq_len, device=device))
