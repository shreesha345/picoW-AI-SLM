import os
import struct
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Byte Pair Encoding (BPE) Tokenizer
# =============================================================================

class BinTokenizer:
    def __init__(self, bin_path):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Tokenizer file not found: {bin_path}")
        with open(bin_path, 'rb') as f:
            self.max_token_length = struct.unpack('i', f.read(4))[0]
            self.vocab = []
            self.vocab_scores = []
            self.vocab_map = {}
            
            # tok512.bin contains exactly 512 tokens
            for i in range(512):
                score = struct.unpack('f', f.read(4))[0]
                length = struct.unpack('i', f.read(4))[0]
                token_bytes = f.read(length)
                token_str = token_bytes.decode('utf-8', errors='replace')
                self.vocab.append(token_str)
                self.vocab_scores.append(score)
                self.vocab_map[token_str] = i
                
    def encode(self, text, bos=True, eos=False):
        tokens = []
        if bos:
            tokens.append(1) # BOS token (<s>)
        
        if len(text) > 0:
            # dummy prefix space
            dummy = self.vocab_map.get(" ", -1)
            if dummy != -1:
                tokens.append(dummy)
                
        # Split text into characters/raw bytes and map to vocab IDs
        raw_tokens = []
        for c in text:
            tok_id = self.vocab_map.get(c, -1)
            if tok_id != -1:
                raw_tokens.append(tok_id)
            else:
                raw_tokens.append(ord(c) + 3) # Byte fallback
                
        tokens.extend(raw_tokens)
        
        # Iterative BPE merging
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            
            for j in range(len(tokens) - 1):
                pair_str = self.vocab[tokens[j]] + self.vocab[tokens[j+1]]
                pair_id = self.vocab_map.get(pair_str, -1)
                if pair_id != -1 and self.vocab_scores[pair_id] > best_score:
                    best_score = self.vocab_scores[pair_id]
                    best_id = pair_id
                    best_idx = j
                    
            if best_idx == -1:
                break
                
            tokens = tokens[:best_idx] + [best_id] + tokens[best_idx+2:]
            
        if eos:
            tokens.append(2) # EOS token (</s>)
        return tokens

    def decode(self, token_ids):
        res = []
        for t in token_ids:
            if t == 1: # BOS
                continue
            if t == 2: # EOS
                break
            res.append(self.vocab[t])
        return "".join(res)

# =============================================================================
# 2. PyTorch Llama-2 Model Definition (Matching C++ layout exactly)
# =============================================================================

class TinyLlama(nn.Module):
    def __init__(self, dim=64, hidden_dim=172, n_layers=5, n_heads=8, n_kv_heads=4, vocab_size=512, seq_len=128):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.head_size = dim // n_heads

        # Embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, dim)

        # RMSNorm weights
        self.rms_att_weight = nn.Parameter(torch.ones(n_layers, dim))
        self.rms_ffn_weight = nn.Parameter(torch.ones(n_layers, dim))

        # Query, Key, Value, Output weight matrices
        self.wq = nn.Parameter(torch.randn(n_layers, dim, dim) * 0.02)
        dim_kv = dim * n_kv_heads // n_heads
        self.wk = nn.Parameter(torch.randn(n_layers, dim_kv, dim) * 0.02)
        self.wv = nn.Parameter(torch.randn(n_layers, dim_kv, dim) * 0.02)
        self.wo = nn.Parameter(torch.randn(n_layers, dim, dim) * 0.02)

        # Feed Forward Network weights
        self.w1 = nn.Parameter(torch.randn(n_layers, hidden_dim, dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_layers, dim, hidden_dim) * 0.02)
        self.w3 = nn.Parameter(torch.randn(n_layers, hidden_dim, dim) * 0.02)

        # Final RMSNorm
        self.rms_final_weight = nn.Parameter(torch.ones(dim))

    def apply_rope(self, x):
        # x shape: (batch, seq_len, heads, head_size)
        batch_size, seq_len, heads, head_size = x.shape
        pos = torch.arange(seq_len, dtype=torch.float32, device=x.device).view(1, seq_len, 1, 1)
        i = torch.arange(0, head_size, 2, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        freq = 1.0 / (10000.0 ** (i / float(head_size)))
        angle = pos * freq
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        rx0 = x0 * cos - x1 * sin
        rx1 = x0 * sin + x1 * cos
        
        rx = torch.empty_like(x)
        rx[..., 0::2] = rx0
        rx[..., 1::2] = rx1
        return rx

    def forward(self, tokens, targets=None):
        batch_size, seq_len = tokens.shape
        x = self.token_embedding_table(tokens)

        for l in range(self.n_layers):
            # Attention Input RMSNorm
            xb = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_att_weight[l]

            # Q, K, V projections
            q = F.linear(xb, self.wq[l]).view(batch_size, seq_len, self.n_heads, self.head_size)
            k = F.linear(xb, self.wk[l]).view(batch_size, seq_len, self.n_kv_heads, self.head_size)
            v = F.linear(xb, self.wv[l]).view(batch_size, seq_len, self.n_kv_heads, self.head_size)

            # Apply RoPE
            q = self.apply_rope(q)
            k = self.apply_rope(k)

            # Transpose for attention computation
            q = q.transpose(1, 2) # (batch, heads, seq_len, head_size)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Grouped Query Attention (GQA)
            if self.n_heads != self.n_kv_heads:
                num_queries_per_kv = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
            
            # Causal Mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device) * float('-inf'), diagonal=1)
            scores = scores + mask

            probs = F.softmax(scores, dim=-1)
            output = torch.matmul(probs, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

            # Residual connection
            x = x + F.linear(output, self.wo[l])

            # FFN Input RMSNorm
            xb = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_ffn_weight[l]
            
            # SwiGLU: w2( SiLU(w1(xb)) * w3(xb) )
            h1 = F.linear(xb, self.w1[l])
            h2 = F.linear(xb, self.w3[l])
            ffn_out = F.silu(h1) * h2
            x = x + F.linear(ffn_out, self.w2[l])

        # Final RMSNorm
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_final_weight
        
        # Classifier (Shared weights)
        logits = F.linear(x, self.token_embedding_table.weight)

        loss = None
        if targets is not None:
            # Shift targets and logits for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_targets.view(-1))

        return logits, loss


# =============================================================================
# 3. Conversational Dataset & Training Loop
# =============================================================================

def train_chatbot():
    print("Loading tokenizer...")
    tokenizer = BinTokenizer("../tok512.bin")
    
    # Base dialogues for our purpose: A smart home controller & friendly companion
    base_dialogues = [
        # Greet
        ("hello", "Hello! How can I help you?"),
        ("hi", "Hi there! What can I do for you?"),
        ("hey", "Hey! Nice to talk to you."),
        ("how are you?", "I am doing great, thank you!"),
        ("who are you?", "I am PicoAI, your smart companion."),
        ("what is your name?", "My name is PicoAI."),
        ("are you human?", "No, I am a tiny AI running on a Pico W."),
        # Smart home
        ("turn on light", "Light turned on."),
        ("turn off light", "Light turned off."),
        ("turn on lights", "Lights turned on."),
        ("turn off lights", "Lights turned off."),
        ("turn on fan", "Fan turned on."),
        ("turn off fan", "Fan turned off."),
        ("lock door", "Door is now locked."),
        ("unlock door", "Door is now unlocked."),
        ("open door", "Door opened."),
        ("close door", "Door closed."),
        # Status / Sensors
        ("status report", "All systems online and healthy."),
        ("check status", "Status: active. Battery: 100%."),
        ("is it hot?", "Yes, temp is 32C."),
        ("is it cold?", "No, temp is pleasant."),
        ("is it raining?", "No, it is sunny outside."),
        ("what time is it?", "Time is just a concept for me."),
        ("check weather", "Weather: sunny and warm."),
        # Fun
        ("do you like cookies?", "Yes, but I cannot eat them!"),
        ("tell me a joke", "Why did Pico cross the road? To compute!"),
        ("say cheese", "Cheese! Smile!"),
        ("help me", "Sure! What do you need?"),
        ("goodbye", "Goodbye! Have a nice day."),
        ("bye", "Bye! See you later.")
    ]

    # Synthesize variation dataset to make training robust
    dataset = []
    for user, bot in base_dialogues:
        # Standard
        dataset.append((user, bot))
        # Add politeness prefix
        dataset.append((f"please {user}", bot))
        dataset.append((f"hey pico {user}", bot))
        dataset.append((f"pico {user}", bot))
        # Add question marks variations
        if user.endswith("?"):
            dataset.append((user[:-1], bot))
        else:
            dataset.append((f"{user}?", bot))

    print(f"Generated {len(dataset)} training dialogue examples.")

    # Max context sequence length
    max_len = 64
    
    tokenized_inputs = []
    tokenized_targets = []
    
    for user, bot in dataset:
        # Format: <s>[INST] {user} [/INST] {bot} </s>
        prompt = f"[INST] {user} [/INST]"
        response = f" {bot}"
        
        prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
        response_tokens = tokenizer.encode(response, bos=False, eos=True)
        
        input_ids = prompt_tokens + response_tokens
        
        # Targets: mask prompt tokens with -100 so cross_entropy ignores them
        target_ids = [-100] * len(prompt_tokens) + response_tokens
        
        # Pad to max_len
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [2] * pad_len # Pad with EOS/2
            target_ids = target_ids + [-100] * pad_len
        else:
            input_ids = input_ids[:max_len]
            target_ids = target_ids[:max_len]
            
        tokenized_inputs.append(input_ids)
        tokenized_targets.append(target_ids)
        
    x_train = torch.tensor(tokenized_inputs, dtype=torch.long)
    y_train = torch.tensor(tokenized_targets, dtype=torch.long)

    # Initialize model
    model = TinyLlama(dim=64, hidden_dim=172, n_layers=5, n_heads=8, n_kv_heads=4, vocab_size=512, seq_len=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    
    print(f"Training model on {device}...")
    model.train()
    
    epochs = 150
    batch_size = 32
    
    for epoch in range(epochs):
        permutation = torch.randperm(x_train.size(0))
        epoch_loss = 0
        batches = 0
        
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = x_train[indices]
            batch_y = y_train[indices]
            
            optimizer.zero_grad()
            _, loss = model(batch_x, batch_y)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/batches:.4f}")

    # Evaluate some samples
    model.eval()
    print("\n--- Testing Model Predictions ---")
    test_prompts = ["hello", "turn on light", "tell me a joke", "status report"]
    for prompt in test_prompts:
        full_prompt = f"[INST] {prompt} [/INST]"
        tokens = tokenizer.encode(full_prompt, bos=True, eos=False)
        inp = torch.tensor([tokens], dtype=torch.long, device=device)
        
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(30):
                logits, _ = model(torch.tensor([generated], dtype=torch.long, device=device))
                next_token = torch.argmax(logits[0, -1, :]).item()
                generated.append(next_token)
                if next_token == 2: # EOS
                    break
        response = tokenizer.decode(generated[len(tokens):])
        print(f"Prompt: {repr(prompt)}")
        print(f"Bot:    {repr(response.strip())}\n")

    return model

# =============================================================================
# 4. Exporter to C Header File (model_weights.h)
# =============================================================================

def export_model(model, bin_path, header_path):
    print(f"Exporting model weights to binary: {bin_path}...")
    
    # Precompute RoPE cos/sin frequencies for header size = 8, seq_len = 512
    seq_len = 512
    head_size = 8
    
    freq_cis_real = np.zeros((seq_len, head_size // 2), dtype=np.float32)
    freq_cis_imag = np.zeros((seq_len, head_size // 2), dtype=np.float32)
    
    for pos in range(seq_len):
        for i in range(head_size // 2):
            freq = 1.0 / (10000.0 ** (2. * i / head_size))
            val = pos * freq
            freq_cis_real[pos, i] = np.cos(val)
            freq_cis_imag[pos, i] = np.sin(val)

    with open(bin_path, 'wb') as f:
        # Write config header (dim=64, hidden_dim=172, n_layers=5, n_heads=8, n_kv_heads=4, vocab_size=512, seq_len=512)
        # vocab_size is positive to denote shared weights
        header = struct.pack('iiiiiii', 64, 172, 5, 8, 4, 512, 512)
        f.write(header)
        
        # Helper to write tensor
        def write_tensor(tensor):
            f.write(tensor.detach().cpu().numpy().astype(np.float32).tobytes())
            
        write_tensor(model.token_embedding_table.weight)
        write_tensor(model.rms_att_weight)
        write_tensor(model.wq)
        write_tensor(model.wk)
        write_tensor(model.wv)
        write_tensor(model.wo)
        write_tensor(model.rms_ffn_weight)
        write_tensor(model.w1)
        write_tensor(model.w2)
        write_tensor(model.w3)
        write_tensor(model.rms_final_weight)
        f.write(freq_cis_real.tobytes())
        f.write(freq_cis_imag.tobytes())

    print(f"Converting {bin_path} to C header file: {header_path}...")
    with open(bin_path, 'rb') as f:
        data = f.read()

    with open(header_path, 'w') as f:
        f.write(f"// Auto-generated from {os.path.basename(bin_path)}\n")
        f.write(f"// Size: {len(data):,} bytes\n")
        f.write("#ifndef MODEL_DATA_H\n")
        f.write("#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int model_data_len = {len(data)};\n\n")
        f.write("const uint8_t __attribute__((aligned(4))) model_data[] = {\n")
        
        # Write bytes in chunks of 16 for readable formatting
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")
            
        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
        
    print("Export complete!")

if __name__ == "__main__":
    trained_model = train_chatbot()
    # Export weights to both the local weights directory and the global C++ build target
    export_model(trained_model, "../chatbot.bin", "../../model_weights.h")
