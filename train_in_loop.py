import os
import struct
import math
import json
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
            dummy = self.vocab_map.get(" ", -1)
            if dummy != -1:
                tokens.append(dummy)
                
        raw_tokens = []
        for c in text:
            tok_id = self.vocab_map.get(c, -1)
            if tok_id != -1:
                raw_tokens.append(tok_id)
            else:
                raw_tokens.append(ord(c) + 3) # Byte fallback
                
        tokens.extend(raw_tokens)
        
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
    def __init__(self, dim=64, hidden_dim=172, n_layers=5, n_heads=8, n_kv_heads=2, vocab_size=512, seq_len=256):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.head_size = dim // n_heads

        self.token_embedding_table = nn.Embedding(vocab_size, dim)
        self.rms_att_weight = nn.Parameter(torch.ones(n_layers, dim))
        self.rms_ffn_weight = nn.Parameter(torch.ones(n_layers, dim))

        self.wq = nn.Parameter(torch.randn(n_layers, dim, dim) * 0.02)
        dim_kv = dim * n_kv_heads // n_heads
        self.wk = nn.Parameter(torch.randn(n_layers, dim_kv, dim) * 0.02)
        self.wv = nn.Parameter(torch.randn(n_layers, dim_kv, dim) * 0.02)
        self.wo = nn.Parameter(torch.randn(n_layers, dim, dim) * 0.02)

        self.w1 = nn.Parameter(torch.randn(n_layers, hidden_dim, dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_layers, dim, hidden_dim) * 0.02)
        self.w3 = nn.Parameter(torch.randn(n_layers, hidden_dim, dim) * 0.02)
        self.rms_final_weight = nn.Parameter(torch.ones(dim))

    def apply_rope(self, x):
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
            xb = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_att_weight[l]
            q = F.linear(xb, self.wq[l]).view(batch_size, seq_len, self.n_heads, self.head_size)
            k = F.linear(xb, self.wk[l]).view(batch_size, seq_len, self.n_kv_heads, self.head_size)
            v = F.linear(xb, self.wv[l]).view(batch_size, seq_len, self.n_kv_heads, self.head_size)

            q = self.apply_rope(q)
            k = self.apply_rope(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.n_heads != self.n_kv_heads:
                num_queries_per_kv = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(num_queries_per_kv, dim=1)
                v = v.repeat_interleave(num_queries_per_kv, dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device) * float('-inf'), diagonal=1)
            scores = scores + mask

            probs = F.softmax(scores, dim=-1)
            output = torch.matmul(probs, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
            x = x + F.linear(output, self.wo[l])

            xb = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_ffn_weight[l]
            h1 = F.linear(xb, self.w1[l])
            h2 = F.linear(xb, self.w3[l])
            ffn_out = F.silu(h1) * h2
            x = x + F.linear(ffn_out, self.w2[l])

        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.rms_final_weight
        logits = F.linear(x, self.token_embedding_table.weight)

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_targets.view(-1))

        return logits, loss

# =============================================================================
# 3. Helper & API Functions
# =============================================================================

def sanitize_text(text):
    mapping = {
        '‘': "'", '’': "'", '“': '"', '”': '"',
        '—': '-', '–': '-', '…': '...', '•': '*',
        'é': 'e', 'è': 'e', 'á': 'a', 'à': 'a',
    }
    for k, v in mapping.items():
        text = text.replace(k, v)
    sanitized = []
    for c in text:
        val = ord(c)
        if val < 256 and val + 3 < 512:
            sanitized.append(c)
        else:
            sanitized.append('?')
    return "".join(sanitized)

def query_teacher(prompt, format_json=False):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma4:31b-cloud",
        "prompt": prompt,
        "stream": False
    }
    if format_json:
        data["format"] = "json"
        
    try:
        response = requests.post(url, json=data, timeout=45)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error querying teacher API: {e}")
    return ""

def parse_json_response(response_str):
    cleaned = response_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            valid = []
            for item in data:
                if isinstance(item, dict) and "prompt" in item and "response" in item:
                    valid.append({
                        "prompt": sanitize_text(item["prompt"]),
                        "response": sanitize_text(item["response"])
                    })
            return valid
    except Exception as e:
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1:
            try:
                data = json.loads(cleaned[start:end+1])
                if isinstance(data, list):
                    valid = []
                    for item in data:
                        if isinstance(item, dict) and "prompt" in item and "response" in item:
                            valid.append({
                                "prompt": sanitize_text(item["prompt"]),
                                "response": sanitize_text(item["response"])
                            })
                    return valid
            except:
                pass
    return []

# =============================================================================
# 4. Evaluation Loop logic
# =============================================================================

def generate_student_response(model, tokenizer, prompt, device='cpu', max_tokens=128):
    model.eval()
    full_prompt = f"[INST] {prompt} [/INST]"
    tokens = tokenizer.encode(full_prompt, bos=True, eos=False)
    
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(max_tokens):
            inp_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(inp_tensor)
            next_token = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_token)
            if next_token == 2: # EOS
                break
    response = tokenizer.decode(generated[len(tokens):])
    return response.strip()

def evaluate_student(prompt, response, category):
    response_lower = response.lower()
    
    # 1. First run high-speed heuristic checks to catch complete failure modes early
    if len(response) < 3 or "]" in response or "[" in response:
        print(f"Heuristic FAIL: Response '{response}' is empty or invalid.")
        return False, get_default_corrective_examples(prompt, category)
        
    if category == "Name":
        if "shreesha" not in response_lower:
            print(f"Heuristic FAIL: Name query response '{response}' does not contain 'shreesha'.")
            return False, get_default_corrective_examples(prompt, category)
            
    elif category == "Description":
        if not ("pico" in response_lower or "model" in response_lower or "llama" in response_lower):
            print(f"Heuristic FAIL: Description response '{response}' doesn't mention Pico or model.")
            return False, get_default_corrective_examples(prompt, category)
            
    elif category == "Greeting":
        # Ensure it greets AND asks how they are doing (requires "how" or "doing")
        has_greet = any(w in response_lower for w in ["hello", "hi", "greetings", "hey"])
        has_question = any(w in response_lower for w in ["how", "doing"])
        if not (has_greet and has_question):
            print(f"Heuristic FAIL: Greeting response '{response}' lacks greeting and 'how/doing' question.")
            return False, get_default_corrective_examples(prompt, category)

    # 2. Call the Teacher Model to evaluate and give feedback
    teacher_prompt = f"""You are an AI teacher grading a student chatbot.
We want the chatbot to behave as follows:
- Greeting prompts: greet the user back AND ask how they are doing AND say something welcoming (e.g. "Hello! I am doing well, thank you! How are you doing? I hope you are having a great day.").
- Name prompts: say its name is "shreesha" and describe its role (e.g. "My name is shreesha. I am your companion chatbot here to assist you with whatever you need.").
- Description prompts: describe itself in detail (e.g. "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights.").

The prompt asked: "{prompt}"
The student answered: "{response}"

Analyze the answer.
If the student answered correctly and met all the behavior criteria above, reply with:
GRADE: PASS

If the student failed, reply with:
GRADE: FAIL
Then list 3 new training prompt-response examples that teach the correct detailed behavior. Make responses conversational, complete, and around 15 to 30 words. Format these 3 examples strictly as a JSON array:
[
  {{"prompt": "...", "response": "..."}},
  {{"prompt": "...", "response": "..."}},
  {{"prompt": "...", "response": "..."}}
]
"""
    teacher_res = query_teacher(teacher_prompt)
    print(f"Teacher Feedback on prompt '{prompt}':")
    print(f"Student response: '{response}'")
    print(teacher_res)
    print("-" * 50)
    
    if "GRADE: PASS" in teacher_res:
        return True, []
        
    new_examples = parse_json_response(teacher_res)
    if not new_examples or len(new_examples) < 3:
        new_examples = get_default_corrective_examples(prompt, category)
        
    return False, new_examples

def get_default_corrective_examples(prompt, category):
    if category == "Name":
        return [
            {"prompt": prompt, "response": "My name is shreesha. I am your companion chatbot here to assist you with whatever you need."},
            {"prompt": "who are you?", "response": "My name is shreesha, and I am a tiny model running on Raspberry Pi Pico W. I am here to help you."},
            {"prompt": "tell me your name", "response": "My name is shreesha. I am your companion chatbot here to assist you with whatever you need."}
        ]
    elif category == "Description":
        return [
            {"prompt": prompt, "response": "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights."},
            {"prompt": "describe yourself", "response": "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights."},
            {"prompt": "what are you?", "response": "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights."}
        ]
    else:
        return [
            {"prompt": prompt, "response": "Hello! I am doing well, thank you! How are you doing today? I hope you are having a wonderful day."},
            {"prompt": "hello, how are you?", "response": "Hello! I am doing well, thank you! How are you doing? I hope everything is going great with you."},
            {"prompt": "hi there", "response": "Hi! Nice to meet you. How are you doing today? I am ready to help you."}
        ]

# =============================================================================
# 5. Exporter to C Header File (model_weights.h)
# =============================================================================

def export_model(model, bin_path, header_path):
    print(f"Exporting model weights to binary: {bin_path}...")
    seq_len = 256
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
        # Header layout: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        header = struct.pack('iiiiiii', 64, 172, 5, 8, 2, 512, 256)
        f.write(header)
        
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
        
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")
            
        f.write("};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
        
    print("Export complete!")

# =============================================================================
# 6. Main Self-Training Loop Pipeline
# =============================================================================

def train_student_model(model, tokenizer, dataset, epochs=150, lr=0.005, batch_size=32, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    max_len = 256
    tokenized_inputs = []
    tokenized_targets = []
    
    # Add variations for prompts to make student robust
    expanded_dataset = []
    for item in dataset:
        u, b = item["prompt"], item["response"]
        expanded_dataset.append((u, b))
        expanded_dataset.append((f"please {u}", b))
        expanded_dataset.append((f"pico {u}", b))
        expanded_dataset.append((f"hey pico {u}", b))
        if u.endswith("?"):
            expanded_dataset.append((u[:-1], b))
        else:
            expanded_dataset.append((f"{u}?", b))
            
    # De-duplicate expanded_dataset to prevent conflicting labels!
    unique_expanded = {}
    for p, r in expanded_dataset:
        p_clean = p.strip().lower()
        unique_expanded[p_clean] = r.strip()
    expanded_dataset = [(p, r) for p, r in unique_expanded.items()]
    
    for prompt, response in expanded_dataset:
        prompt_full = f"[INST] {prompt} [/INST]"
        response_full = f" {response}"
        
        p_toks = tokenizer.encode(prompt_full, bos=True, eos=False)
        r_toks = tokenizer.encode(response_full, bos=False, eos=True)
        
        input_ids = p_toks + r_toks
        target_ids = [-100] * len(p_toks) + r_toks
        
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [2] * pad_len
            target_ids = target_ids + [-100] * pad_len
        else:
            input_ids = input_ids[:max_len]
            target_ids = target_ids[:max_len]
            
        tokenized_inputs.append(input_ids)
        tokenized_targets.append(target_ids)
        
    x_train = torch.tensor(tokenized_inputs, dtype=torch.long, device=device)
    y_train = torch.tensor(tokenized_targets, dtype=torch.long, device=device)
    
    print(f"Training on {x_train.size(0)} dialogue examples...")
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Avg Loss: {epoch_loss/batches:.4f}")
            
    return model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = BinTokenizer("../tok512.bin")
    
    # Generate Initial Dataset
    print("\n--- Generating Initial Training Dataset from Teacher (gemma4:31b-cloud) ---")
    
    greetings_prompt = """Generate a JSON list of 20 distinct user greeting prompts and friendly, conversational responses (around 15 to 30 words) that greet the user back, ask how they are doing, and say something welcoming.
Format strictly as a JSON array of objects with keys "prompt" and "response". Do not output markdown blocks or other text, just the raw JSON array.
"""
    name_prompt = """Generate a JSON list of 20 distinct user name query prompts.
Every single response must be exactly: "My name is shreesha. I am your companion chatbot here to assist you with whatever you need."
Format strictly as a JSON array of objects with keys "prompt" and "response".
"""
    desc_prompt = """Generate a JSON list of 20 distinct user self-description query prompts.
Every single response must be exactly: "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights."
Format strictly as a JSON array of objects with keys "prompt" and "response".
"""

    print("Generating greeting examples...")
    greetings_res = query_teacher(greetings_prompt, format_json=True)
    greetings_data = parse_json_response(greetings_res)
    
    print("Generating name query examples...")
    name_res = query_teacher(name_prompt, format_json=True)
    name_data = parse_json_response(name_res)
    
    print("Generating description examples...")
    desc_res = query_teacher(desc_prompt, format_json=True)
    desc_data = parse_json_response(desc_res)
    
    # Fallback default dataset if Ollama fails or returns empty
    if not greetings_data:
        greetings_data = [{"prompt": "hello", "response": "Hello! I am doing well, thank you! How are you doing today? I hope you are having a wonderful day."}]
    if not name_data:
        name_data = [{"prompt": "what is your name?", "response": "My name is shreesha. I am your companion chatbot here to assist you with whatever you need."}]
    if not desc_data:
        desc_data = [{"prompt": "describe yourself", "response": "I am a tiny model running on Raspberry Pi Pico W. I am designed to process text efficiently using small-scale neural network weights."}]
        
    dataset = greetings_data + name_data + desc_data
    print(f"Dataset generated. Initial size: {len(dataset)} examples.")
    
    # Initialize student model with 2 KV heads and 256 seq_len
    model = TinyLlama(dim=64, hidden_dim=172, n_layers=5, n_heads=8, n_kv_heads=2, vocab_size=512, seq_len=256)
    
    # Test Prompts for Evaluation Loop
    test_cases = [
        {"prompt": "hello, how are you", "category": "Greeting"},
        {"prompt": "what is your name?", "category": "Name"},
        {"prompt": "who are you?", "category": "Name"},
        {"prompt": "describe yourself", "category": "Description"},
        {"prompt": "what are you?", "category": "Description"}
    ]
    
    max_loops = 5
    for loop in range(max_loops):
        print(f"\n==========================================")
        print(f"       ITERATIVE LOOP {loop+1}/{max_loops}")
        print(f"==========================================")
        
        # De-duplicate dataset prompts to prevent conflicting labels
        unique_dataset = {}
        for item in dataset:
            p_clean = item["prompt"].strip().lower()
            unique_dataset[p_clean] = item["response"].strip()
        dataset = [{"prompt": p, "response": r} for p, r in unique_dataset.items()]
        
        # Train student on current dataset
        model = train_student_model(model, tokenizer, dataset, epochs=150, batch_size=32, device=device)
        
        # Evaluate student outputs
        print("\n--- Evaluating Student Predictions ---")
        all_passed = True
        new_training_items = []
        
        for case in test_cases:
            prompt = case["prompt"]
            category = case["category"]
            
            response = generate_student_response(model, tokenizer, prompt, device=device)
            passed, corrective_data = evaluate_student(prompt, response, category)
            
            if not passed:
                all_passed = False
                new_training_items.extend(corrective_data)
                
        if all_passed:
            print("\n>>> SUCCESS: Student passed all teacher evaluations!")
            break
        else:
            print(f"\n>>> Student failed some tests. Expanding training dataset.")
            for item in new_training_items:
                dataset.append(item)
            print(f"Dataset expanded. New raw size: {len(dataset)} examples.")
            
    # Save the final trained model
    print("\n--- Exporting Final Model ---")
    export_model(model, "../chatbot.bin", "../../model_weights.h")
    print("Iterative self-training loop completed successfully!")

if __name__ == "__main__":
    main()
