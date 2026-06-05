#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/vreg.h"
#include "hardware/clocks.h"
#include "hardware/i2c.h"
#include "ssd1306_font.h"

// ----------------------------------------------------------------------------
// Align and include the model weights and tokenizer data arrays from flash
// ----------------------------------------------------------------------------
#define model_data __attribute__((aligned(4))) model_data
#include "../../model_weights.h"
#undef model_data

#define tokenizer_data __attribute__((aligned(4))) tokenizer_data
#include "../../tokenizer_data.h"
#undef tokenizer_data

// Helper functions to safely read potentially unaligned values from flash (tokenizer vocab structure)
float read_unaligned_float(const uint8_t* ptr) {
    float val;
    memcpy(&val, ptr, sizeof(float));
    return val;
}

int read_unaligned_int(const uint8_t* ptr) {
    int val;
    memcpy(&val, ptr, sizeof(int));
    return val;
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        printf("malloc failed! Out of memory on RP2040 SRAM!\n");
        while(true) { sleep_ms(1000); }
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip RoPE freq_cis_real
    ptr += p->seq_len * head_size / 2; // skip RoPE freq_cis_imag
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void load_model_from_flash(Config* config, TransformerWeights* weights) {
    // Read config from start of model_data
    memcpy(config, model_data, sizeof(Config));
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    
    // Save original seq_len to index the weights correctly
    int original_seq_len = config->seq_len;
    
    // Memory map the Transformer weights in flash
    float* weights_ptr = (float*)(model_data + sizeof(Config));
    
    // Verify alignment
    if (((uintptr_t)weights_ptr % 4) != 0) {
        printf("WARNING: weights_ptr is not aligned to 4-byte boundary!\n");
    }
    
    memory_map_weights(weights, config, weights_ptr, shared_weights);
    
    // Override seq_len to fit in Pico W SRAM constraints (max 256 context length)
    if (config->seq_len > 256) {
        config->seq_len = 256;
    }
}

void build_transformer(Transformer *t) {
    load_model_from_flash(&t->config, &t->weights);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    for(int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        int loff = l * p->seq_len * kv_dim; 
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; 
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; 
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Byte Pair Encoding (BPE) Tokenizer

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; 
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void load_tokenizer_from_flash(Tokenizer* t, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; 
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    const uint8_t* ptr = tokenizer_data;
    t->max_token_length = read_unaligned_int(ptr);
    ptr += sizeof(int);

    for (int i = 0; i < vocab_size; i++) {
        t->vocab_scores[i] = read_unaligned_float(ptr);
        ptr += sizeof(float);

        int len = read_unaligned_int(ptr);
        ptr += sizeof(int);

        t->vocab[i] = (char*)malloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0'; 
        ptr += len;
    }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

// ----------------------------------------------------------------------------
// SSD1306 OLED Display Driver via I2C
// ----------------------------------------------------------------------------
#define I2C_PORT i2c0
#define I2C_SDA_PIN 0
#define I2C_SCL_PIN 1
#define SSD1306_I2C_ADDR 0x3C
#define OLED_WIDTH 128
#define OLED_HEIGHT 32
#define OLED_PAGES 4
#define CHAR_WIDTH 6
#define CHAR_HEIGHT 8
#define MAX_COLS (OLED_WIDTH / CHAR_WIDTH) // 21
#define MAX_ROWS OLED_PAGES                // 4

// 512 bytes frame buffer + 1 control byte at the beginning (0x40)
static uint8_t oled_fb[513] = {0x40};

// Terminal emulator state
static char terminal_buffer[MAX_ROWS][MAX_COLS + 1];
static int cursor_col = 0;
static int cursor_row = 0;

void ssd1306_send_cmd(uint8_t cmd) {
    uint8_t buf[2] = {0x00, cmd};
    i2c_write_blocking(I2C_PORT, SSD1306_I2C_ADDR, buf, 2, false);
}

void ssd1306_init() {
    // I2C Init at 400kHz
    i2c_init(I2C_PORT, 400 * 1000);
    gpio_set_function(I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(I2C_SDA_PIN);
    gpio_pull_up(I2C_SCL_PIN);
    
    sleep_ms(100); // Wait for OLED to power up

    // SSD1306 initialization sequence for 128x32 OLED
    uint8_t init_cmds[] = {
        0xAE, // Display OFF
        0xD5, 0x80, // Set Display Clock Divide Ratio/Oscillator Frequency
        0xA8, 0x1F, // Set Multiplex Ratio (32 lines)
        0xD3, 0x00, // Set Display Offset
        0x40,       // Set Display Start Line (0)
        0x8D, 0x14, // Charge Pump (Enable)
        0x20, 0x00, // Set Memory Addressing Mode (Horizontal)
        0xA1,       // Set Segment Re-map (COL 127 is SEG0)
        0xC8,       // Set COM Output Scan Direction (vertically flipped)
        0xDA, 0x02, // Set COM Pins Hardware Configuration
        0x81, 0xCF, // Set Contrast
        0xD9, 0xF1, // Set Pre-charge Period
        0xDB, 0x40, // Set VCOMH Deselect Level
        0xA4,       // Entire Display ON (follow RAM)
        0xA6,       // Normal Display
        0x2E,       // Deactivate Scroll
        0xAF        // Display ON
    };
    for (size_t i = 0; i < sizeof(init_cmds); i++) {
        ssd1306_send_cmd(init_cmds[i]);
    }
}

void oled_clear() {
    memset(oled_fb + 1, 0, sizeof(oled_fb) - 1);
}

void ssd1306_update() {
    ssd1306_send_cmd(0x21); // Column Address range
    ssd1306_send_cmd(0x00);
    ssd1306_send_cmd(127);
    ssd1306_send_cmd(0x22); // Page Address range
    ssd1306_send_cmd(0x00);
    ssd1306_send_cmd(OLED_PAGES - 1);
    i2c_write_blocking(I2C_PORT, SSD1306_I2C_ADDR, oled_fb, sizeof(oled_fb), false);
}

void oled_draw_char(int col, int row, char c) {
    if (col < 0 || col >= MAX_COLS || row < 0 || row >= MAX_ROWS) return;
    int x = col * CHAR_WIDTH;
    int page = row;
    uint8_t u_c = (uint8_t)c;
    const uint8_t* char_data = &font[u_c * 5];
    int fb_offset = 1 + page * 128 + x;
    for (int i = 0; i < 5; i++) {
        oled_fb[fb_offset + i] = char_data[i];
    }
    oled_fb[fb_offset + 5] = 0x00; // Spacing pixel column
}

void oled_clear_terminal() {
    for (int r = 0; r < MAX_ROWS; r++) {
        memset(terminal_buffer[r], ' ', MAX_COLS);
        terminal_buffer[r][MAX_COLS] = '\0';
    }
    cursor_col = 0;
    cursor_row = 0;
    oled_clear();
    ssd1306_update();
}

void oled_write_char(char c) {
    if (c == '\n') {
        cursor_col = 0;
        cursor_row++;
        if (cursor_row >= MAX_ROWS) {
            // Scroll all lines up
            for (int r = 0; r < MAX_ROWS - 1; r++) {
                memcpy(terminal_buffer[r], terminal_buffer[r + 1], MAX_COLS + 1);
            }
            memset(terminal_buffer[MAX_ROWS - 1], ' ', MAX_COLS);
            terminal_buffer[MAX_ROWS - 1][MAX_COLS] = '\0';
            cursor_row = MAX_ROWS - 1;
        }
    } else if (c == '\r') {
        cursor_col = 0;
    } else if (c >= 32 && c <= 126) {
        if (cursor_col >= MAX_COLS) {
            oled_write_char('\n');
        }
        terminal_buffer[cursor_row][cursor_col] = c;
        cursor_col++;
    }
}

void oled_print_str(const char* str) {
    if (str == NULL) return;
    while (*str) {
        oled_write_char(*str++);
    }
    // Refresh OLED
    oled_clear();
    for (int r = 0; r < MAX_ROWS; r++) {
        for (int c = 0; c < MAX_COLS; c++) {
            oled_draw_char(c, r, terminal_buffer[r][c]);
        }
    }
    ssd1306_update();
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; 
        }
    }
    printf("%s", piece);
    oled_print_str(piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok;
    tok.str = str;
    tok.id = 0;
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { printf("cannot encode NULL text\n"); return; }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c; 
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; 
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; 
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; 
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex; 
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; 
}

int compare_probs(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probs);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; 
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; 
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; 
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { 
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Utilities: time tracking

long time_in_ms() {
    return to_ms_since_boot(get_absolute_time());
}

// ----------------------------------------------------------------------------
// Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = (char*)"";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); 
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("something is wrong, expected at least 1 prompt token\n");
        free(prompt_tokens);
        return;
    }

    long start = 0;  
    int next;        
    int token = prompt_tokens[0]; 
    int pos = 0;     
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); 
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    oled_print_str("\n");

    if (pos > 1) {
        long end = time_in_ms();
        printf("\nAchieved speed: %f tok/s\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin_line(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    fflush(stdout);
    
    size_t index = 0;
    while (index < bufsize - 1) {
        int c = getchar();
        if (c == EOF || c == '\n' || c == '\r') {
            if (c == '\r') {
                int next_c = getchar_timeout_us(100);
                if (next_c != '\n' && next_c != EOF && next_c != (int)PICO_ERROR_TIMEOUT) {
                    // ignore
                }
            }
            break;
        }
        
        putchar(c);
        fflush(stdout);
        
        buffer[index++] = (char)c;
    }
    buffer[index] = '\0';
    printf("\n");
    fflush(stdout);
}

// ----------------------------------------------------------------------------
// Chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[256] = {0};
    char user_prompt[256] = {0};
    char rendered_prompt[512] = {0};
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(512 * sizeof(int));
    int user_idx = 0;

    int8_t user_turn = 1; 
    int next = 0;        
    int token = 0;       
    int pos = 0;     
    
    printf("Starting Chat Session. Press Enter on an empty line to exit.\n");
    
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin_line("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin_line("User: ", user_prompt, sizeof(user_prompt));
            }
            
            if (strlen(user_prompt) == 0) {
                break;
            }

            oled_print_str("\nUser: ");
            oled_print_str(user_prompt);
            oled_print_str("\n");

            if (pos == 0 && system_prompt[0] != '\0') {
                sprintf(rendered_prompt, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt, "[INST] %s [/INST]", user_prompt);
            }
            
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; 
            user_turn = 0;
            printf("Assistant: ");
            fflush(stdout);
            oled_print_str("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        
        if (token == 2) { user_turn = 1; }

        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); 
            fflush(stdout);
        }
        if (next == 2) { 
            printf("\n"); 
            fflush(stdout);
            oled_print_str("\n");
        }
    }
    printf("\nChat ended.\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Main CLI Interface for Pico W Serial Console

int main() {
    // 1. Overclock to 250 MHz (safe and doubles software-float inference speed!)
    vreg_set_voltage(VREG_VOLTAGE_1_15);
    set_sys_clock_khz(250000, true);

    // 2. Initialize stdio (redirects stdout/stdin to USB serial & UART)
    stdio_init_all();
    
    // Give the user 2 seconds to open their serial monitor
    sleep_ms(2000);

    // 2b. Initialize SSD1306 OLED via I2C
    ssd1306_init();
    oled_clear_terminal();
    oled_print_str("=====================\n");
    oled_print_str("     PICO LLAMA2     \n");
    oled_print_str("=====================\n");
    oled_print_str("System clock: 250MHz \n");
    oled_print_str("Initializing...\n");

    printf("\n");
    printf("===============================================\n");
    printf("     __  __      ___   __  __      ___   \n");
    printf("    |  \\/  |    / _ \\ |  \\/  |    / _ \\  \n");
    printf("    | \\  / |   | | | || \\  / |   | | | | \n");
    printf("    | |\\/| |   | | | || |\\/| |   | | | | \n");
    printf("    | |  | | _ | |_| || |  | | _ | |_| | \n");
    printf("    |_|  |_|(_)[_____]|_|  |_|(_)[_____] \n");
    printf("===============================================\n");
    printf(" Llama-2 Transformer Inference on Raspberry Pi Pico W\n");
    printf(" Running at 250 MHz Clock Speed\n");
    printf("===============================================\n");

    // Build the Transformer using embedded model weights
    printf("Loading model weights from Flash...\n");
    Transformer transformer;
    build_transformer(&transformer);
    printf("Model configuration:\n");
    printf(" - dim: %d\n", transformer.config.dim);
    printf(" - hidden_dim: %d\n", transformer.config.hidden_dim);
    printf(" - layers: %d\n", transformer.config.n_layers);
    printf(" - heads: %d\n", transformer.config.n_heads);
    printf(" - KV heads: %d\n", transformer.config.n_kv_heads);
    printf(" - vocab_size: %d\n", transformer.config.vocab_size);
    printf(" - context_len (overridden max): %d\n", transformer.config.seq_len);

    // Build the Tokenizer
    printf("Loading tokenizer vocabulary from Flash...\n");
    Tokenizer tokenizer;
    load_tokenizer_from_flash(&tokenizer, transformer.config.vocab_size);
    printf("Tokenizer loaded successfully (max_token_len=%u).\n\n", tokenizer.max_token_length);
    oled_print_str("Model & vocab loaded!\nReady for prompt.\n");

    // Default inference parameters
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = transformer.config.seq_len;
    unsigned long long rng_seed = time_in_ms();

    // Build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    char command[16];
    char prompt_buffer[128];

    while (true) {
        printf("\n--- Pico Llama CLI ---\n");
        printf("1. Run Generate Mode (Storytelling)\n");
        printf("2. Run Chat Mode (Interactive dialog)\n");
        printf("3. Configure hyperparameters (temp=%0.2f, top-p=%0.2f, steps=%d)\n", temperature, topp, steps);
        printf("Choose option (1-3): ");
        fflush(stdout);

        read_stdin_line("", command, sizeof(command));
        int choice = atoi(command);

        if (choice == 1) {
            printf("\nEnter prompt (press enter for default): ");
            read_stdin_line("", prompt_buffer, sizeof(prompt_buffer));
            
            char* prompt = prompt_buffer;
            if (strlen(prompt) == 0) {
                prompt = (char*)"Once upon a time";
            }
            oled_clear_terminal();
            oled_print_str("Prompt: ");
            oled_print_str(prompt);
            oled_print_str("\n\n");
            printf("\nGenerating story (prompt: \"%s\"):\n", prompt);
            printf("----------------------------------------\n");
            generate(&transformer, &tokenizer, &sampler, prompt, steps);
            printf("----------------------------------------\n");
            
        } else if (choice == 2) {
            oled_clear_terminal();
            oled_print_str("=== PICO CHAT ===\n");
            printf("\nStarting chat dialog...\n");
            printf("----------------------------------------\n");
            chat(&transformer, &tokenizer, &sampler, NULL, NULL, steps);
            printf("----------------------------------------\n");
            
        } else if (choice == 3) {
            char val_buf[16];
            printf("\nEnter temperature (0.0 for greedy, 1.0 default): ");
            read_stdin_line("", val_buf, sizeof(val_buf));
            temperature = atof(val_buf);
            
            printf("Enter top-p value (0.0-1.0, 0.9 default): ");
            read_stdin_line("", val_buf, sizeof(val_buf));
            topp = atof(val_buf);
            
            printf("Enter number of steps (1-%d, %d default): ", transformer.config.seq_len, transformer.config.seq_len);
            read_stdin_line("", val_buf, sizeof(val_buf));
            int input_steps = atoi(val_buf);
            if (input_steps > 0 && input_steps <= transformer.config.seq_len) {
                steps = input_steps;
            } else {
                steps = transformer.config.seq_len;
            }
            
            // Rebuild sampler with new params
            free_sampler(&sampler);
            build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
            
            printf("\nHyperparameters updated!\n");
        } else {
            printf("\nInvalid option. Please choose 1, 2, or 3.\n");
        }
    }

    free(sampler.probindex);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
