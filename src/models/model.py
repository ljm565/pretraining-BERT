import torch
import torch.nn as nn
import torch.nn.functional as F



# word embedding layer
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output



# positional embedding layer
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_dim, device):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.device = device

        self.pos = torch.arange(0, self.max_len * 2)
        self.emb_layer = nn.Embedding(self.max_len * 2, self.hidden_dim)


    def forward(self, x):
        return self.emb_layer(self.pos.unsqueeze(0).to(self.device))[:, :x.size(1)]



# segment embedding layer
class SegmentEmbedding(nn.Module):
    def __init__(self, hidden_dim, pad_token_id):
        super(SegmentEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.emb_layer = nn.Embedding(3, self.hidden_dim, padding_idx=self.pad_token_id)


    def forward(self, x):
        output = self.emb_layer(x)
        return output
        


# mulithead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, bias, self_attn, causal):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.bias = bias
        self.self_attn = self_attn
        self.causal = causal
        self.head_dim = self.hidden_dim // self.num_head
        assert self.hidden_dim == self.num_head * self.head_dim

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.attn_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)


    def head_split(self, x):
        x = x.view(self.batch_size, -1, self.num_head, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x


    def scaled_dot_product(self, q, k, v, mask):
        attn_wts = torch.matmul(q, torch.transpose(k, 2, 3))/(self.head_dim ** 0.5)
        if not mask == None:
            attn_wts = attn_wts.masked_fill(mask==0, float('-inf'))
        attn_wts = F.softmax(attn_wts, dim=-1)
        attn_out = torch.matmul(attn_wts, v)
        return attn_wts, attn_out


    def reshaping(self, attn_out):
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(self.batch_size, -1, self.hidden_dim)
        return attn_out


    def forward(self, query, key, value, mask):
        if self.self_attn:
            assert (query == key).all() and (key==value).all()

        self.batch_size = query.size(0)
        q = self.head_split(self.q_proj(query))
        k = self.head_split(self.k_proj(key))
        v = self.head_split(self.v_proj(value))

        attn_wts, attn_out = self.scaled_dot_product(q, k, v, mask)
        attn_out = self.attn_proj(self.reshaping(attn_out))

        return attn_wts, attn_out



# postion wise feed forward
class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout, bias):
        super(PositionWiseFeedForward, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.bias = bias

        self.FFN1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=self.bias),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=self.bias),
        )
        self.init_weights()


    def init_weights(self):
        for _, param in self.named_parameters():
            if param.requires_grad:
                nn.init.normal_(param.data, mean=0, std=0.02)

    
    def forward(self, x):
        output = self.FFN1(x)
        output = self.FFN2(output)
        return output



# a single encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_head, bias, dropout, layernorm_eps):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.layernorm_eps = layernorm_eps
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)

        self.self_attention = MultiHeadAttention(self.hidden_dim, self.num_head, self.bias, self_attn=True, causal=False)
        self.positionWiseFeedForward = PositionWiseFeedForward(self.hidden_dim, self.ffn_dim, self.dropout, self.bias)


    def forward(self, x, mask):
        attn_wts, output = self.self_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        x = output
        output = self.positionWiseFeedForward(output)
        output = self.dropout_layer(output)
        output = self.layer_norm(x + output)

        return attn_wts, output



# all encoders
class Encoder(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(Encoder, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.ffn_dim = config.ffn_dim
        self.num_head = config.num_head
        self.max_len = config.max_len
        self.bias = bool(config.bias)
        self.dropout = config.dropout
        self.layernorm_eps = config.layernorm_eps
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.tok_emb = TokenEmbedding(self.vocab_size, self.hidden_dim, self.pad_token_id)
        self.pos_emb = PositionalEmbedding(self.max_len, self.hidden_dim, self.device)
        self.seg_emb = SegmentEmbedding(self.hidden_dim, self.pad_token_id)
        self.encoders = nn.ModuleList([EncoderLayer(self.hidden_dim, self.ffn_dim, self.num_head, self.bias, self.dropout, self.layernorm_eps) for _ in range(self.num_layers)])


    def forward(self, x, segment, mask=None):
        output = self.tok_emb(x) + self.pos_emb(x) + self.seg_emb(segment)
        output = self.dropout_layer(output)

        all_attn_wts = []
        for encoder in self.encoders:
            attn_wts, output = encoder(output, mask)
            all_attn_wts.append(attn_wts.detach().cpu())
        
        return all_attn_wts, output


# BERT
class BERT(nn.Module):
    def __init__(self, config, tokenizer, device):
        super(BERT, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        self.hidden_dim = self.config.hidden_dim

        self.encoder = Encoder(self.config, self.tokenizer, self.device)
        self.nsp_fc = nn.Linear(self.hidden_dim, 2)
        self.mlm_fc = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)


    def make_mask(self, input):
        enc_mask = torch.where(input==self.tokenizer.pad_token_id, 0, 1).unsqueeze(1).unsqueeze(2)
        return enc_mask


    def forward(self, x, segment):
        enc_mask = self.make_mask(x)
        all_attn_wts, x = self.encoder(x, segment, enc_mask)

        nsp_output = self.nsp_fc(x[:, 0])
        mlm_output = self.mlm_fc(x)

        return all_attn_wts, (nsp_output, mlm_output)