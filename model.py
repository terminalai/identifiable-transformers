"""
An Identifiable Transformer (Usage-Ready Copy)

From "More Identifiable yet Equally Performant Transformers for Text Classification" (https://arxiv.org/abs/2106.01269)

Credits to Rishabh Bhardwaj for the original code.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MultiHeadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, concat_heads, dropout, bias=True, kdim=None, vdim=None):
        """
        embed_dim: dimension of embedding vector
        num_heads: no of heads in MHA
        concat_heads: head output: concatenate or add
        dropout: well, dropout
        bias: bias
        kdim: dimension of key vector
        vdim: dimension of value vector, no longer = embed_dim // num_heads, also head dimension
        
        Usage:
        >>> MultiHeadAttention(embed_dim=d_model, num_heads=nhead, concat_heads=concat_heads, dropout=dropout, kdim=kdim, vdim=vdim)
        """
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, kdim=kdim, vdim=vdim)
        self.embed_dim = embed_dim
        self.kdim = kdim
        self.vdim = vdim
        self.head_dim = vdim
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat_head_output = concat_heads

        # projection matrices to obtain query, key and value vectors
        self.q_proj_weight = nn.Parameter(Tensor(embed_dim, self.kdim*num_heads))
        self.k_proj_weight = nn.Parameter(Tensor(embed_dim, self.kdim*num_heads))
        self.v_proj_weight = nn.Parameter(Tensor(embed_dim, self.vdim*num_heads))

        # initialize bias parameters for projection matrices
        self.in_proj_bias = nn.Parameter(torch.empty(kdim*num_heads*2 + vdim*num_heads))

        # weights for output transformation
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def forward(self, query, key, value, key_padding_mask, need_weights=False):
        # query is [number of sentence tokens, batch, embedding dim]
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        #number of tokens and batch size should be same in key and value tensor
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        #scaling factor
        scaling = float(self.head_dim) ** -0.5

        #qh = qdim * num_heads
        _, qh = self.q_proj_weight.size()

        #qh = kdim * num_heads
        _, kh = self.k_proj_weight.size()

        #vh = vdim * num_heads
        _, vh = self.v_proj_weight.size()

        assert self.in_proj_bias.size()[0] == qh + kh + vh

        '''
        Input transformation
        : linear(x,w,b) = x*w^T + b
        '''

        #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, qh]
        q = linear(query, self.q_proj_weight.transpose(0, 1), self.in_proj_bias[0: qh])

        #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, kh]
        k = linear(key, self.k_proj_weight.transpose(0, 1), self.in_proj_bias[qh : qh + kh])

        #[no of tokens, batch size, embedding dim] -> [no of tokens, batch size, vh]
        v = linear(value, self.v_proj_weight.transpose(0, 1), self.in_proj_bias[qh + kh : ])
    
        #scaling query vectors
        q = q * scaling

        #[no of tokens, batch size * num_heads, qdim] -> [batch size * num_heads, no of tokens, qdim]
        q = q.view(tgt_len, bsz * self.num_heads, qh // self.num_heads).transpose(0, 1)

        #[no of tokens, batch size * num_heads, kdim] -> [batch size * num_heads, no of tokens, kdim]
        k = k.view(tgt_len, bsz * self.num_heads, kh // self.num_heads).transpose(0, 1)

        #[no of tokens, batch size * num_heads, vdim] -> [batch size * num_heads, no of tokens, vdim]
        v = v.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        #[batch size * num_heads, no of tokens, qdim] x [batch size * num_heads, no of tokens, kdim].T -> [batch size * num_head, no of tokens, no of tokens]
        attn_logits = torch.bmm(q, k.transpose(1, 2))

        #mask unwanted attentions from pad tokens
        if key_padding_mask != None:
            mask = key_padding_mask
            mask = mask.repeat(1,tgt_len)
            mask = mask.view(-1,tgt_len,tgt_len)
            mask = (mask*(mask.transpose(1,2))) == 1
            mask = mask.repeat(self.num_heads,1,1)
            attn_logits = attn_logits.masked_fill_(mask, -1e10)

        assert list(attn_logits.size()) == [bsz * self.num_heads, tgt_len, tgt_len]

        #softmax attention logits
        attn_output_weights = softmax(attn_logits, dim=-1)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, training=self.training)

        #[batch size * num_heads, no of tokens, no of tokens] * [batch size * num_heads, no of tokens, vdim].T  -> [batch size * num_heads, no of tokens, vdim]                                                   
        head_output = torch.bmm(attn_output_weights, v)

        assert list(head_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        if self.concat_head_output == True:
            #concat head outputs
            assert self.head_dim == embed_dim // self.num_heads
            head_output = head_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)

        elif self.concat_head_output == False:
            #add head outputs
            assert self.head_dim == embed_dim
            head_output = head_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim)
            head_output = head_output.view(tgt_len, bsz, self.head_dim, self.num_heads)
            head_output = reduce(torch.add,[head_output[:,:,:,i] for i in range(head_output.size(3))])
        
        else:
            raise Exception("Unexpected type of operation over head outputs!")

        #output transformation
        head_output = linear(head_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            return head_output, attn_output_weights
        else:
            return head_output, None
    

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, concat_heads, kdim, vdim, dim_feedforward, dropout):
        super().__init__(d_model, nhead)
        self.self_attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead, concat_heads=concat_heads, dropout=dropout, kdim=kdim, vdim=vdim)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # requirements of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=True)
        self.activation = F.relu    #ReLU activation
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=True)
        self.norm2 = nn.LayerNorm(d_model)
        #
    def forward(self, src, mask, return_attn_weights=False):
        src1, attn_weights = self.self_attn(query=src, key=src, value=src, key_padding_mask=mask, need_weights=return_attn_weights)
        src = src + self.dropout(src1)   #Currently all the dropouts happen with the same probability
        src = self.norm1(src)
        src1 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src1)
        src = self.norm2(src)
        return src, attn_weights