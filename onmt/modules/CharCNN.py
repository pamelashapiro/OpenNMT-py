"""
Implementation of "CharCNN"
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt.modules
from onmt.Models import EncoderBase
import onmt
from onmt.Utils import aeq

class Highway(nn.Module):
    def __init__(self, input_size, num_layers):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.output_lin = nn.ModuleList()
        self.transform_lin = nn.ModuleList()
        self.proj_lin = nn.ModuleList()
        for i in range(num_layers):
            self.output_lin.append(nn.Linear(input_size, input_size))
            self.transform_lin.append(nn.Linear(input_size, input_size))
            self.proj_lin.append(nn.Linear(input_size, input_size))

    def forward(self, x):
        for i in range(self.num_layers):
            output = F.relu(self.output_lin[i](x))
            transform_gate = F.sigmoid(self.transform_lin[i](x))
            proj = self.proj_lin[i](x)

            x = transform_gate * output + (1 - transform_gate) * proj

        return x


class CharCNNEncoder(EncoderBase):
    """ An encoder built from character embeddings, convolutional layers with max pooling,
    highway layers to form words, and then a recurrent neural network over words.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, kernel_width, num_kernels, num_highway_layers, 
                 dropout=0.0, embeddings=None):
        super(CharCNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        print("kernel_width: ", kernel_width)

        self.conv = nn.Conv1d(embeddings.embedding_size, num_kernels, kernel_width)
        self.highway = Highway(num_kernels, num_highway_layers)

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=num_kernels,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        #print("emb", emb)
        s_len, batch, emb_dim = emb.size()

        # Convert to batch of words
        num_words = max(lengths)
        word_len = int(s_len / num_words)
        cnn_emb = emb.view(num_words, word_len, batch, emb_dim)
        cnn_emb = cnn_emb.transpose(0, 1).contiguous()
        cnn_emb = cnn_emb.view(word_len, num_words * batch, emb_dim)
        cnn_emb = cnn_emb.transpose(0,2).transpose(0,1).contiguous()
        cnn_emb = cnn_emb.view(num_words * batch, emb_dim, word_len)
        cnn_output = self.conv(cnn_emb)
        pool_output = torch.max(F.tanh(cnn_output), 2)[0]
        highway_output = self.highway(pool_output)
        word_emb = highway_output.view(num_words, batch, -1)

        packed_emb = word_emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(word_emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs