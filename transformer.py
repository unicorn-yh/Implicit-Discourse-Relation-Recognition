import torch.nn as nn
import torch as torch
import math
import torch.nn.functional as F
import copy
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

#-----------------------------------------------------begin-----------------------------------------------------#
# 类别数
num_classes = 15
#------------------------------------------------------end------------------------------------------------------#

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, config, vocab_size, ninp=300, ntoken=150, nhid=1200, nhead=10, nlayers=6, dropout=0.2, embedding_weight=None, name=None):
        super(Model, self).__init__()
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计词嵌入层
        self.attention = True
        self.name = "TRANSFORMER"
        print(self.name.center(105,'='))
        #self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=ninp, padding_idx=vocab_size-1)
        #self.embed.weight.data.copy_(torch.from_numpy(embedding_weight))
        self.config = config
        self.label_embedding = nn.Parameter(torch.randn(config.label_num, config.label_embedding_size, dtype=torch.float32))
        nn.init.xavier_normal_(self.label_embedding.data)
        #------------------------------------------------------end------------------------------------------------------#
        
        self.pos_encoder = PositionalEncoding(d_model=ninp, max_len=ntoken)

        if self.attention == False:
            self.encode_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 请自行设计对 transformer 隐藏层数据的处理和选择方法
    # 请自行设计分类器
        
        elif self.attention == True:  # 使用 Multi-head Attention

            # Multi-Head Attention
            self.nlayers = nlayers
            self.nhead = nhead
            assert ninp % nhead == 0 
            self.dim_head = ninp // self.nhead
            self.fc_Q = nn.Linear(ninp, nhead * self.dim_head)
            self.fc_K = nn.Linear(ninp, nhead * self.dim_head)
            self.fc_V = nn.Linear(ninp, nhead * self.dim_head)
            self.fc = nn.Linear(nhead * self.dim_head, ninp)
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(ninp)

            # Position Wise Feed Forward
            self.fc1 = nn.Linear(ninp, nhid)
            self.fc2 = nn.Linear(nhid, ninp)

        self.fc_out = nn.Linear(ninp*config.max_sent_len, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def Scaled_Dot_Product_Attention(self, Q, K, V, d_k=None):
        d_k = Q.size(-1) ** -0.5  # 缩放因子
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if d_k:
            attention = attention * d_k
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

    def Position_Wise_Feed_Forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

    #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
              
        if self.attention == False:
            x = self.embed(x)                # [batch_size, sentence_length, embedding_size]   64 30 300
            x = x.permute(1, 0, 2)           # [sentence_length, batch_size, embedding_size]   30 64 300 
            x = self.pos_encoder(x)          # [sentence_length, batch_size, embedding_size]   30 64 300
            x = self.transformer_encoder(x)  # [sentence_length, batch_size, embedding_size]   30 64 300                 
            x = x.permute(1, 0, 2)           # [batch_size, sentence_length, embedding_size]   64 30 300
    
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类

        elif self.attention == True:   
            x = self.embed(x)
            x = self.pos_encoder(x)

            for i in range(12):  
                batch_size = x.size(0)
                Q = self.fc_Q(x)
                K = self.fc_K(x)
                V = self.fc_V(x)
                Q = Q.view(batch_size * self.nhead, -1, self.dim_head)
                K = K.view(batch_size * self.nhead, -1, self.dim_head)
                V = V.view(batch_size * self.nhead, -1, self.dim_head)

                context = self.Scaled_Dot_Product_Attention(Q, K, V)

                context = context.view(batch_size, -1, self.dim_head * self.nhead)
                out = self.fc(context)
                out = self.dropout(out)
                out = out + x  # 残差连接
                out = self.layer_norm(out)

                x = self.Position_Wise_Feed_Forward(out)

        x = x.reshape(x.size(0), -1)
        x = self.fc_out(x)
        x = self.softmax(x)
        #------------------------------------------------------end------------------------------------------------------#

        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ninp=300, ntoken=150,  nhid=150, nlayers=4, dropout=0.2, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 自行设计词嵌入层
        self.name = "Bi-LSTM"
        print(self.name.center(105,'='))
        self.embed = nn.Embedding(vocab_size, ninp, padding_idx=vocab_size - 1)
        self.embed.weight.data.copy_(torch.from_numpy(embedding_weight))
        #------------------------------------------------------end------------------------------------------------------#

        self.lstm = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        # 请自行设计分类器

        self.num_layers = nlayers
        self.num_directions = 2
        self.hidden_size = nhid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(nhid, num_classes)
        self.act_func = nn.Softmax(dim=1)
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)        # [batch_size, sentence_length, embedding_size]      64 30 100
        #x = self.lstm(x)[0]
        #x = self.dropout(x)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        
        batch_size = x.size(0)  #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(0)获取当前数据实际的batch

        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)

        #out[max_sent_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out    #[batch_size, max_sent_len, hidden_size] 64 30 300

        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)   #[batch_size, num_layers * num_directions, hidden_size]
        h_n = torch.sum(h_n, dim=1)  #[batch_size, 1, hidden_size]
        h_n = h_n.squeeze(dim=1)     #[batch_size, hidden_size]

        attention_w = self.attention_weights_layer(h_n)  #[batch_size, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch_size, 1, hidden_size]

        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch_size, 1, max_sent_len]
        softmax_w = F.softmax(attention_context, dim=-1)                 #[batch_size, 1, max_sent_len]  权重归一化

        x = torch.bmm(softmax_w, out)  #[batch_size, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch_size, hidden_size]
        x = self.fc(x)
        x = self.act_func(x)
        #------------------------------------------------------end------------------------------------------------------#
        
        #x = self.classifier(x)
        return x

