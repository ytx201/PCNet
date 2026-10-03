import torch
import torch.nn as nn


class SimpleCrossAttention(nn.Module):
    def __init__(self, 
                 dropout=0.5, 
                 input_len=96,
                 val_num=7,
                 output_len=96,
                 cycle_len=24,
                 template_num=7,
                 use_proj=False,
                 use_position_embedding=True,
                 use_full_template_as_value=False):
        super().__init__()
        self.cycle_len = cycle_len
        self.seq_len = input_len
        self.pred_len = output_len
        self.dropout = nn.Dropout(dropout)
        self.use_proj = use_proj
        self.use_position_embedding = use_position_embedding
        self.template_num = template_num
        self.use_full_template_as_value = use_full_template_as_value
        
        if use_position_embedding:
            self.positionEmbedding_Q = torch.nn.Parameter(torch.randn(val_num, input_len) * 0.02, requires_grad=True)
            self.positionEmbedding_K = torch.nn.Parameter(torch.randn(template_num, input_len) * 0.02, requires_grad=True)
        
        if use_proj:
            self.proj_Q = nn.Linear(input_len, input_len)
            self.proj_K = nn.Linear(input_len, input_len)
        
        self.cycleTemplate = torch.nn.Parameter(torch.randn(self.cycle_len, template_num) * 0.02, requires_grad=True)
        
    
    def forward(self, x, cycle_index): 
        """
        输入:
        - x: [batch, channels, seq_len]
        - cycle_index: [batch]
        输出: 注意力加权后的特征
        """
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len+self.pred_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        template = self.cycleTemplate[gather_index].permute(0, 2, 1)  # (b, c, s)
        
        Q = x
        K = template[:, :, :self.seq_len]
        
        if self.use_full_template_as_value:
            V = template
        else:
            V = template[:, :, self.seq_len:]
            
        if self.template_num == 1:
            return V.expand(-1, Q.shape[1], -1), None
            
        if self.use_position_embedding:
            Q = Q + self.positionEmbedding_Q.unsqueeze(0)
            K = K + self.positionEmbedding_K.unsqueeze(0)
            
        if self.use_proj:
            Q = self.proj_Q(Q)
            K = self.proj_K(K)
            
        attn_scores = torch.matmul(Q, K.transpose(1, 2))
        attn_scores = attn_scores / (self.seq_len ** 0.5)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.template_num = configs.template_num
        
        self.pcnet_mode = getattr(configs, 'pcnet_mode', 'DPCF')
        
        self.use_dynamic_template = self.pcnet_mode in ['DPCF', 'DPRF']
        self.use_residual_forecast = self.pcnet_mode in ['DPRF', 'FPRF']
        self.use_template_concat = self.pcnet_mode in ['DPCF', 'FPCF']
        
        if self.use_dynamic_template:
            use_full_template = self.pcnet_mode in ['DPRF', 'FPRF']
            self.simpleCrossAttention = SimpleCrossAttention(
                dropout=0.5,
                input_len=self.seq_len,
                val_num=self.enc_in,
                output_len=self.pred_len,
                cycle_len=self.cycle_len,
                template_num=self.template_num,
                use_position_embedding=True,
                use_proj=True,
                use_full_template_as_value=use_full_template)
        else:
            self.cycleTemplate = torch.nn.Parameter(torch.randn(self.cycle_len, self.enc_in) * 0.02, requires_grad=True)
        
        if self.use_template_concat:
            self.input_proj = nn.Linear(self.seq_len + self.pred_len, self.d_model)
        else:
            self.input_proj = nn.Linear(self.seq_len, self.d_model)
        
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def _get_template(self, x_input, cycle_index):
        if self.use_dynamic_template:
            return self.simpleCrossAttention(x_input, cycle_index)[0]
        else:
            if self.use_residual_forecast:
                gather_index = (cycle_index.view(-1, 1) + torch.arange(self.pred_len + self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            else:
                gather_index = (cycle_index.view(-1, 1) + self.seq_len + torch.arange(self.pred_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            return self.cycleTemplate[gather_index].permute(0, 2, 1)

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)
        
        template = self._get_template(x_input, cycle_index)

        if self.use_template_concat:
            model_input = self.input_proj(torch.cat([x_input, template], dim=-1))
        else:
            model_input = self.input_proj(x_input - template[:, :, :self.seq_len])

        hidden = self.model(model_input)

        if self.use_residual_forecast:
            output = (self.output_proj(hidden + model_input) + template[:, :, self.seq_len:]).permute(0, 2, 1)
        else:
            output = self.output_proj(hidden + model_input).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output