import argparse
from os.path import isfile, join
from os import listdir
import pandas as pd 
import torch 
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class single_seq_LSTM(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 hidden, 
                 embed_dim,
                 nlstm,
                 dropout):
        
        super(single_seq_LSTM, self).__init__()
        
        embed_dims = {'Text' : embed_dim,'G':embed_dim,'S':embed_dim,'P':embed_dim,'NA':embed_dim}

        self.embeddings = nn.ModuleDict({k : nn.Embedding(vocab_size, embed_dims[k]) 
                                         for k in embed_dims.keys()})
        self.dimension = hidden

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden,
                            num_layers=nlstm,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(p = dropout)
        self.fc = nn.Linear(2*hidden, 6)

    def forward(self, text, G_tens, S_tens, P_tens,NA_tens,Lens):

        text_emb = self.embeddings['Text'](text)
        G_emb = self.embeddings['G'](G_tens)
        S_emb = self.embeddings['S'](S_tens)
        P_emb = self.embeddings['P'](P_tens)
        NA_emb = self.embeddings['NA'](NA_tens)

        packed_input = pack_padded_sequence(torch.cat([text_emb,G_emb,S_emb,P_emb,NA_emb],dim=1),
                                            Lens, batch_first=True,
                                            enforce_sorted=False)

        packed_output, _ = self.lstm(packed_input)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), Lens - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)
        return text_out


class double_seq_LSTM(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 hidden_t, 
                 hidden_f, 
                 embed_dim_t,
                 embed_dim_f,
                 nlstm_t,
                 nlstm_f,
                 dropout,
                 ):
        
        super(double_seq_LSTM, self).__init__()

        embed_dims = {'Text' : embed_dim_t,'G':embed_dim_f,'S':embed_dim_f,'P':embed_dim_f,'NA':embed_dim_f}
        self.embeddings = nn.ModuleDict({k : nn.Embedding(vocab_size, embed_dims[k]) 
                                         for k in embed_dims.keys()})  
        self.dimension_t = hidden_t
        self.dimension_f = hidden_f

        self.lstm_t = nn.LSTM(input_size=embed_dim_t,
                            hidden_size=hidden_t,
                            num_layers=nlstm_t,
                            batch_first=True,
                            bidirectional=True)

        self.lstm_f = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(p = dropout)
        self.fc = nn.Linear(2*hidden_t + 2*hidden_f, 6)

    def forward(self, text, G_tens, S_tens, P_tens,NA_tens,Lens_f,Lens_t):

        text_emb = self.embeddings['Text'](text)
        G_emb = self.embeddings['G'](G_tens)
        S_emb = self.embeddings['S'](S_tens)
        P_emb = self.embeddings['P'](P_tens)
        NA_emb = self.embeddings['NA'](NA_tens)

        packed_input_f = pack_padded_sequence(torch.cat([G_emb,S_emb,P_emb,NA_emb],dim=1),
                                            Lens_f, batch_first=True,
                                            enforce_sorted=False)
        packed_output_f, _ = self.lstm_f(packed_input_f)
        output_f, _ = pad_packed_sequence(packed_output_f, batch_first=True)
        out_forward_f = output_f[range(len(output_f)), Lens_f - 1, :self.dimension_f]
        out_reverse_f = output_f[:, 0, self.dimension_f:]
        out_reduced_f = torch.cat((out_forward_f, out_reverse_f), 1)

        packed_input_t = pack_padded_sequence(text_emb,
                                            Lens_t, batch_first=True,
                                            enforce_sorted=False)
        packed_output_t, _ = self.lstm_t(packed_input_t)
        output_t, _ = pad_packed_sequence(packed_output_t, batch_first=True)
        out_forward_t = output_t[range(len(output_t)), Lens_t - 1, :self.dimension_t]
        out_reverse_t = output_t[:, 0, self.dimension_t:]
        out_reduced_t = torch.cat((out_forward_t, out_reverse_t), 1)

        out_reduced = torch.cat((out_reduced_t, out_reduced_f), 1)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out

class Multi_LSTM(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 hidden_t, 
                 hidden_f, 
                 embed_dim_t,
                 embed_dim_f,
                 nlstm_t,
                 nlstm_f,
                 dropout,
                 ):
        
        super(Multi_LSTM, self).__init__()

        embed_dims = {'Text' : embed_dim_t,'G':embed_dim_f,'S':embed_dim_f,'P':embed_dim_f,'NA':embed_dim_f}
        self.embeddings = nn.ModuleDict({k : nn.Embedding(vocab_size, embed_dims[k]) 
                                         for k in embed_dims.keys()})  
        self.dimension_t = hidden_t
        self.dimension_f = hidden_f

        self.lstm_t = nn.LSTM(input_size=embed_dim_t,
                            hidden_size=hidden_t,
                            num_layers=nlstm_t,
                            batch_first=True,
                            bidirectional=True)

        self.lstm_s = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=True)

        self.lstm_p = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=True)
        self.lstm_g = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=True)

        self.drop = nn.Dropout(p = dropout)
        self.fc = nn.Linear(2*hidden_t + 6*hidden_f, 6)

    def forward(self, text, G_tens, S_tens, P_tens,Lens_g,Lens_s,Lens_p,Lens_t):
        text_emb = self.embeddings['Text'](text)
        G_emb = self.embeddings['G'](G_tens)
        S_emb = self.embeddings['S'](S_tens)
        P_emb = self.embeddings['P'](P_tens)

        packed_input_s = pack_padded_sequence(S_emb,
                                            Lens_s, batch_first=True,
                                            enforce_sorted=False)
        packed_output_s, _ = self.lstm_s(packed_input_s)
        output_s, _ = pad_packed_sequence(packed_output_s, batch_first=True)
        out_forward_s = output_s[range(len(output_s)), Lens_s - 1, :self.dimension_f]
        out_reverse_s = output_s[:, 0, self.dimension_f:]
        out_reduced_s = torch.cat((out_forward_s, out_reverse_s), 1)

        packed_input_p = pack_padded_sequence(P_emb,
                                            Lens_p, batch_first=True,
                                            enforce_sorted=False)
        packed_output_p, _ = self.lstm_p(packed_input_p)
        output_p, _ = pad_packed_sequence(packed_output_p, batch_first=True)
        out_forward_p = output_p[range(len(output_p)), Lens_p - 1, :self.dimension_f]
        out_reverse_p = output_p[:, 0, self.dimension_f:]
        out_reduced_p = torch.cat((out_forward_p, out_reverse_p), 1)

        packed_input_g = pack_padded_sequence(G_emb,
                                            Lens_g, batch_first=True,
                                            enforce_sorted=False)
        packed_output_g, _ = self.lstm_g(packed_input_g)
        output_g, _ = pad_packed_sequence(packed_output_g, batch_first=True)
        out_forward_g = output_g[range(len(output_g)), Lens_g - 1, :self.dimension_f]
        out_reverse_g = output_g[:, 0, self.dimension_f:]
        out_reduced_g = torch.cat((out_forward_g, out_reverse_g), 1)

        packed_input_t = pack_padded_sequence(text_emb,
                                            Lens_t, batch_first=True,
                                            enforce_sorted=False)
        packed_output_t, _ = self.lstm_t(packed_input_t)
        output_t, _ = pad_packed_sequence(packed_output_t, batch_first=True)
        out_forward_t = output_t[range(len(output_t)), Lens_t - 1, :self.dimension_t]
        out_reverse_t = output_t[:, 0, self.dimension_t:]
        out_reduced_t = torch.cat((out_forward_t, out_reverse_t), 1)

        out_reduced = torch.cat((out_reduced_t, out_reduced_s,out_reduced_p,out_reduced_g), 1)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out

class noyaux_LSTM(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 hidden_t, 
                 hidden_f, 
                 embed_dim_t,
                 embed_dim_f,
                 nlstm_t,
                 nlstm_f,
                 dropout,
                 ):

        super(noyaux_LSTM, self).__init__()

        embed_dims = {  'Text' :embed_dim_t,
                        'gsp':embed_dim_f,
                        'gps':embed_dim_f,
                        'spg':embed_dim_f,
                        'sgp':embed_dim_f,
                        'pgs':embed_dim_f,
                        'psg':embed_dim_f}

        self.embeddings = nn.ModuleDict({k : nn.Embedding(vocab_size, embed_dims[k]) 
                                         for k in embed_dims.keys()})  
        self.dimension_t = hidden_t
        self.dimension_f = hidden_f

        self.lstm_t = nn.LSTM(input_size=embed_dim_t,
                            hidden_size=hidden_t,
                            num_layers=nlstm_t,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_gsp = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_gps = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_spg = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_sgp = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_pgs = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)
        self.lstm_psg = nn.LSTM(input_size=embed_dim_f,
                            hidden_size=hidden_f,
                            num_layers=nlstm_f,
                            batch_first=True,
                            bidirectional=False)

        self.drop = nn.Dropout(p = dropout)
        #self.fc = nn.Linear(2*hidden_t + 12*hidden_f, 6)
        self.fc = nn.Linear(hidden_t + 6*hidden_f, 6)
        
    def forward(self, text,gsp, gps,sgp, spg,pgs, psg,Lens_t,Lens_gsp, Lens_gps,Lens_sgp ,Lens_spg,Lens_pgs,Lens_psg):
        text_emb = self.embeddings['Text'](text)
        gsp_emb = self.embeddings['gsp'](gsp)
        gps_emb = self.embeddings['gps'](gps)
        spg_emb = self.embeddings['spg'](spg)
        sgp_emb = self.embeddings['sgp'](sgp)
        pgs_emb = self.embeddings['pgs'](pgs)
        psg_emb = self.embeddings['psg'](psg)


        packed_input_t = pack_padded_sequence(text_emb,
                                            Lens_t, batch_first=True,
                                            enforce_sorted=False)
        packed_output_t, _ = self.lstm_t(packed_input_t)
        output_t, _ = pad_packed_sequence(packed_output_t, batch_first=True)
        out_forward_t = output_t[range(len(output_t)), Lens_t - 1, :self.dimension_t]
        #out_reverse_t = output_t[:, 0, self.dimension_t:]
        #out_reduced_t = torch.cat((out_forward_t, out_reverse_t), 1)
        out_reduced_t = out_forward_t

        packed_input_gsp = pack_padded_sequence(gsp_emb,
                                            Lens_gsp, batch_first=True,
                                            enforce_sorted=False)
        packed_output_gsp, _ = self.lstm_gsp(packed_input_gsp)
        output_gsp, _ = pad_packed_sequence(packed_output_gsp, batch_first=True)
        out_forward_gsp = output_gsp[range(len(output_gsp)), Lens_gsp - 1, :self.dimension_f]
        #out_reverse_gsp = output_gsp[:, 0, self.dimension_f:]
        #out_reduced_gsp = torch.cat((out_forward_gsp, out_reverse_gsp), 1)
        out_reduced_gsp = out_forward_gsp

        packed_input_gps = pack_padded_sequence(gps_emb,
                                            Lens_gps, batch_first=True,
                                            enforce_sorted=False)
        packed_output_gps, _ = self.lstm_gps(packed_input_gps)
        output_gps, _ = pad_packed_sequence(packed_output_gps, batch_first=True)
        out_forward_gps = output_gps[range(len(output_gps)), Lens_gps - 1, :self.dimension_f]
        #out_reverse_gps = output_gps[:, 0, self.dimension_f:]
        #out_reduced_gps = torch.cat((out_forward_gps, out_reverse_gps), 1)
        out_reduced_gps = out_forward_gps

        packed_input_sgp = pack_padded_sequence(sgp_emb,
                                            Lens_sgp, batch_first=True,
                                            enforce_sorted=False)
        packed_output_sgp, _ = self.lstm_sgp(packed_input_sgp)
        output_sgp, _ = pad_packed_sequence(packed_output_sgp, batch_first=True)
        out_forward_sgp = output_sgp[range(len(output_sgp)), Lens_sgp - 1, :self.dimension_f]
        #out_reverse_sgp = output_sgp[:, 0, self.dimension_f:]
        #out_reduced_sgp = torch.cat((out_forward_sgp, out_reverse_sgp), 1)
        out_reduced_sgp = out_forward_sgp

        packed_input_spg = pack_padded_sequence(spg_emb,
                                            Lens_spg, batch_first=True,
                                            enforce_sorted=False)
        packed_output_spg, _ = self.lstm_spg(packed_input_spg)
        output_spg, _ = pad_packed_sequence(packed_output_spg, batch_first=True)
        out_forward_spg = output_spg[range(len(output_spg)), Lens_spg - 1, :self.dimension_f]
        #out_reverse_spg = output_spg[:, 0, self.dimension_f:]
        #out_reduced_spg = torch.cat((out_forward_spg, out_reverse_spg), 1)
        out_reduced_spg = out_forward_spg


        packed_input_pgs = pack_padded_sequence(pgs_emb,
                                            Lens_pgs, batch_first=True,
                                            enforce_sorted=False)
        packed_output_pgs, _ = self.lstm_pgs(packed_input_pgs)
        output_pgs, _ = pad_packed_sequence(packed_output_pgs, batch_first=True)
        out_forward_pgs = output_pgs[range(len(output_pgs)), Lens_pgs - 1, :self.dimension_f]
        #out_reverse_pgs = output_pgs[:, 0, self.dimension_f:]
        #out_reduced_pgs = torch.cat((out_forward_pgs, out_reverse_pgs), 1)
        out_reduced_pgs = out_forward_pgs

        packed_input_psg = pack_padded_sequence(psg_emb,
                                            Lens_psg, batch_first=True,
                                            enforce_sorted=False)
        packed_output_psg, _ = self.lstm_psg(packed_input_psg)
        output_psg, _ = pad_packed_sequence(packed_output_psg, batch_first=True)
        out_forward_psg = output_psg[range(len(output_psg)), Lens_psg - 1, :self.dimension_f]
        #out_reverse_psg = output_psg[:, 0, self.dimension_f:]
        #out_reduced_psg = torch.cat((out_forward_psg, out_reverse_psg), 1)
        out_reduced_psg = out_forward_psg

        out_reduced = torch.cat((out_reduced_t,
                                 out_reduced_gsp, out_reduced_gps,
                                 out_reduced_sgp, out_reduced_spg,
                                 out_reduced_pgs, out_reduced_psg,),
                                1)

        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out

