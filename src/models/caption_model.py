import torch
import torch.nn as nn

import models.utils_models as utils

#########################################
#               CONSTANTS               #
#########################################
WORD_EMB_DIM    = 1000
NB_HIDDEN_LSTM1 = 1000
NB_HIDDEN_LSTM2 = 1000
NB_HIDDEN_ATT   = 512


#########################################
#               MAIN MODEL              #
#########################################
class Caption_Model(nn.Module):
    def __init__(self, dict_size, image_feature_dim):
        super(Caption_Model, self).__init__()
        self.dict_size = dict_size
        self.image_feature_dim = image_feature_dim

        self.embed_word = nn.Linear(dict_size, WORD_EMB_DIM, bias=False)
        self.lstm1 = Attention_LSTM(WORD_EMB_DIM,
                                    NB_HIDDEN_LSTM2,
                                    image_feature_dim,
                                    NB_HIDDEN_LSTM1)
        self.lstm2 = Language_LSTM(NB_HIDDEN_LSTM1,
                                   NB_HIDDEN_ATT,
                                   NB_HIDDEN_LSTM2)
        self.attention = Visual_Attention(image_feature_dim,
                                          NB_HIDDEN_LSTM1,
                                          NB_HIDDEN_ATT)
        self.predict_word = nn.Linear(NB_HIDDEN_LSTM2, dict_size)
    
    def forward(self, image_features, nb_timesteps, true_words):
        nb_batch, nb_image_feats, _ = image_features.size()
        v_mean = image_features.mean(dim=2)
        h1, c1, h2, c2, current_word = self.initialize_inference()
        y_out = utils.make_zeros((nb_batch, nb_timesteps),
                                 cuda = image_features.is_cuda)
        for t in range(nb_timesteps):
            word_emb = self.emb_word(current_word)
            h1, c1 = self.lstm1(h1, c1, h2, v_mean, word_emb)
            v_hat = self.attention(h1, image_features)
            h2, c2 = self.lstm2(h2, c2, v_hat, h1)
            y, current_word = self.predict_word(h2, true_word[t])
            y_out[:,t] = y

        return y_out


#####################################################
#               LANGUAGE SUB-MODULES                #
#####################################################
class Attention_LSTM(nn.Module):
    def __init__(self, dim_word_emb, dim_lang_lstm, dim_image_feats, nb_hidden):
        super(Attention_LSTM,self).__init__()
        self.lstm_cell = nn.LSTMCell(dim_lang_lstm+dim_image_feats+dim_word_emb,
                                     nb_hidden,
                                     bias=True)
        
    def forward(self, h1, c1, h2, v_mean, word_emb):
        input_feats = torch.cat((h2, v_mean, word_emb),dim=1)
        h_out, c_out = self.lstm_cell(input_feats, h1, c1)
        return h_out, c_out

class Language_LSTM(nn.Module):
    def __init__(self, dim_att_lstm, dim_visual_att, nb_hidden):
        super(Language_LSTM,self).__init__()
        self.lstm_cell = nn.LSTMCell(dim_att_lstm+dim_visual_att,
                                     nb_hidden,
                                     bias=True)
        
    def forward(self, h2, c2, h1, v_hat):
        input_feats = torch.cat((h1, v_hat, word_emb),dim=1)
        h_out, c_out = self.lstm_cell(input_feats, h2, c2)
        return h_out, c_out


class Visual_Attention(nn.Module):
    def __init__(self, dim_image_feats, dim_att_lstm, nb_hidden):
        super(Visual_Attention,self).__init__()
        self.fc_image_feats = nn.Linear(dim_image_feats, nb_hidden, bias=False)
        self.fc_att_lstm = nn.Linear(dim_att_lstm, nb_hidden, bias=False)
        self.act_tan = nn.Tanh()
        self.fc_att = nn.Linear(nb_hidden, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_feats, h1):
        nb_batch, nb_feats, feat_dim = image_feats.size()

        att_lstm_emb = self.fc_att_lstm(h1).unsqueeze(1)
        image_feats_emb = self.fc_image_feats(image_feats)
        all_feats_emb = image_feats_emb + att_lstm_emb.repeat(1,nb_feats,1)

        activate_feats = self.act_tan(all_feats_emb)
        unnorm_attention = self.fc_att(activate_feats)
        normed_attention = self.softmax(unnorm_feats).unsqueeze(2)

        weighted_feats = normed_attention.repeat(1,1,nb_feats) * image_feats
        attended_image_feats = weighted_feats.sum(dim=2)
        return attended_image_feats

class Predict_Word(nn.Module):
    def __init__(self, dim_language_lstm, dict_size):
        super(Predict_Word, self).__init__()
        self.fc = nn.Linear(dim_language_lstm, dict_size)
        
    def forward(self, h2, true_word):
        true_idx = (true_word==1).nonzero()
        y = self.fc(h2)[true_idx]
        return y, true_word
