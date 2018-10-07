import torch
import torch.nn as nn

import utils_model

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

        self.embed_word = #TODO
        self.lstm1 = Attention_LSTM(WORD_EMB_DIM,
                                    NB_HIDDEN_LSTM2,
                                    image_feature_dim,
                                    NB_HIDDEN_LSTM1)
        self.lstm2 = #TODO
        self.attention = #TODO
        self.predict_word = #TODO
    
    def forward(self, image_features, nb_timesteps):
        nb_batch, nb_image_feats, _ = image_features.size()
        v_mean = image_features.mean(dim=2)
        h1, c1, h2, c2, current_word = self.initialize_inference()
        y_out = utils.make_zeros((nb_batch, nb_timesteps, self.dict_size),
                                 cuda = image_features.is_cuda)
        for t in range(nb_timesteps):
            word_emb = self.emb_word(current_word)
            h1, c1 = self.lstm1(h1, c1, h2, v_mean, word_emb)
            v_hat = self.attention(h1, image_features)
            h2, c2 = self.lstm2(h2, c2, v_hat, h1)
            y, current_word = self.predict_word(h2)
            y_out[:,t] = y

        return y_out


#####################################################
#               LANGUAGE SUB-MODULES                #
#####################################################
class Attention_LSTM(nn.Module):
    def __init__(self, dim_word_emb, dim_lang_lstm, dim_image_feats, nb_hidden):
        self.lstm_cell = nn.LSTMCell(dict_lang_lstm+dim_image_feats+dim_word_emb,
                                     nb_hidden,
                                     bias=True)
        
    def forward(self, h1, c1, h2, v_mean, word_emb):
        input_feats = torch.cat((h2, v_mean, word_emb),dim=1)
        h_out, c_out = self.lstm_cell(input_feats, h1, c1)
        return h_out, c_out
