import torch
from torch import nn
import torch.nn.functional as F
import re


Embed_dim = 300
class Rnn(nn.Module):
    
    def __init__(self):
        super(Rnn, self).__init__()
        #self.rnnL1 = nn.RNN(input_size=Embed_dim ,hidden_size=1024,
        #        nonlinearity= 'tanh' ,num_layers=num_layers, batch_first= True)
        self.rnnL1 = nn.RNNCell(input_size=Embed_dim ,hidden_size=1024)
        self.rnnL2 = nn.RNNCell(input_size=1024 ,hidden_size=256)
        self.linL3 = nn.Linear(256,5)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
    def forward(self,x):
        #x,hidden =self.rnnL1(x)
        x=self.dropout(self.rnnL1(x))
        x=self.rnnL2(x)
        x=self.relu(self.linL3(x))
        x=self.softmax(x)
        
        return x



class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.Q = nn.Linear(d_in,d_out)
        self.V = nn.Linear(d_in,d_out)
        self.K = nn.Linear(d_in,d_out)

    def forward(self, x):
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)
        
        scores = torch.bmm(queries, keys.transpose(2,1))
        scores /= (self.d_out**0.5)
        attention = F.softmax(scores, dim=2)
        hidden_state = torch.bmm(attention, values)
        return hidden_state





#review_set = {word.replace('/', '') for word in review_set}

#Test section (main) - relevant but after attempting to implement the attention mechanism
"""SOS_token = 0 #Start of Sequence
EOS_token = 1 #End of Sequence

index2words = {
    SOS_token : 'SOS',
    EOS_token : 'EOS'
}
words = " How are you doing ? I am good and you ?"
words_list = set(words.lower().split(' '))

for word in words_list:
    index2words[len(index2words)] = word
words2index = {w:i for i, w in index2words.items()}
print(words2index)"""

def tupleToArray(str_tuple)->list:
    str_tuple = str(str_tuple).strip("{}")
    str_tuple = str_tuple.split(' ')
    str_tuple = [word.strip("(),'") for word in str_tuple]
    return [word.strip('"') for word in str_tuple]




def organize_text(txt_array) -> list:
    review_set = [word.replace('\\', '') for sentence in txt_array
               for word in re.split(',| ',sentence)]
    #review_set = [word.replace('\\', '') for word in review_set]
    return review_set