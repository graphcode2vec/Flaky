import torch
import torch.nn as nn



class PredictionLinearFlakyModelFineTune( nn.Module ):
    def __init__(self,in_dim, out_dim, encoder, dropratio=0.25):
        super(PredictionLinearFlakyModelFineTune, self).__init__()
        self.encoder = encoder
        self.dense = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropratio)
        self.out_proj = nn.Linear( in_dim, out_dim)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, batch):
        #batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.ins_length
        x_s, _, _ = self.encoder.getVector(batch)   
     #   torch.subtract(x1-x2)
       # x = self.dropout(x_s)
        x = self.dense(x_s)
        x = torch.relu(x)
        x = self.dropout(x)
        return  self.softmax(self.out_proj(x))
    
    def loadWholeModel(self, model_file, device, maps={} ):
        gnn_weights = torch.load(model_file,  map_location="cpu")
        self.load_state_dict(gnn_weights)
