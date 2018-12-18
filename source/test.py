import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,FILTER_LENGTH1,NUM_FILTERS):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.embeddingXD = nn.Embedding(100, 128)
        self.conv1XD = nn.Conv1d(128, 32, FILTER_LENGTH1, padding=0)

    def forward(self, XDinput,XTinput):
        XDinput = torch.tensor(XDinput, dtype=torch.long)
        # print(XDinput)
        Embedding1 = self.embeddingXD(XDinput)
        # print(Embedding.shape)
        encode_smiles = F.relu(self.conv1XD(Embedding1))
        return x

model = Net(3,30)
criterion=nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
n_epochs = 30
