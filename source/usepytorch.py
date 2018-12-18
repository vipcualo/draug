import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,FILTER_LENGTH1,NUM_FILTERS):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.embedding=nn.Embedding(100,64)
        self.conv1 = nn.Conv1d(30, NUM_FILTERS, FILTER_LENGTH1, padding=0)
        self.conv1 = nn.Conv1d(NUM_FILTERS, NUM_FILTERS*2, FILTER_LENGTH1, padding=0)
        self.conv1 = nn.Conv1d(NUM_FILTERS*2, NUM_FILTERS*3, FILTER_LENGTH1, padding=0)
        self.fc1 = nn.Linear(1024, 10)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4= nn.Linear(512,1)
        self.dropout = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)

    def forward(self, XDinput,XTinput):
        Embedding=self.embedding(XDinput)
        encode_smiles=F.relu(self.conv1(Embedding))
        encode_smiles = F.relu(self.conv2(encode_smiles))
        encode_smiles = F.relu(self.conv3(encode_smiles))
        x = torch.mean(encode_smiles.view(encode_smiles.size(0), encode_smiles.size(1), -1), dim=2)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc3(1))
        return x

model = Net(3,30)
criterion=nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
n_epochs = 30
