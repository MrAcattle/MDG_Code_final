import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import confusion_matrix, plot_confusion_matrix
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import accuracy_score
from tepNoDynamicDateset import TepNoDynamicDateset
from sklearn.model_selection import train_test_split


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, fc_dim, n_classes, dropout):
        super(Classifier, self).__init__()

        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(82 * hidden_dim, fc_dim)
        self.linear2 = nn.Linear(fc_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, this_batch_num):

        h = g.ndata['h'].float()
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv3(g, h))  # [N, hidden_dim]
        n = this_batch_num
        h = h.view(n, -1)
        h = F.relu(self.linear1(h))
        h = self.dropout(h)
        h = self.linear2(h)
        return h

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)


data_set = TepNoDynamicDateset()
train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=42)
batch_size = 128

data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         collate_fn=collate)
DEVICE = torch.device("cuda:0")

model = Classifier(10, 20, 300, data_set.num_labels, 0.5)
model.to(DEVICE)

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
conf_matrix = torch.zeros(data_set.num_labels, data_set.num_labels)

model.train()
epoch_losses = []
for epoch in range(30):
    epoch_loss = 0
    for iter, (batchg, label) in enumerate(data_loader):
        batchg, label = batchg.to(DEVICE), label.to(DEVICE)
        prediction = model(batchg, label.size(0))
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('train loss ：Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

    model.eval()
    epoch_loss2 = 0
    test_pred, test_label = [], []
    test_features = []

    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            batchg, label = batchg.to(DEVICE), label.to(DEVICE)

            h = model(batchg, label.size(0))
            pred = torch.max(h, 1)[1].view(-1)
            test_features += h.detach().cpu().numpy().tolist()
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
            loss2 = loss_func(h, label)
            epoch_loss2 += loss2.detach().item()

    epoch_loss2 /= (it + 1)
    print('test loss ：Epoch {}, loss {:.4f}'.format(epoch, epoch_loss2))
    print("Test accuracy: ", accuracy_score(test_label, test_pred))


conf_matrix = confusion_matrix(test_pred, labels=test_label, conf_matrix=conf_matrix)
# print(conf_matrix)
attack_types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21']
plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=True
                      ,
                      title='Normalized confusion matrix')
