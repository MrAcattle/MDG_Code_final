import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from teCnnDataset import TeCnnDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import confusion_matrix, plot_confusion_matrix


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 2)
        self.conv3 = nn.Conv2d(128, 256, 2)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.liner_1 = nn.Linear(256 * 24 * 3, 300)
        self.liner_2 = nn.Linear(300, 22)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.liner_1(x))
        x = F.dropout(x)
        x = self.liner_2(x)
        return x

def collate(samples):
    matrixs, labels = map(list, zip(*samples))
    return torch.tensor(matrixs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


data_set = TeCnnDataset()
train_set, test_set = train_test_split(data_set, test_size=0.3, random_state=42)
batch_size = 64

data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         collate_fn=collate)

DEVICE = torch.device("cuda:0")
model = Classifier()
model.to(DEVICE)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

conf_matrix = torch.zeros(22, 22)

model.train()
epoch_losses = []
for epoch in range(25):
    epoch_loss = 0
    for iter, (batchMatrix, label) in enumerate(data_loader):
        batchMatrix = batchMatrix.unsqueeze(1)
        batchMatrix, label = batchMatrix.to(DEVICE), label.to(DEVICE)
        prediction = model(batchMatrix)
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
        for it, (batchMatrix, label) in enumerate(test_loader):
            batchMatrix = batchMatrix.unsqueeze(1)
            batchMatrix, label = batchMatrix.to(DEVICE), label.to(DEVICE)

            h = model(batchMatrix)
            pred = torch.softmax(h, 1)
            test_features += h.detach().cpu().numpy().tolist()
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()

            loss2 = loss_func(h, label)
            epoch_loss2 += loss2.detach().item()

    epoch_loss2 /= (it + 1)
    print('test loss ：Epoch {}, loss {:.4f}'.format(epoch, epoch_loss2))
    print("Test accuracy: ", accuracy_score(test_label, test_pred))

conf_matrix = confusion_matrix(test_pred, labels=test_label, conf_matrix=conf_matrix)
attack_types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                '19', '20', '21']
plot_confusion_matrix(conf_matrix.numpy(), classes=attack_types, normalize=True
                      ,
                      title='Normalized confusion matrix')