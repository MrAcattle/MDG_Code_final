from torch.utils.data import Dataset
from utils import newReadData


class TeCnnDataset(Dataset):
    def __init__(self):
        # super(CnnBlastFurnaceDataset, self).__init__(name='CnnBlastFurnaceDataset')
        self.matrixs, self.labels = self.load()

    def load(self):
        whole_data, whole_label = newReadData(unit=list(range(52)), addAbsNode=False)
        matrixs = []
        labels = []
        neighbor = 5
        for i in range(whole_data.shape[0]):
            if i - neighbor < 0:
                left = 0
                right = left + neighbor * 2
            elif i + neighbor > whole_data.shape[0] - 1:
                right = whole_data.shape[0]
                left = right - neighbor * 2
            else:
                left = i - neighbor
                right = i + neighbor
            matrix = whole_data[left:right, :]
            label = whole_label[i]
            matrixs.append(matrix)
            labels.append(label)
        return matrixs, labels

    def __len__(self):
        return len(self.matrixs)

    def __getitem__(self, index):
        return self.matrixs[index], self.labels[index]

    def num_labels(self):
        return 22