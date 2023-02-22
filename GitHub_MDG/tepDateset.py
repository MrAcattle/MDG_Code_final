import numpy as np
from dgl.data import DGLDataset
import torch
import dgl
from utils import newReadData
import os


class TepDataset(DGLDataset):
    def __init__(self):
        super(TepDataset, self).__init__(name='TepDateset')

    def process(self):
        self.graphs, self.labels = self._load_graph()


    def _load_graph(self):
        src = np.array(
            [1, 0, 2, 41, 42, 43, 57, 58, 59, 57, 58, 59, 61, 61, 61, 61, 61, 61, 61, 61, 52, 52, 52, 52, 52, 66, 8, 50,
             51,
             53, 53, 54, 54, 54, 54, 54, 63, 63, 63, 63, 63, 63, 63, 63, 63, 9, 46, 64, 13, 47, 64, 55, 55, 55, 14, 67,
             49,
             16, 55, 48, 65, 65, 65, 65, 65, 55, 65, 60, 60, 44, 3, 62, 56, 62, 56, 45, 56, 10, 22, 22, 24, 24, 70, 73,
             68,
             68, 72, 72, 69, 71, 16, 77, 77, 77, 77, 77, 77, 77, 77, 77, 39, 74, 74, 6, 80, 11, 81, 7, 78, 14, 79])
        dst = np.array(
            [43, 41, 42, 57, 58, 59, 0, 1, 2, 61, 61, 61, 22, 23, 24, 25, 26, 27, 5, 52, 6, 7, 8, 53, 66, 20, 50, 66,
             53,
             21, 54, 11, 12, 64, 56, 63, 28, 29, 30, 31, 32, 33, 34, 35, 9, 46, 63, 13, 47, 64, 55, 14, 15, 17, 67, 18,
             67,
             48, 65, 65, 36, 37, 38, 39, 40, 61, 16, 55, 3, 60, 44, 61, 62, 4, 19, 56, 10, 51, 70, 73, 70, 73, 68, 72,
             69,
             71, 69, 71, 44, 43, 77, 48, 76, 75, 44, 42, 43, 41, 46, 47, 74, 75, 76, 80, 46, 81, 47, 78, 51, 79, 48])

        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])

        dataMatrix, labelMatrix = newReadData(unit=list(range(52)))

        graphs = []
        labels = []

        neighbor = 5
        for i in range(dataMatrix.shape[0]):
            graph = dgl.DGLGraph((u, v))
            dgl.add_self_loop(graph)

            if i - neighbor < 0:
                left = 0
                right = left + neighbor * 2
            elif i + neighbor > dataMatrix.shape[0] - 1:
                right = dataMatrix.shape[0]
                left = right - neighbor * 2
            else:
                left = i - neighbor
                right = i + neighbor

            neighbor_nodes = [x for x in range(left, right) if x != i]

            # add dynamic weight
            di = []
            for j in range(neighbor * 2 - 1):
                d = np.linalg.norm(dataMatrix.T[:, i] - dataMatrix.T[:, neighbor_nodes[j]])
                di.append(d)
            beta = np.mean(di)
            di = [np.exp(-x / beta) for x in di]
            di = [x / np.sum(di) for x in di]

            update_node = []
            for p, index in enumerate(neighbor_nodes):

                if i == p == 0:
                    update_node.append(dataMatrix.T[:, i])

                update_node.append(di[p] * dataMatrix.T[:, index])
                if index + 1 == i:
                    update_node.append(dataMatrix.T[:, i])

            graph.ndata['h'] = torch.tensor(np.array(update_node).T)
            label = labelMatrix[i]

            graphs.append(graph)
            labels.append(label)


        return graphs, labels

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def num_labels(self):
        return 22
