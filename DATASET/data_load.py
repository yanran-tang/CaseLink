import dgl
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
import random

import json
import torch

import os


# # construct graph data bin
#
# # graph_and_label = load_graphs("./test_data_2022.bin")
# graph_and_label = load_graphs("./train_data_2022.bin")
# CaseGraph = {}
# graphs = graph_and_label[0]
# labels = graph_and_label[1]['name_list'].tolist()
#
# for i in range(len(labels)):
#     CaseGraph[str(int(labels[i])).zfill(6)] = graphs[i]
#
# with open('./train_labels_2022.json', 'r') as f:
# # with open('./test_labels_2022.json', 'r') as f:
#     noticed_case_list = json.load(f)
#     f.close()
#
# query_graph_dict = {}
# query_graph_list = []
# query_graph_label = []
# for key, value in noticed_case_list.items():
#     k = key.split('.')[0]
#     query_graph_dict.update({k: (CaseGraph[str(k)])})
#     query_graph_list.append(CaseGraph[str(k)])
#     query_graph_label.append(int(k))
# print(query_graph_list, query_graph_label)
#
# graph_labels = {"glabel": torch.Tensor(query_graph_label)}
#
# save_graphs("./task1_test_data_2022_Synthetic.bin", query_graph_list, graph_labels)


class PoolDataset(DGLDataset):
    def __init__(self, file_path):
        self.graph_and_label = load_graphs(file_path)
        super(PoolDataset, self).__init__(name='Pool')

    def process(self):
        case_pool, pool_label = self.graph_and_label
        CaseGraph = {}
        graphs = case_pool
        labels = pool_label['name_list'].tolist()
        for y in range(len(labels)):
            CaseGraph.update({str(int(labels[y])): graphs[y]})
            CaseGraph.update({str(int(labels[y])).zfill(6): graphs[y]})
        self.graphs = CaseGraph

        label_dict = {}
        labels = pool_label['name_list'].tolist()
        for x in range(len(labels)):
            label_dict.update({str(int(labels[x])): [str(int(labels[x]))]})
            label_dict.update({str(int(labels[x])).zfill(6): [str(int(labels[x])).zfill(6)]})
        self.labels = label_dict

        self.graph_list = case_pool
        self.label_list = pool_label['name_list'].tolist()
        # self.label_list = torch.LongTensor(self.label_list)

    def __getitem__(self, i):
        return self.graph_list[i], self.label_list[i]

    def __len__(self):
        return len(self.graphs)


class SyntheticDataset(DGLDataset):
    def __init__(self, file_path, label_dict, train_pool, hard_neg_num, hard_bm25_dict):
        self.graph_and_label = torch.load(file_path)
        self.label_dict = label_dict
        self.hard_bm25_dict = hard_bm25_dict
        self.train_pool = train_pool
        self.hard_neg_num = hard_neg_num

        super(SyntheticDataset, self).__init__(name='Synthetic')

    def process(self):
        # graphs = self.graph_and_label[0]
        # labels = self.graph_and_label[1]['case_name_list']
        self.graph_list = self.graph_and_label[0]
        self.label_list = self.graph_and_label[1]['case_name_list']
        self.label_list = [str(int(self.label_list[x])).zfill(6) for x in range(len(self.label_list))]
        self.case_index = self.graph_and_label[2]
        # self.label_list = torch.LongTensor(self.label_list)
        label_dict = self.label_dict
        hard_bm25_dict = self.hard_bm25_dict
        train_pool = self.train_pool
        hard_neg_num = self.hard_neg_num
        pos_case_list = []
        ran_neg_list = []
        hard_neg_list = []
        query_index_list = []
        for x in range(len(self.label_list)):
            query_name = self.label_list[x]+'.txt'
            pos_case = random.choice(label_dict[query_name]).split('.')[0]
            pos_case_index = train_pool.index(pos_case)
            pos_case_list.append(pos_case_index)
            query_index_list.append(train_pool.index(self.label_list[x]))

            i = 0
            while i<4400: 
                ran_neg_case = random.choice(train_pool)
                if ran_neg_case+'.txt' not in label_dict[query_name]:
                    break
                break
            
            ran_neg_case_index = train_pool.index(ran_neg_case)
            ran_neg_list.append(ran_neg_case_index)
            
            hard_neg_sublist = []
            for i in range(hard_neg_num):
                bm25_neg_case = random.choice(hard_bm25_dict[query_name]).split('.')[0]
                bm25_neg_case_index = train_pool.index(bm25_neg_case)
                hard_neg_sublist.append(bm25_neg_case_index)  
            hard_neg_list.append(hard_neg_sublist)
               
        self.pos_case_list = pos_case_list
        self.ran_neg_list = ran_neg_list
        self.hard_neg_list = hard_neg_list
        self.query_index_list = query_index_list
    
    def __getitem__(self, i):
        return self.query_index_list[i], self.label_list[i], self.pos_case_list[i], self.ran_neg_list[i], self.hard_neg_list[i]

    def __len__(self):
        return len(self.graph_list)


def collate(samples):
    query_index, labels, pos_case, ran_neg, hard_neg = map(list, zip(*samples))
            
    return query_index, labels, pos_case, ran_neg, hard_neg
