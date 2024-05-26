import numpy as np
import torch
import json
from tqdm import tqdm
import os

import dgl
from dgl.data.utils import save_graphs
from transformers import AutoModel, AutoTokenizer

import argparse
parser = argparse.ArgumentParser()
## model parameters
parser.add_argument("--data", type=str, default='2022', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", default='test', type=str, help="train or test")
parser.add_argument("--topk_neighbor", default=5, type=int, help="5 10 20")
parser.add_argument("--charge_threshold", default=0.9, type=float, help="0.85 0.9 0.95") 
args = parser.parse_args()

model_name = 'CSHaitao/SAILER_en_finetune'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

##Load bm25 matrix
path = os.getcwd()
with open(path+'/Graph_generation/bm25/bm25_matrix/coliee'+args.data+'_'+args.dataset+'_BM25_score_matrix_case_sequence.json', 'rb') as fIn:
    bm25_score_matrix_case_name = json.load(fIn)
with open(path+'/Graph_generation/bm25/bm25_matrix/coliee'+args.data+'_'+args.dataset+'_BM25_score_matrix.pt', "rb") as fIn:
    bm25_score_matrix =torch.load(fIn) 
bm25_score_matrix = bm25_score_matrix.to('cpu')

##bm25
bm25_score_tensor = bm25_score_matrix.tolist()
bm25_score_matrix_norm_list = []
for i in tqdm(range(len(bm25_score_tensor))):
    bm25_score_tensor_norm = torch.tensor(bm25_score_tensor[i])/bm25_score_tensor[i][i]    
    bm25_score_matrix_norm_list.append(bm25_score_tensor_norm)
bm25_score_matrix_norm = torch.stack(bm25_score_matrix_norm_list)
bm25_score_norm_top10_tensor = torch.topk(bm25_score_matrix_norm, args.topk_neighbor, dim=1)[1]
bm25_score_list = []
for i in range(len(bm25_score_matrix_norm_list)):
    bm25_score_matrix_norm_list[i][bm25_score_norm_top10_tensor[i]] = 1
    tensor = (bm25_score_matrix_norm_list[i]>=1).float() 
    bm25_score_list.append(tensor)
bm25_score_matrix_binary = torch.stack(bm25_score_list)

## charge matrix
dataset_path = './Graph_generation/COLIEE'+args.data+'/task1_'+args.dataset+'_files_'+args.data+'/'
with open('./Graph_generation/federal_charges_coliee.txt', 'r') as f:
    issues  = f.readlines()
    f.close()
issue_name_list = []
encode_issue_name_list = []
for issue in issues:
    issue_1 = issue.split('\n')[0]
    issue_name_list.append(issue_1)
    if ',' in issue_1:
        issue_name = issue_1.split(',')[0]
        encode_issue_name_list.append(issue_name)
    else:
        encode_issue_name_list.append(issue_1)

charge_name_tokenized_id = tokenizer(encode_issue_name_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
charge_name_embedding = model(**charge_name_tokenized_id)[0][:,0,:]
charge_name_embedding_norm = charge_name_embedding / charge_name_embedding.norm(dim=1)[:, None]
charge_matrix = torch.mm(charge_name_embedding_norm, charge_name_embedding_norm.T)
binary_charge_matrix = (charge_matrix>args.charge_threshold).float()

##charge matrix construction
num = 0
empty_charge_case_num = 0
case_charge_dict = {}
max_charge_num = 0
for file in tqdm(bm25_score_matrix_case_name):
    num += 1
    case_charge_list = []      
    with open (dataset_path+file, 'r') as f:
        text = f.read()
        f.close()
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('"', '')
    text = text.replace(',', '')
    text = text.replace('\\', '')
    text = text.replace('\n', ' ')
    for i in issue_name_list:
        if ',' in i:
            multi_charge_list = i.split(',')
            multi_charge_list_1 = []
            for x in multi_charge_list:
                if x in text:
                    case_charge_list.append(1)
                    multi_charge_list_1.append(x)
                    break
                elif x.lower() in text:
                    case_charge_list.append(1)
                    multi_charge_list_1.append(x)
                    break
            if multi_charge_list_1 == []:
                case_charge_list.append(0)
        else:
            if i in text:
                case_charge_list.append(1)
            elif i.lower() in text:
                case_charge_list.append(1)
            else:
                case_charge_list.append(0)
    if 1 not in case_charge_list:
        empty_charge_case_num += 1
        print(file)

    if num == 1:
        binary_case_charge_tensor = torch.unsqueeze(torch.FloatTensor(case_charge_list), 1)
    else:
        tensor_0 = torch.unsqueeze(torch.FloatTensor(case_charge_list), 1)
        binary_case_charge_tensor = torch.cat((binary_case_charge_tensor, tensor_0), 1) 
print('Empty charge case: '+str(empty_charge_case_num))  

binary_edge_matrix_1 = torch.cat((bm25_score_matrix_binary, binary_case_charge_tensor), 0)  ##concat (bm25_score_matrix[case_num, case_num], case_charge_matrix[charge_num, case_num]) --> [case_num+charge_num, case_num]
binary_edge_matrix_2 = torch.cat((binary_case_charge_tensor.T, binary_charge_matrix), 0)  ##concat (case_charge_matrix[charge_num, case_num], charge_matrix[charge_num, charge_num]) --> [case_num+charge_num, charge_num]
edge_adjacency_matrix = torch.cat((binary_edge_matrix_1, binary_edge_matrix_2), 1)

weight_edge_matrix_1 = torch.cat((bm25_score_matrix_norm, binary_case_charge_tensor), 0) ##concat (bm25_score_matrix[case_num, case_num], case_charge_matrix[charge_num, case_num]) --> [case_num+charge_num, case_num]
weight_edge_matrix_2 = torch.cat((binary_case_charge_tensor.T, binary_charge_matrix), 0)  ##concat (case_charge_matrix[charge_num, case_num], charge_matrix[charge_num, charge_num]) --> [case_num+charge_num, charge_num]
edge_weight_matrix = torch.cat((weight_edge_matrix_1, weight_edge_matrix_2), 1)

# ##Load node embedding  
with open(path+'/Graph_generation/casegnn_embedding/coliee'+args.data+'_'+args.dataset+'_casegnn_embedding.pt', "rb") as fIn:
    case_embedding_matrix =torch.load(fIn) 
case_embedding_matrix = case_embedding_matrix.to('cpu')
with open(path+'/Graph_generation/casegnn_embedding/coliee'+args.data+'_'+args.dataset+'_casegnn_embedding_case_name_list.json', 'rb') as fIn:
    case_embedding_matrix_case_name_list = json.load(fIn)

node_embedding = []
for i in range(len(bm25_score_matrix_case_name)):
    case_name = bm25_score_matrix_case_name[i]
    index = case_embedding_matrix_case_name_list.index(case_name.split('.')[0])

    case_embedding = case_embedding_matrix[index,:]
    node_embedding.append(case_embedding)
case_node_embedding_matrix = torch.stack(node_embedding).to('cpu')

charge_node_embedding = torch.cat((charge_name_embedding,charge_name_embedding), 1)
node_embedding = torch.cat((case_node_embedding_matrix, charge_node_embedding), 0)

top10_edge_weight_matrix = edge_adjacency_matrix.mul(edge_weight_matrix)

weight_ajacency = (top10_edge_weight_matrix + top10_edge_weight_matrix.T) / (edge_adjacency_matrix + edge_adjacency_matrix.T)
weight_ajacency = weight_ajacency.nan_to_num()

src, dst = np.nonzero(weight_ajacency.numpy())
g = dgl.graph((src, dst))
g.ndata['feat'] = node_embedding

num = 0
edge_weight = []
for i in range(len(src)):
    if src[i] == num:
        edge_weight.append(weight_ajacency[src[i],dst[i]])
    else:
        num += 1
        edge_weight.append(weight_ajacency[src[i],dst[i]])
g.edata['feat'] = torch.stack(edge_weight)

graph_labels = {}
node_name = [int(i.split('.')[0])  for i in bm25_score_matrix_case_name]
tensor_node_name = torch.FloatTensor(node_name)
graph_labels.update({'case_name_list': tensor_node_name})
save_graphs(path+"/Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+args.dataset+"_bm25top"+str(args.topk_neighbor)+"_charge_thres"+str(args.charge_threshold)+".bin", g, graph_labels)

with open('./label/task1_'+args.dataset+'_labels_'+args.data+'.json', 'r') as f:
    noticed_case_list = json.load(f)
    f.close()

labels = graph_labels['case_name_list'].tolist()
graphs = g.ndata['feat']
query_graph_list = []
query_graph_label = []
index_list = []
for key, value in noticed_case_list.items():
    k = key.split('.')[0]
    index = labels.index(int(k))
    index_list.append(index)
    query_graph_list.append(graphs[index,:])
    query_graph_label.append(int(k))
graph_labels = {"case_name_list": torch.FloatTensor(query_graph_label)}

torch.save([query_graph_list, graph_labels, index_list], path+"/Graph_generation/graph/graph_bin_"+args.data+"/bidirec_"+args.data+args.dataset+"_bm25top"+str(args.topk_neighbor)+"_charge_thres"+str(args.charge_threshold)+'.pt')
print('CaseLink graph construction finished.')