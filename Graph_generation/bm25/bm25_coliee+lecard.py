import os
import sys
sys.path.append('.')
sys.path.append('..')

import tqdm
from tqdm import tqdm
import json
import torch

from bm25_model import BM25

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## Training config
parser.add_argument("--ngram_1", type=int, default=3,
                    help="ngram")
parser.add_argument("--ngram_2", type=int, default=4,
                    help="ngram")
parser.add_argument("--data", type=str, default='2023', help="coliee2022 or coliee2023")
parser.add_argument("--dataset", default='train', type=str, help="train or test")
args = parser.parse_args()

print(args)

# bm25 computing starts    
corpus =[]
corpus_sequence_names = []
corpus_dict = {}
path = os.getcwd()
RDIR = path+'/Graph_generation/COLIEE'+args.data+'/task1_'+args.dataset+'_files_'+args.data
files = os.listdir(RDIR)

print('Corpus loading: ')
for pfile in tqdm(files[:]):
    file_name = pfile.split('.')[0]+'.txt'
    with open(os.path.join(RDIR, pfile), 'r') as f:
        original_text = f.read()
        f.close()

    text = original_text

    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('"', '')
    text = text.replace(',', '')
    text = text.replace('\\', '')

    txt = text
    corpus.append(txt)
    corpus_dict[file_name] = txt
    corpus_sequence_names.append(file_name)

bm25 = BM25(ngram_range=(args.ngram_1, args.ngram_2))
bm25.fit(corpus)

score_dict = {}
prediction_dict = {}
final_prediction_dict = {}
print('BM25 calculation start: ')
for i in tqdm(range(len(corpus_sequence_names))):
    if i == 0:
        query_name = corpus_sequence_names[i]
        print(query_name)
        que_text = corpus_dict[query_name]
        doc_scores = bm25.transform(que_text, corpus)
        bm25_matrix = torch.unsqueeze(torch.FloatTensor(doc_scores), 0)
    else:
        query_name = corpus_sequence_names[i]
        print(query_name)
        que_text = corpus_dict[query_name]
        doc_scores = bm25.transform(que_text, corpus)
        score_tensor = torch.unsqueeze(torch.FloatTensor(doc_scores), 0)         
        bm25_matrix = torch.cat((bm25_matrix, score_tensor), 0) 
        
torch.save(bm25_matrix, path+'/Graph_generation/bm25/bm25_matrix/coliee'+args.data+'_'+args.dataset+'_BM25_score_matrix.pt')
with open (path+'/Graph_generation/bm25/bm25_matrix/coliee'+args.data+'_'+args.dataset+'_BM25_score_matrix_case_sequence.json', 'w') as f:
    json.dump(corpus_sequence_names, f)
    f.close()

print('Finished.')