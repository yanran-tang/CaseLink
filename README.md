# CaseLink
Code for CaseLink paper (SIGIR 2024).

Title: [CaseLink: Inductive Graph Learning for Legal Case Retrieval](https://arxiv.org/abs/2403.17780)

Author: Yanran Tang, Ruihong Qiu, Hongzhi Yin, Xue Li, and Zi Huang

# Installation
Requirements can be seen in `/requirements.txt`

# Dataset
Datasets can be downloaded from [COLIEE2022](https://sites.ualberta.ca/~rabelo/COLIEE2022/) and [COLIEE2023](https://sites.ualberta.ca/~rabelo/COLIEE2023/). 

Specifically, the downloaded COLIEE2022 folders `task1_train_files_2022` and `task1_test_files_2022` should be put into `/Graph_generation/COLIEE2022/`. 

The label file `task1_train_labels_2022.json` and `task1_test_labels_2022.json` shoule be put into folder `/label/`. 

COLIEE2023 folders should be set in a similar way. 

The final project files are as follows:

    ```
    $ ./CaseGNN/
    .
    ├── DATASET
    │   └── data_load.py
    ├── Grpah_generation
    │   ├── bm25   
    │   │   ├── bm25_matrix
    │   │   ├── bm25_coliee+lecard.py
    │   │   └── bm25_model.py
    │   ├── casegnn_embedding
    │   ├── graph
    │   │   ├── graph_bin_2022
    │   │   └── graph_bin_2023
    │   ├── COLIEE2022
    │   ├── COLIEE2023
    │   ├── graph_construction.py
    │   └── federal_charges_coliee.txt
    ├── label 
    │   ├── hard_neg_top50_train_2022.json
    │   ├── hard_neg_top50_train_2023.json
    │   ├── task1_test_labels_2022.json            
    │   ├── task1_test_labels_2023.json 
    │   ├── task1_train_labels_2022.json 
    │   ├── task1_train_labels_2023.json 
    │   ├── test_2022_candidate_with_yearfilter.json
    │   └── test_2023_candidate_with_yearfilter.json     
    ├── CaseLink2022_run.sh
    ├── CaseLink2023_run.sh
    ├── Graph.sh
    ├── BM25_matrix_generation.sh
    ├── CaseLink_model.py
    ├── main.py
    ├── train.py
    ├── torch_metrics.py
    ├── requirements.txt
    └── README.md          
    ```

# Data Preparation
## 1. BM25 ranking matrix generation
  - Run `. ./BM25_matrix_generation.sh` to generate COLIEE2022 bm25 ranking matrix files in folder `./Graph_generation/bm25/bm25_matrix/`.

  - The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `BM25_matrix_generation.sh`.

  - The BM25 ranking matrix can be also downloaded [here](https://drive.google.com/drive/folders/1R4ggI9Tq-dES-gtLsE_s1odP7yWuW5n4?usp=drive_link).


## 2. CaseGNN case embedding generation
  - To generate the CaseGNN case embeddings, please run the CaseGNN2022_run.sh and CaseGNN2023_run.sh in [CaseGNN github](https://github.com/yanran-tang/CaseGNN)

  - The CaseGNN case embedding can be also downloaded [here](https://drive.google.com/drive/folders/15-l-BJn9X5xpeaHDyk8VE0ncTPR6qaXS?usp=drive_link).

# CaseLink Graph Construction
- CaseLink graph constrction utilises the result of bm25 ranking result and case embedding from CaseGNN, please ensure the folders of  `./Graph_generation/bm25/bm25_matrix/` and `./Graph_generation/casegnn_embedding/` have been generated or downloaded.
- Run `. ./Graph.sh`
  - `--topk_neighbor` can be chosen from {3, 5, 10, 20}. 
  - `--charge_threshold` can be chosen from {0.85, 0.9, 0.95}.

- The CaseLink graphs are saved in folder `/Graph_generation/graph/`

- The same process for COLIEE2023, please change the `--data 2022` to `--data 2023` in `Graph.sh`.

- The CaseLink graphs can be also downloaded [here](https://drive.google.com/drive/folders/1jOYhGDFlLcvga7Ij4SF72MYsn0ulfSmr?usp=sharing).

# Model Training
## 1. CaseGNN Model Training
Run `. ./CaseLink2022_run.sh` and `. ./CaseLink2023_run.sh` for COLIEE2022 and COLIEE2023, respectively.

# Cite
If you find this repo useful, please cite
```
@article{CaseLink,
  author       = {Yanran Tang and
                  Ruihong Qiu and
                  Hongzhi Yin and
                  Xue Li and
                  Zi Huang},
  title        = {CaseLink: Inductive Graph Learning for Legal Case Retrieval},
  journal      = {CoRR},
  volume       = {abs/2403.17780},
  year         = {2024}
}
```