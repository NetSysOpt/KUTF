# KKT-based Unsupervised Training Framework (Still improving)

## Introduction

Solving complex optimization problems with machine learning, commonly referred to as Learn-to-Optimize, has attracted significant attention due to growing industrial demands.
However, existing L2O methods typically rely on large labeled datasets, where generating labels incurs substantial computational overhead. 
Furthermore, unlike traditional optimization algorithms, these methods often target a single dataset, limiting their ability to generalize to out-of-distribution instances.
To address these limitations, we propose KUTF, an unsupervised learning framework that leverages KKT conditions in its loss function, enabling effective training for convex optimization problems without the need for labeled data.
Additionally, KUTF mimics classical iterative optimization behaviors, unlocking the potential for auto-regressive improvements that progressively enhance solution quality.
We evaluate KUTF on several publicly available datasets, each containing optimization problems from diverse domains and backgrounds. 
Extensive experiments demonstrate that KUTF achieves over 10X speedup compared to conventional solvers, highlighting its efficiency and strong generalization to out-of-distribution instances.

## Usage

### Dateset
To generate datasets, you can use the script ./ins/src/gen_ins.py. The code provides detailed usage instructions.

### Sample Collection
After generating instances, you can use ./src/julia/PDQP.jl/gen_bat.py to generate a task file that collects training/testing samples from generated cases. The code provides detailed usage instructions.
Then, please run ./src/extract_sample.py -t XXX to extract pickle files. (XXXX is the dataset name. For example, if you have gen_train_XXXX in your ins folder, you will use -t XXXX.)

### Training
Using ./src/train_new.py to train the unsupervised model, ./src/train_supervised.py to train the supervised model, and ./src/train_gnn.py for the mentioned GNNs model.

### Test
After training, you first need to generate predictions by running ./src/predict_*.py.
Then, use ./src/julia/PDQP.jl/gen_bat.py to generate a batch file that runs the test.

## Reference to used Repository
**PDQP.jl**: [Lu, Haihao, and Jinwen Yang. "A practical and optimal first-order method for large-scale convex quadratic programming." arXiv preprint arXiv:2311.07710 (2023).](https://github.com/jinwen-yang/PDQP.jl)

## Contact
TODO
