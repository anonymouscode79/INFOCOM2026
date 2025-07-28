
## To Run our Method 
#### **You can find the command with specific hyper parameters in the configurations.md.**
**The Drive link to the dataset** : https://drive.google.com/drive/folders/1Y8oBeMAbrQzpOItXQdjN1bvn_HvId5D2?usp=sharing
```bash
python3 SSCL_main.py --ds=<dataset> --training_cutoff=<training_cutoff_value> --lr=<learning_rate> --wd=<weight_decay>  --label_ratio=1 --nps=<int> --bool_gpm=<bool> --b_m=<float> --bma=<float>  --analyst_labels=<int> --upper_threshold=<float>
```
- `--ds`: Specifies the name of the dataset to use.
- `--training_cutoff`: Defines the cutoff point for separating training and testing data.
- `--lr`: Sets the learning rate for the model's optimizer.
- `--wd`: Determines the weight decay for optimizer regularization.
- `--label_ratio`: Specifies the ratio of labeled data for seen tasks used during training(0.2 for 20 %).
- `--nps`: Number of projection samples for constructing gradient projection Memory.
- `--bool_gpm`: Enables or disables the gradient projection mechanism.
- `--b_m`: Sets the batch memory ratio for training.
- `--bma`: Allocates the batch minority ratio during training.
- `--upper_thresh`:Maximum Upper threshold on cosine distance to find the suitable label samples .
- `--analyst_labels`: Number of labeled samples provided by analysts for semi-supervised learning. 

#### To run it on your device

**Use the requirements.txt file to install the required libraries.**

**Use Python Version 3.8.13 and Cuda Version v11.6.0**


## 1. Baseline Methods

Below command is for HCL method and CADE
```bash
python3 SSCL_main_HCL_CADE.py --ds=<dataset> --training_cutoff=<training_cutoff_value> --lr=<learning_rate> --wd=<weight_decay> --family_info=<true_or_false> --label_ratio=1 --uncertainity=<sample_selector>
```
where:
- `--ds`: dataset name (e.g., 'api_graph', 'androzoo', 'bodmas')
- `--lr`: learning rate (e.g., 0.001)
- `--wd`: weight decay (e.g., 0.0001)
- `--training_cutoff`: training cutoff value (e.g., 12, 5)
- `--family_info`: true or false (e.g., true)
- `--uncertainity`: sample selector (e.g., 'pseudo-loss', 'cade')
- `--label_ratio`: label ratio (e.g., 1,0.2)

## 2. Continual Learning Methods:
For MIR and CBRS
```bash
python3 SSCL_cl_implemented.py --ds=<dataset> --training_cutoff=<training_cutoff_value> --lr=<learning_rate> --wd=<weight_decay> --cl_method=<method>
```

where:
- `--ds`: dataset name (e.g., 'api_graph', 'androzoo', 'bodmas')
- `--lr`: learning rate (e.g., 0.001)
- `--wd`: weight decay (e.g., 0.0001)
- `--training_cutoff`: training cutoff value (e.g., 12, 5)
- `--cl_method`: Continual Learning method ( 'MIR', 'CBRS')
---
FOR EWC and AGEM
```bash
python3 SSCL_cl.py --ds=<dataset> --training_cutoff=<training_cutoff_value> --lr=<learning_rate> --wd=<weight_decay> --cl_method=<method>
```

where:
- `--ds`: dataset name (e.g., 'api_graph', 'androzoo', 'bodmas')
- `--lr`: learning rate (e.g., 0.001)
- `--wd`: weight decay (e.g., 0.0001)
- `--training_cutoff`: training cutoff value (e.g., 12, 5)
- `--cl_method`: Continual Learning method ( 'EWC', 'AGEM')


## Dataset:
Edit the Metadata.py file to change the dataset path.


## Limitations

This section outlines the limitations of the proposed method, considering both its design assumptions and observations from empirical studies.

### 1. Size of Buffer Memory
The method assumes a growing memory size to store partially labeled samples from all previously seen tasks. After an initial phase, the memory grows steadily based on the labeling budget. This assumption may not hold when deploying the method on memory-constrained hardware.

### 2. Security of the ML Model
It is important to recognize that the developed model could potentially be exploited by adversaries to generate more sophisticated malware. By publishing these findings, we aim to balance advancing defensive capabilities with the risk of misuse. However, issues such as adversarial attacks are beyond the scope of this work and are not addressed.

### 3. Labeling Assumption
The approach assumes that security analysts can provide correct labels for uncertain, unlabeled samples. This implies a high level of domain expertise and error-free labeling. Consequently, this work does not explicitly handle scenarios involving noisy or incorrect labels.



## Additional Details About Our Proposed Model

The model used in our experiments consists of two subnetworks: an **encoder** and a **classifier**.

- The **encoder** first reduces the dimensionality to 100, then progressively increases it to 250 and 500, and finally reduces it to 150 and 50. After each layer, we apply Batch Normalization, Dropout (with a ratio of 0.2), and a ReLU activation. Layer weights are initialized using the **Kaiming Uniform** method to improve convergence.

- The **classifier** is a simple linear layer that outputs two neurons—one for each class (benign and malware). The model outputs raw logits from the classifier.

### Hyperparameters

The key hyperparameters include:

- `b_m`: Percentage of batch size that contains memory samples  
- `bma`: Percentage of malware samples within `b_m`  
- `learning rate`, `weight decay`, and `maximum threshold (τ)`  
- Analyst-provided labels

We perform **grid search** to find optimal values. The search spaces are:

- `b_m`: 0.1 to 0.7  
- `bma`: 0.1 to 0.9  
- `learning rate`: {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}  
- `weight decay`: {1e-1 to 1e-9}  
- `τ_max`: 0.1 to 0.3

### Best Hyperparameters per Dataset

- **BODMAS**: `b_m = 0.5`, `bma = 0.8`, `lr = 1e-1`, `weight decay = 1e-9`, `τ_max = 0.09`  
- **AndroZoo**: `b_m = 0.3`, `bma = 0.4`, `lr = 1e-2`, `weight decay = 1e-1`, `τ_max = 0.05`  
- **APIGraph**: `b_m = 0.6`, `bma = 0.7`, `lr = 1e-1`, `weight decay = 1e-4`, `τ_max = 0.05`
