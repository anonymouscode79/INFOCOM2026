
# Experiment Configuration for Different Datasets and CL Methods

This Markdown outlines the commands used for running experiments on various datasets using different training configurations and continual learning (CL) methods.

## Our Method
### **Bodmas**
```bash 
python3 SSCL_main.py --ds=bodmas --training_cutoff=5 --lr=0.1 --wd=1e-9 --label_ratio=0.2 --b_m=0.5 --bma=0.8 --analyst_labels=100 --upper_thresh=0.09

```
### **Androzoo**

```bash
python3 SSCL_main.py --ds=androzoo --training_cutoff=12 --lr=0.01 --wd=0.1 --label_ratio=0.2 --b_m=0.3 --bma=0.4 --analyst_labels=100 --upper_thresh=0.05

```
### **ApiGraph**
```bash
python3 SSCL_main.py --ds=api_graph --training_cutoff=35 --lr=0.1 --wd=1e-4 --label_ratio=0.2  --b_m=0.6 --bma=0.7 --analyst_labels=100 --upper_thresh=0.05
```
---


###  HCL Method
### **Bodmas**
```bash
python3 SSCL_main_HCL_CADE.py --ds=bodmas --training_cutoff=5 --lr=0.0001 --wd=1e-07 --family_info=True --label_ratio=1 --uncertainity=pseudo-loss
```
### **ApiGraph**

```bash
python3 SSCL_main_HCL_CADE.py --ds=api_graph --training_cutoff=35 --lr=0.0001 --wd=1e-06 --family_info=True --label_ratio=1 --uncertainity=pseudo-loss
```

### **Androzoo**



```bash
python3 SSCL_main_HCL_CADE.py --ds=androzoo --training_cutoff=12 --lr=1e-05 --wd=1e-05 --family_info=True --label_ratio=1 --uncertainity=pseudo-loss
```
---
### CADE Method
### **ApiGraph**
```bash
python3 SSCL_main_HCL_CADE.py --ds=api_graph --training_cutoff=35 --lr=0.001 --wd=1e-07 --label_ratio=1 --uncertainity=cade
```
### **Androzoo**
```bash
python3 SSCL_main_HCL_CADE.py --ds=androzoo --training_cutoff=12 --lr=0.001 --wd=1e-06 --label_ratio=1 --uncertainity=cade
```
### **Bodmas**
```bash
python3 SSCL_main_HCL_CADE.py --ds=bodmas --training_cutoff=5 --lr=0.01 --wd=0.001 --label_ratio=1 --uncertainity=cade
```


---

## CL Methods

### **MIR**

#### ApiGraph
```bash
python3 SSCL_cl_implemented.py --ds=api_graph --training_cutoff=35 --lr=0.01 --wd=0.01 --cl_method=MIR
```

#### Androzoo
```bash
python3 SSCL_cl_implemented.py --ds=androzoo --training_cutoff=12 --lr=0.01 --wd=0.001 --cl_method=MIR
```

#### Bodmas
```bash
python3 SSCL_cl_implemented.py --ds=bodmas --training_cutoff=5 --lr=0.001 --wd=1e-6 --cl_method=MIR
```

---

### **CBRS**

#### ApiGraph
```bash
python3 SSCL_cl_implemented.py --ds=api_graph --training_cutoff=35 --lr=0.1 --wd=0.001 --cl_method=CBRS
```

#### Androzoo
```bash
python3 SSCL_cl_implemented.py --ds=androzoo --training_cutoff=12 --lr=0.1 --wd=0.001 --cl_method=CBRS
```

#### Bodmas
```bash
python3 SSCL_cl_implemented.py --ds=bodmas --training_cutoff=5 --lr=0.01 --wd=1e-6 --cl_method=CBRS
```

---

### **EWC**

#### ApiGraph
```bash
python3 SSCL_cl.py --ds=api_graph --training_cutoff=35 --lr=0.01 --wd=1e-5 --cl_method=EWC
```

#### Androzoo
```bash
python3 SSCL_cl.py --ds=androzoo --training_cutoff=12 --lr=0.01 --wd=1e-5 --cl_method=EWC
```

#### Bodmas
```bash
python3 SSCL_cl.py --ds=bodmas --training_cutoff=5 --lr=0.1 --wd=1e-7 --cl_method=EWC
```

---

### **AGEM**

#### ApiGraph
```bash
python3 SSCL_cl.py --ds=api_graph --training_cutoff=35 --lr=0.01 --wd=1e-6 --cl_method=AGEM
```

#### Androzoo
```bash
python3 SSCL_cl.py --ds=androzoo --training_cutoff=12 --lr=0.01 --wd=1e-7 --cl_method=AGEM
```

#### Bodmas
```bash
python3 SSCL_cl.py --ds=bodmas --training_cutoff=5 --lr=0.01 --wd=1e-7 --cl_method=AGEM
```
