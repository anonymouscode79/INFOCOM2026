from avalanche.benchmarks.utils.data import np
from avalanche.logging import InteractiveLogger
from avalanche.training import EWC, AGEM, MIR, LwF, ICaRL
from avalanche.training.plugins import EvaluationPlugin, evaluation
import torch
from avalanche.training.plugins import EarlyStoppingPlugin
import torch.nn as nn
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from enum import Enum

from baselines.continual_learning.evaluation import CustomAttackMetric, PRAUCMetric
from baselines.continual_learning.icarl_fc import ICARL_FC


## DEFINITION OF SOME ENUMS
class MODELNAME(Enum):
    EWC = 1
    AGEM = 2
    MIR = 3
    LWF = 4
    ICARC = 5

class Model_class():
    def __init__(self, model:nn.Module,model_name: MODELNAME, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, train_mb_size: int, eval_mb_size: int, device: torch.device,epochs:int):
        self.model_name = model_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size
        self.device = device
        self.model = model 
        self.model.to(device)
        self.eval_plugin  = eval 
        self.epochs = epochs
    def load_model(self):
        # Load the model based on the chosen continual learning strategy
        if self.model_name == MODELNAME.EWC:
            # Create an EWC model with the given parameters
            model = EWC(
                evaluator=EvaluationPlugin(
                    PRAUCMetric(),
                        loggers=[InteractiveLogger()],
                        strict_checks=False,
                    ),
                    plugins=[
                        EarlyStoppingPlugin(
                            metric_name =  "pr_auc_attack",
                            verbose=True,
                            patience=3,  # Stop training if loss doesn't improve for 3 validation checks,
                            val_stream_name="test_stream"
                        )
                    ], 
                model = self.model,
                optimizer=self.optimizer,
                train_epochs = self.epochs,
                criterion=self.criterion,
                ewc_lambda=0.4,  
                train_mb_size=self.train_mb_size,
                device=self.device,
                eval_every=1,
            )
        elif self.model_name == MODELNAME.AGEM:
            # Create an AGEM model with the given parameters
            model = AGEM( 
                evaluator=EvaluationPlugin(
                    PRAUCMetric(),
                        loggers=[InteractiveLogger()],
                        strict_checks=False,
                    ),
                    plugins=[
                        EarlyStoppingPlugin(
                            metric_name =  "pr_auc_attack",
                            verbose=True,
                            patience=3,  # Stop training if loss doesn't improve for 3 validation checks,
                            val_stream_name="test_stream"
                        )
                    ],               
                model=self.model,
                optimizer=self.optimizer,
                train_epochs = self.epochs,
                criterion=self.criterion,
                patterns_per_exp=256, 
                train_mb_size=self.train_mb_size,
                device=self.device,
                eval_every=1,
            )
        elif self.model_name == MODELNAME.MIR:
            # Create a MIR model with the given parameters
            model = MIR(
                 evaluator=EvaluationPlugin(
                    PRAUCMetric(),
                        loggers=[InteractiveLogger()],
                        strict_checks=False,
                    ),
                    plugins=[
                        EarlyStoppingPlugin(
                            metric_name =  "pr_auc_attack",
                            verbose=True,
                            patience=3,  # Stop training if loss doesn't improve for 3 validation checks,
                            val_stream_name="test_stream"
                        )
                    ],             
                subsample=100,
                model=self.model,
                optimizer=self.optimizer,
                train_epochs = self.epochs,
                criterion=self.criterion,
                mem_size=1000, 
                train_mb_size=self.train_mb_size,
                device=self.device
            )
        elif self.model_name == MODELNAME.LWF:
            # Create a LwF model with the given parameters
            model = LwF(
                evaluator=EvaluationPlugin(
                    PRAUCMetric(),
                        loggers=[InteractiveLogger()],
                        strict_checks=False,
                    ),
                    plugins=[
                        EarlyStoppingPlugin(
                            metric_name =  "pr_auc_attack",
                            verbose=True,
                            patience=3,  # Stop training if loss doesn't improve for 3 validation checks,
                            val_stream_name="test_stream"
                        )
                    ],              
                optimizer=self.optimizer,
                train_epochs = self.epochs,
                criterion=self.criterion,
                alpha=1.0,  
                temperature=2.0,
                train_mb_size=self.train_mb_size,
                device=self.device
            )
        elif self.model_name == MODELNAME.ICARC:
            # Create an ICaRL model with the given parameters
            model = ICaRL(
            feature_extractor = self.model,
            classifier=ICARL_FC(),
            optimizer= self.optimizer,
           memory_size=256,
           fixed_memory=False,
           buffer_transform=None,
           train_mb_size=self.train_mb_size,
           eval_mb_size=128,
           evaluator=EvaluationPlugin(
                    PRAUCMetric(),
                        loggers=[InteractiveLogger()],
                        strict_checks=False,
                    ),
                    plugins=[
                        EarlyStoppingPlugin(
                            metric_name =  "pr_auc_attack",
                            verbose=True,
                            patience=3,  # Stop training if loss doesn't improve for 3 validation checks,
                            val_stream_name="test_stream"
                        )
                    ],            
           device=self.device,
       )           
        else:
            raise ValueError("Unknown model name")

        return model

## Some Random Model to test the baseline strategies


class MalwareDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MalwareDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)
        self.fc4 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc5 = nn.Linear(hidden_size*2, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.softmax= nn.Softmax(dim=1)    
    def forward(self, x):
        out = self.fc1(x.float())
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out =self.fc5(out)
        out = self.relu(out)
        out=self.fc6(out)
        return out
from sklearn.metrics import precision_recall_curve, auc

def pr_auc_attack_metric(output, target):
    """
    Custom metric to compute PR-AUC for the 'Attack' class.
    """
    # Extract probabilities for 'Attack' class (usually class 1 in binary classification)
    prob_attack = output[:, 1].detach().cpu().numpy()  # Assuming binary classification with Attack as class 1
    true_labels = target.detach().cpu().numpy()

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, prob_attack)

    # Calculate PR-AUC
    pr_auc = auc(recall, precision)
    return pr_auc