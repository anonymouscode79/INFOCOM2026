import os
from avalanche.benchmarks.utils.data_attribute import torch
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc
from avalanche.evaluation.metrics import StreamConfusionMatrix, accuracy_metrics, forgetting_metrics, timing_metrics
from avalanche.logging import CSVLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import InteractiveLogger 
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, auc
import torch
from torchvision.datasets.celeba import csv
from torchvision.utils import save_image
import numpy as np



class CustomAttackMetric(PluginMetric[float]):
    """
    Custom plugin metric to compute F1 score, False Positive Rate (FPR), False Negative Rate (FNR),
    and Precision-Recall AUC separately for benign (class 0) and attack (class 1) samples.
    """

    def __init__(self,name):
        """
        Initialize the custom metric.
        """
        print("here")
        super().__init__()
        self.name = name
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def reset(self, **kwargs) -> None:
        """
        Reset the metric's internal state at the beginning of each epoch.
        """
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def result(self, **kwargs) -> dict:
        """
        Emit the result of all metrics for both benign and attack samples.
        """
        if len(self.y_true) == 0:
            return {
                'f1_benign': None,
                'f1_attack': None,
                'fpr_benign': None,
                'fpr_attack': None,
                'fnr_benign': None,
                'fnr_attack': None,
                'pr_auc_benign': None,
                'pr_auc_attack': None
            }
        print("here")
        y_true = torch.tensor(self.y_true)
        y_pred = torch.tensor(self.y_pred)
        # Calculate metrics for benign (class 0) and attack (class 1) samples
        
        print(np.unique(self.y_true,return_counts=True))
        # F1 Scores
        f1_benign = f1_score(y_true, y_pred,zero_division=0,pos_label=0)
        f1_attack= f1_score(y_true, y_pred,zero_division=0,pos_label=1)
        # Confusion Matrix (to get FP and FN separately for each class)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels=[0,1]).ravel()
        # FPR and FNR for benign and attack samples
        fpr_benign = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_benign = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr_attack = fn / (fn + tp) if (fn + tp) > 0 else 0  # FPR for attack (considering attack as positive)
        fnr_attack = fp / (fp + tn) if (fp + tn) > 0 else 0  # FNR for attack (considering attack as negative)

        # Precision-Recall AUC
        if len(self.y_prob) > 0:
            print(self.y_prob)
            precision_benign, recall_benign, _ = precision_recall_curve(y_true, torch.sub(1,torch.tensor(self.y_prob)),pos_label=0 )
            pr_auc_benign = auc(recall_benign, precision_benign)

            precision_attack, recall_attack, _ = precision_recall_curve(y_true , torch.tensor(self.y_prob),pos_label=1 )
            pr_auc_attack = auc(recall_attack, precision_attack)
        else:
            pr_auc_benign = None
            pr_auc_attack = None

        return {
            'f1_benign': f"{f1_benign:.3f}",
            'f1_attack': f"{f1_attack:.3f}",
            'fpr_benign':f"{ fpr_benign:.3f}",
            'fpr_attack':f"{fpr_attack:.3f}",
                'fnr_benign': f"{fnr_benign:.3f}",
        'fnr_attack': f"{fnr_attack:.3f}",
        'pr_auc_benign': f"{pr_auc_benign:.3f}",
                'pr_auc_attack': f"{pr_auc_attack:.3f}"
        }

    def after_eval_iteration(self, strategy) -> None:
        """
        Update the metric with the current predictions and true labels after each evaluation iteration.
        """
        self._update_metric(strategy)

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the metric with the current predictions and true labels after each training iteration.
        """
        self._update_metric(strategy)

    def _update_metric(self, strategy):
        """
        Helper function to update the metric's state with the current batch.
        """
        # Model outputs (logits or probabilities) and predictions

        outputs = strategy.mb_output
        # print(outputs)
        softmax  =  torch.nn.Softmax(dim=1)
        outputs = softmax(outputs) 
        preds = torch.argmax(outputs, dim=1)
        # Save true labels, predicted labels, and probabilities
        self.y_true.extend(strategy.mb_y.cpu().detach().numpy())  # True labels
        self.y_pred.extend(preds.cpu().detach().numpy())  # Predicted labels
        self.y_prob.extend(outputs[:, 1].cpu().detach().numpy())  # Probabilities for positive class (attack)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the metric at the beginning of each training epoch.
        """
        self.reset()

    def before_eval_exp(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the metric at the beginning of each evaluation epoch.
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result of the custom metrics after each training epoch.
        """
        return self._package_result(strategy)

    def after_eval_exp(self, strategy: 'PluggableStrategy'):
        """
        Emit the result of the custom metrics after each evaluation epoch.
        """
        return self._package_result(strategy)

    def _package_result(self, strategy: 'PluggableStrategy'):
        """
        Package the metric result for logging purposes and manually save to CSV during the evaluation phase.
        Only logs results after evaluation, not during training.
        """
        metric_values = self.result()

        # Prepare the CSV file path for this strategy
        csv_file = f"eval/logs/androzoo/{self.name}/metrics.csv"
        os.makedirs(f"eval/logs/androzoo/{self.name}", exist_ok=True)

        # If CSV file doesn't exist, create it with headers
        file_exists = os.path.isfile(csv_file)
        print("here")
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file doesn't exist
            if not file_exists:
                writer.writerow(['Evaluation_Experience', 'F1_Benign', 'F1_Attack', 'FPR_Benign', 'FPR_Attack', 
                                 'FNR_Benign', 'FNR_Attack', 'PR_AUC_Benign', 'PR_AUC_Attack'])

            # Write metric values for the current evaluation experience
            writer.writerow([strategy.clock.train_iterations, 
                             metric_values['f1_benign'], metric_values['f1_attack'], 
                             metric_values['fpr_benign'], metric_values['fpr_attack'], 
                             metric_values['fnr_benign'], metric_values['fnr_attack'], 
                             metric_values['pr_auc_benign'], metric_values['pr_auc_attack']])

        return []


    def __str__(self):
        """
        Name of the custom metric for logging purposes.
        """
        return "CustomAttackMetrics"## Evaluation Plugin

