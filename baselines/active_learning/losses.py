import torch
import torch.nn as nn  
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(TripletLoss, self).__init__()
        self.reduce = reduce

    def forward(self, features, labels = None, margin = 10.0,
                weight = None, split = None):
        """
        Triplet loss for model.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        batch_size = features.shape[0]
        # if batch_size % 3 != 0:
        #     print(f"Warning: Batch size {batch_size} is not divisible by 3. Adjusting to the nearest multiple of 3.")
        #     batch_size = (batch_size // 3) * 3
        pass_size = batch_size // 3
        # print(pass_size)
        """
        three shares of pass_size
        1) training data sample
        2) positive samples
        3) negative samples
        """
        batch_size = features.shape[0]
        device = features.device
        # Split batch into left and right pairs
        batch_split = batch_size // 2
        left_p = torch.arange(0, batch_split, device=device)
        right_p = torch.arange(batch_split, batch_split * 2, device=device)

        # Compute pairwise distances
        dist = torch.linalg.norm(features[left_p] - features[right_p], dim=1)
        
        # Check if pairs are from the same class
        is_same = (labels[left_p] == labels[right_p]).float().squeeze()
        # Contrastive loss
        positive_loss = is_same * dist
        negative_loss = (1 - is_same) * torch.nn.functional.relu(margin - dist)
        
        # Final loss
        loss = torch.mean(positive_loss + negative_loss)
        return loss

class TripletMSELoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        super(TripletMSELoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce

    def forward(self, cae_lambda,
            x, x_prime,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            cae_lambda: scale the CAE loss
            x: input to the Autoencoder
            x_prime: decoded x' from Autoencoder
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(features, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_prime, reduction = self.reduce)
        
        loss = cae_lambda * supcon_loss + mse_loss
        
        del Triplet
        torch.cuda.empty_cache()

        return loss, supcon_loss, mse_loss
class HiDistanceLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce='mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, features, binary_cat_labels, labels = None, margin = 10.0,
                weight = None, split = None,family_info=False,device=None):
        """
        Pair distance loss.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            binary_cat_labels: one-hot binary labels.
            labels: ground truth of shape [bsz].
            margin: margin for dissimilar distance.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore entries for these
        Returns:
            A loss scalar.
        """


        if labels == None:
            raise ValueError('Need to define labels in DistanceLoss')

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # similar masks
        # mask_{i,j}=1 if sample j has the same class as sample i.

        binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        # mask: both malware, or both benign
        binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        # multi_mask: same malware family, or benign
        multi_mask = torch.eq(labels, labels.T).float().to(device)
        # malware but n fft the same family. does not have benign.
        other_mal_mask = binary_mask - multi_mask
        # both benign samples
        ben_labels = torch.logical_not(binary_labels).float().to(device)
        same_ben_mask = torch.matmul(ben_labels, ben_labels.T)
        # same malware family mask
        same_mal_fam_mask = multi_mask - same_ben_mask
        
        # logging.debug("=== new batch ===")
        # pseudo loss
        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_mal_fam_mask
            same_mal_fam_mask = tmp
            # debug
            # split_index = torch.nonzero(split, as_tuple=True)[0]
            # logging.debug(f'split_index, {split_index}')
        # logging.debug(f'binary_labels {binary_labels}')
        # logging.debug(f'binary_mask {binary_mask}')
        # logging.debug(f'labels {labels}')
        # logging.debug(f'multi_mask {multi_mask}')
        # logging.debug(f'other_mal_mask = binary_mask - multi_mask {other_mal_mask}')
        # logging.debug(f'ben_labels {ben_labels}')
        # logging.debug(f'same_ben_mask {same_ben_mask}')
        # logging.debug(f'same_mal_fam_mask = multi_mask - same_ben_mask {same_mal_fam_mask}')
        
        # dissimilar mask. malware vs benign binary labels
        binary_negate_mask = torch.logical_not(binary_mask).float().to(device)
        # multi_negate_mask = torch.logical_not(multi_mask).float().to(device)

        # mask-out self-contrast cases
        diag_mask = (torch.logical_not(torch.eye(batch_size)).float()).to(device)
        # similar mask
        diag_mask = diag_mask.to(device)
        binary_mask = binary_mask * diag_mask
        multi_mask = multi_mask * diag_mask
        other_mal_mask = other_mal_mask * diag_mask
        same_ben_mask = same_ben_mask * diag_mask
        same_mal_fam_mask = same_mal_fam_mask * diag_mask

        # adjust the masks based on test indices
        if split is not None:
            split_index = torch.nonzero(split, as_tuple=True)[0]
            # instance-level loss, paired with training samples, pseudo loss
            # logging.debug(f'split_index, {split_index}')
            binary_negate_mask[:, split_index] = 0
            # multi_negate_mask[:, split_index] = 0
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0

        # reference: https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/functional/pairwise/euclidean.py
        # not taking the sqrt for numerical stability
        x = features
        y = features
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1).T
        distance_matrix = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        distance_matrix = torch.maximum(torch.tensor(1e-10), distance_matrix)
        # logging.debug(f'distance_matrix {distance_matrix}')
        # #logging.debug(f'torch.isnan(distance_matrix).any() {torch.isnan(distance_matrix).any()}')
        # logging.debug(f'same_ben_mask {same_ben_mask}')
        # logging.debug(f'other_mal_mask {other_mal_mask}')
        # logging.debug(f'same_mal_fam_mask {same_mal_fam_mask}')
        # logging.debug(f'binary_negate_mask {binary_negate_mask}')
        
        # four types of pairs
        # 1. ben, ben. same_ben_mask
        # 2. mal, mal from different families. other_mal_mask
        # 3. mal, mal from same families. same_mal_fam_mask
        # 4. ben, mal. binary_negate_mask
        # default is to compute mean for these values per sample
        if family_info:
            if self.sample_reduce == 'mean' or self.sample_reduce == None:
                if weight == None:
                    sum_same_ben = torch.maximum(
                                        torch.sum(same_ben_mask * distance_matrix, dim=1) - \
                                                same_ben_mask.sum(1) * torch.tensor(margin),
                                        torch.tensor(0))
                    sum_other_mal = torch.maximum(
                                        torch.sum(other_mal_mask * distance_matrix, dim=1) - \
                                                other_mal_mask.sum(1) * torch.tensor(margin),
                                        torch.tensor(0))
                    sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix, dim=1)
                    sum_bin_neg = torch.maximum(
                                        binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                                torch.sum(binary_negate_mask * distance_matrix,
                                                        dim=1),
                                        torch.tensor(0))
                    # logging.debug(f'sum_same_ben {sum_same_ben}, same_ben_mask.sum(1) {same_ben_mask.sum(1)}')
                    # logging.debug(f'sum_other_mal {sum_other_mal}, other_mal_mask.sum(1) {other_mal_mask.sum(1)}')
                    # logging.debug(f'sum_same_mal_fam {sum_same_mal_fam}, same_mal_fam_mask.sum(1) {same_mal_fam_mask.sum(1)}')
                    # logging.debug(f'sum_bin_neg {sum_bin_neg}, binary_negate_mask.sum(1) {binary_negate_mask.sum(1)}')
                # weighted loss
                else:
                    weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                    sum_same_ben = torch.maximum(
                                        torch.sum(same_ben_mask * distance_matrix * weight_matrix, dim=1) - \
                                                same_ben_mask.sum(1) * torch.tensor(margin),
                                        torch.tensor(0))
                    sum_other_mal = torch.maximum(
                                        torch.sum(other_mal_mask * distance_matrix * weight_matrix, dim=1) - \
                                                other_mal_mask.sum(1) * torch.tensor(margin),
                                        torch.tensor(0))
                    sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix * weight_matrix, dim=1)
                    weight_prime = torch.div(1.0, weight)
                    weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                    sum_bin_neg = torch.maximum(
                                        binary_negate_mask.sum(1) * torch.tensor(2 * margin) - \
                                                torch.sum(binary_negate_mask * distance_matrix * weight_matrix_prime,
                                                        dim=1),
                                        torch.tensor(0))
                loss = sum_same_ben / torch.maximum(same_ben_mask.sum(1), torch.tensor(1)) + \
                        sum_other_mal / torch.maximum(other_mal_mask.sum(1), torch.tensor(1)) + \
                        sum_same_mal_fam / torch.maximum(same_mal_fam_mask.sum(1), torch.tensor(1)) + \
                        sum_bin_neg / torch.maximum(binary_negate_mask.sum(1), torch.tensor(1))
            elif self.sample_reduce == 'max':
                max_same_ben = torch.maximum(
                                    torch.amax(same_ben_mask * distance_matrix, 1) - \
                                            torch.tensor(margin),
                                    torch.tensor(0))
                max_other_mal = torch.maximum(
                                    torch.amax(other_mal_mask * distance_matrix, 1) - \


                                            torch.tensor(margin),
                                torch.tensor(0))
                max_same_mal_fam = torch.amax(same_mal_fam_mask * distance_matrix, 1)
                max_bin_neg = torch.maximum(
                                    torch.tensor(2 * margin) - \
                                            torch.amin(binary_negate_mask * distance_matrix, 1),
                                    torch.tensor(0))
                loss = max_same_ben + max_other_mal + max_same_mal_fam + max_bin_neg
            else:
                raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')
        else:
            # print("no Family Info")
            separate_ben_mal = torch.maximum(
            binary_negate_mask.sum(1) * torch.tensor(margin) - \
            torch.sum(binary_negate_mask * distance_matrix, dim=1),
            torch.tensor(0)
            )
            cluster_within_class = torch.maximum(
                        torch.sum(binary_mask * distance_matrix, dim=1) - \
                        binary_mask.sum(1) * torch.tensor(margin),
                        torch.tensor(0)
                    )

            loss = torch.mean(separate_ben_mal + cluster_within_class)
        if self.reduce == 'mean':
            loss = loss.mean()
        return loss

class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None,
            family_info=True,
            device=None
            ):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        supcon_loss = Dist(features, y_bin_batch, labels = labels, margin = margin, weight = None, split = split,family_info=family_info,device=device)
        # print(f"y_bin_prob {y_bin_pred} y_bin_batch : {y_bin_batch}")
        if y_bin_pred.ndim == 1:
            y_bin_pred = y_bin_pred.unsqueeze(0) 
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1],
                                                        reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()

        loss = supcon_loss + xent_lambda * xent_bin_loss
        del Dist
        torch.cuda.empty_cache()
        return loss, supcon_loss , xent_bin_loss
 