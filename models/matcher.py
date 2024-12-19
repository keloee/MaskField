# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_query_x = x.shape[0]
    for query_idx in range(num_query_x):
        token = x[query_idx, :].unsqueeze(dim=0)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=1)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=0)  # Bx1x(t_x)x(t_y)


def feature_cosin_loss(inputs, targets,):
    return chunk_cosine_sim(inputs, targets)


def feature_l1_loss(inputs, targets):
    l1_dist = torch.abs(inputs[:, None, :] - targets[None, :, :]).mean(dim=-1)
    return l1_dist

def js_divergence_pairwise(P, Q):

    EPS = 1e-10  # 防止数值计算中的零或负值

    # 确保 P 和 Q 是有效的概率分布
    P = P / (P.sum(dim=1, keepdim=True) + EPS)  # 形状 (N, C)
    Q = Q / (Q.sum(dim=1, keepdim=True) + EPS)  # 形状 (M, C)

    # 为了进行广播，调整维度
    P = P.unsqueeze(1)  # 形状 (N, 1, C)
    Q = Q.unsqueeze(0)  # 形状 (1, M, C)

    # 计算平均分布 M，形状为 (N, M, C)
    M = 0.5 * (P + Q)

    # 计算 KL 散度的两个部分，形状为 (N, M)
    KL_PM = torch.sum(P * torch.log((P + EPS) / (M + EPS)), dim=2)
    KL_QM = torch.sum(Q * torch.log((Q + EPS) / (M + EPS)), dim=2)

    # 计算 JS 散度，形状为 (N, M)
    JS = 0.5 * (KL_PM + KL_QM)

    return JS

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 0.05, cost_feature=1, cost_relevance=1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_feature = cost_feature
        self.cost_relevance = cost_relevance
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_mask = targets[b]["masks"].to(out_mask)
            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)

            output_clip = outputs["pred_clip"][b]
            tgt_clip = targets[b]["clip"].to(out_mask)

            cost_clip = 1 - feature_cosin_loss(output_clip, tgt_clip) + feature_l1_loss(output_clip, tgt_clip)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_feature * cost_clip
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):

        return self.memory_efficient_forward(outputs, targets)
