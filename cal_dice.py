from typing import Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.data import to_categorical
from torchmetrics.utilities.distributed import reduce


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    class_index: int,
    argmax_dim: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates the number of true positive, false positive, true negative and false negative for a specific
    class.

    Args:
        preds: prediction tensor
        target: target tensor
        class_index: class to calculate over
        argmax_dim: if pred is a tensor of probabilities, this indicates the
            axis the argmax transformation will be applied over

    Return:
        True Positive, False Positive, True Negative, False Negative, Support

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> y = torch.tensor([0, 2, 3])
        >>> tp, fp, tn, fn, sup = _stat_scores(x, y, class_index=1)
        >>> tp, fp, tn, fn, sup
        (tensor(0), tensor(1), tensor(2), tensor(0), tensor(0))
    """
    if preds.ndim == target.ndim + 1:
        preds = to_categorical(preds, argmax_dim=argmax_dim)

    tp = ((preds == class_index) * (target == class_index)).to(torch.long).sum()
    fp = ((preds == class_index) * (target != class_index)).to(torch.long).sum()
    tn = ((preds != class_index) * (target != class_index)).to(torch.long).sum()
    fn = ((preds != class_index) * (target == class_index)).to(torch.long).sum()
    sup = (target == class_index).to(torch.long).sum()

    return tp, fp, tn, fn, sup


def get_res(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    return_all: bool = False
) -> Tensor:
    """Compute dice score from prediction scores.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor containing dice score
    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
    scores_dict = {}
    preds_list = []
    preds_output = to_categorical(preds, argmax_dim=1)
    for i in range(bg_inv, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg_inv] += no_fg_score
            continue
        # TODO: rewrite to use general `stat_scores`
        tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score

        scores[i - bg_inv] += score_cls
        scores_dict[i] = round(float(score_cls), 4)

        # # TODO: rewrite to use general `stat_scores`
        # preds_output = to_categorical(preds, argmax_dim=1)
        # preds_output = (preds_output == i).float()
        # preds_list.append(preds_output)
    # pred_all = torch.concatenate(preds_list, dim=0)
    return reduce(scores, reduction=reduction), preds_output, scores_dict


# def get_res(
#     preds: Tensor,
#     target: Tensor,
#     bg: bool = False,
#     nan_score: float = 0.0,
#     no_fg_score: float = 0.0,
#     reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
#     return_all: bool = False
# ) -> Tensor:
#     """Compute dice score from prediction scores.
#
#     Args:
#         preds: estimated probabilities
#         target: ground-truth labels
#         bg: whether to also compute dice for the background
#         nan_score: score to return, if a NaN occurs during computation
#         no_fg_score: score to return, if no foreground pixel was found in target
#         reduction: a method to reduce metric score over labels.
#
#             - ``'elementwise_mean'``: takes the mean (default)
#             - ``'sum'``: takes the sum
#             - ``'none'`` or ``None``: no reduction will be applied
#
#     Return:
#         Tensor containing dice score
#     """
#     num_classes = preds.shape[1]
#     bg_inv = 1 - int(bg)
#     scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
#     scores_dict = {}
#     preds_list = []
#     for i in range(bg_inv, num_classes):
#         if not (target == i).any():
#             # no foreground class
#             scores[i - bg_inv] += no_fg_score
#             continue
#         # TODO: rewrite to use general `stat_scores`
#         tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
#         denom = (2 * tp + fp + fn).to(torch.float)
#         # nan result
#         score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
#
#         scores[i - bg_inv] += score_cls
#         scores_dict[i] = round(float(score_cls), 4)
#
#         # TODO: rewrite to use general `stat_scores`
#         preds_output = to_categorical(preds, argmax_dim=1)
#         preds_list.append(preds_output)
#     pred_all = torch.concatenate(preds_list, dim=0)
#     return pred_all, scores_dict
#     #     tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
#     #     denom = (2 * tp + fp + fn).to(torch.float)
#     #     # nan result
#     #     score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score
#     #
#     #     scores[i - bg_inv] += score_cls
#     #     scores_dict[i] = round(float(score_cls), 4)
#     #     # scores_list.append(round(float(score_cls), 4))
#     # if return_all:
#     #     return reduce(scores, reduction=reduction), scores_dict
#     # else:
#     #     return reduce(scores, reduction=reduction)


def dice_score(
    preds: Tensor,
    target: Tensor,
    bg: bool = False,
    nan_score: float = 0.0,
    no_fg_score: float = 0.0,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        return_all: bool = False
) -> Tensor:
    """Compute dice score from prediction scores.

    Args:
        preds: estimated probabilities
        target: ground-truth labels
        bg: whether to also compute dice for the background
        nan_score: score to return, if a NaN occurs during computation
        no_fg_score: score to return, if no foreground pixel was found in target
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Return:
        Tensor containing dice score
    """
    num_classes = preds.shape[1]
    bg_inv = 1 - int(bg)
    scores = torch.zeros(num_classes - bg_inv, device=preds.device, dtype=torch.float32)
    scores_dict = {}
    for i in range(bg_inv, num_classes):
        if not (target == i).any():
            # no foreground class
            scores[i - bg_inv] += no_fg_score
            continue

        # TODO: rewrite to use general `stat_scores`
        tp, fp, _, fn, _ = _stat_scores(preds=preds, target=target, class_index=i)
        denom = (2 * tp + fp + fn).to(torch.float)
        # nan result
        score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else nan_score

        scores[i - bg_inv] += score_cls
        scores_dict[i] = round(float(score_cls), 4)
        # scores_list.append(round(float(score_cls), 4))
    if return_all:
        return reduce(scores, reduction=reduction), scores_dict
    else:
        return reduce(scores, reduction=reduction)