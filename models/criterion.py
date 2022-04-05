import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err

class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        # print("pred",pred.size())
        # print("target",target.size())
        return torch.mean(weight * torch.abs(pred - target))

class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))

class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

class CELoss(BaseLoss):
    def __init__(self):
        super(CELoss, self).__init__()

    def _forward(self, pred, target, weight=None):
        return F.cross_entropy(pred, target)

class TripletLossCosine(BaseLoss):
    """
    Triplet loss with cosine distance
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        distance_positive = 1 - F.cosine_similarity(anchor, positive)
        distance_negative= 1 - F.cosine_similarity(anchor, negative)
        losses = F.relu((distance_positive - distance_negative) + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletLoss(BaseLoss):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _forward(self, anchor, positive, negative, size_average=True):
        #print("Input:")
        #print(anchor.shape)
        #print(positive.shape)
        #print(negative.mean())
        distance_positive = torch.abs((anchor - positive).sum(1)) #.pow(.5)
        distance_negative = torch.abs((anchor - negative).sum(1)) #.pow(.5)
        #print("Distance:")
        #print(distance_positive.mean())
        #print(distance_negative.mean())
        #print((distance_positive - distance_negative).mean())
        losses = F.relu((distance_positive - distance_negative) + self.margin)
        #print("Loss:")
        #print(losses.mean())
        #print("Ket thuc.")
        return losses.mean() if size_average else losses.sum()

