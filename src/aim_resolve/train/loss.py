from torch.nn import BCEWithLogitsLoss



class BCELoss(BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='sum', pos_weight=None):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def __call__(self, y_pred, y, **kwargs):
        return self.forward(y_pred, y)
