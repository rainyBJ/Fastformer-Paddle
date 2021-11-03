import paddle
import numpy as np

# accuracy & macro-f
def acc(y_true, y_hat, eval=False):
    if eval:
        y_hat = y_hat
    else:
        y_hat = paddle.argmax(y_hat, axis=-1)
    tot = y_true.shape[0]
    # accuracy
    hit = paddle.to_tensor(np.sum(y_true.numpy() == y_hat.numpy()))
    f = 0
    if eval:
        # F-macro
        eps = 1e-8
        TP = {}
        FN = {}
        FP = {}
        TN = {}
        precision_dict = {}
        recall_dict = {}
        F = {}
        for cls in range(5):
            TP[cls] = 0
            FN[cls] = 0
            FP[cls] = 0
            TN[cls] = 0
            precision_dict[cls] = 0
            recall_dict[cls] = 0
            F[cls] = 0
        for i in range(len(y_true)):
            predict = y_hat[i]
            true = y_true[i]
            for cls in range(5):
                if true==cls and predict==cls:
                    TP[cls] = TP[cls] + 1
                elif true==cls and predict!=cls:
                    FN[cls] = FN[cls] + 1
                elif true!=cls and predict==cls:
                    FP[cls] = FP[cls] + 1
                else:
                    TN[cls] = TN[cls] + 1

        for cls in range(5):
            precision_dict[cls] = TP[cls] / (TP[cls] + FP[cls] + eps)
            recall_dict[cls] = TP[cls] / (TP[cls] + FN[cls] + eps)
        precision = 0
        recall = 0
        for cls in range(5):
            precision = precision + precision_dict[cls]
            recall = recall + recall_dict[cls]
        precision = precision/5
        recall = recall/5
        f = 2*precision*recall/(precision+recall)

    return  float(hit) * 1.0 / tot, f