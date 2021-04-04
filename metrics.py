import numpy as np

class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return self.avg

    @property
    def _get_sum(self):
        return self.sum

class Evaluator(object):
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.confusion_matrix = np.zeros((self.nb_classes,) * 2) # matrix of 17*17

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum()+1e-10)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1)+1e-10)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix) + 1e-10
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix)+1e-10)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)+1e-10
        )
        # weighted sum
        FWIoU = (freq[freq>0] * iu[freq>0]).sum()
        return FWIoU
    def Kappa(self):
        matrix_sum = np.sum(self.confusion_matrix) # matrix_sum
        true_positive = np.diag(self.confusion_matrix).sum()  # TP
        p0 = true_positive/matrix_sum
        temp = 0
        for i in range(self.confusion_matrix.shape[0]):
            temp=temp+self.confusion_matrix[i,:].sum()*self.confusion_matrix[:,i].sum()
        p1 = temp/(matrix_sum*matrix_sum)
        kappa = (p0-p1)/(1-p1)
        return kappa
    def Precision(self):
        true_positive = np.diag(self.confusion_matrix) # TP
        predicted_condition_positive = np.sum(self.confusion_matrix, axis=0) # TP+FP
        P_per_class = true_positive / (predicted_condition_positive+1e-10) # TP / (TP+FP)
        P = np.nanmean(P_per_class)
        return P, P_per_class

    def Recall(self):
        true_positive = np.diag(self.confusion_matrix) # TP
        condition_positive = np.sum(self.confusion_matrix, axis=1) # TP+FN
        R_per_class = true_positive / (condition_positive+1e-10) # TP/P
        R = np.nanmean(R_per_class)
        return R, R_per_class

    def Fx_Score(self, x=1):
        _, R_per_class = self.Recall()
        _, P_per_class = self.Precision()
        F = ((x * x + 1) * P_per_class * R_per_class) / (x * x * P_per_class + R_per_class + 1e-10)
        F = np.nanmean(F)
        return F

    def generate_confusion_matrix(self, gt_images, pred_images):
        # 0-dim: means predicted condition
        # 1-dim: means True condition
        '''

        :param gt_images: label =>1 * 145 *145
        :param pred_images: the max value index of 9 channels =》 1*145*145
        :return:
        '''
        mask = (gt_images >= 0) & (gt_images < self.nb_classes)
        label = self.nb_classes * gt_images[mask].astype('int') + pred_images[mask]
        count = np.bincount(label, minlength=self.nb_classes ** 2)
        confusion_matrix = count.reshape(self.nb_classes, self.nb_classes)
        return confusion_matrix

    def add_batch(self, gt_images, logits):
        '''

        :param gt_images: label image
        :param logits: predict image
        :return:
        '''
        # here receive cuda.tensor

        gt_images = gt_images.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        pred_images = np.argmax(logits, axis=1) # 找到每个像素点的3个通道数中，数值最大的那个通道数。

        self.confusion_matrix += self.generate_confusion_matrix(gt_images, pred_images)
    def get_class(self, logits):
        '''

        :param gt_images: label image
        :param logits: predict image
        :return:
        '''
        # here receive cuda.tensor


        logits = logits.detach().cpu().numpy()
        pred_images = np.argmax(logits, axis=1) # 找到每个像素点的3个通道数中，数值最大的那个通道数。
        return pred_images
    def reset_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.nb_classes,) * 2)


