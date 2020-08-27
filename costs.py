import logging
from keras.layers import Lambda

import numpy as np
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error
from keras.activations import relu
log = logging.getLogger()
from utils.image_utils import image_show # harric modified
from keras.layers import concatenate # harric added to incorporate the segementation correction when segmentation_option=1
from keras.backend import expand_dims # because mi only in my
eps = 1e-12 # harric added to engage the smooth factor
from keras.layers import Add, Multiply, Lambda, Subtract

def calculate_false_negative(y_true, y_pred, model_type):
    mask_num = y_true.shape[-1]
    if model_type=='unet':
        y_pred_new = y_pred[0]
    elif model_type == 'sdnet':
        y_pred_new = y_pred
    y_pred_new = y_pred_new[..., 0: mask_num]
    y_pred_new = np.round(y_pred_new)

    false_negative = (np.sum(np.logical_and(y_pred_new==0, y_true==1)) + eps) / (np.sum(y_true) + eps)
    false_negative_sep = []
    for ii in range(mask_num):
        y_true_sep = y_true[:, :, :, ii]
        y_pred_sep = y_pred_new[:, :, :, ii]
        this_false_negative = (np.sum(np.logical_and(y_pred_sep==0, y_true_sep==1)) + eps) / (np.sum(y_true_sep) + eps)
        false_negative_sep.append(this_false_negative)
    return false_negative, false_negative_sep

def dice(y_true, y_pred, model_type):
    mask_num = y_true.shape[-1]

    if model_type=='unet':
        y_pred_new = y_pred[0]
    elif model_type == 'sdnet':
        y_pred_new = y_pred
    y_pred_new = y_pred_new[..., 0: mask_num]

    # harric modified
    y_pred_new = np.round(y_pred_new)

    # Symbolically compute the intersection
    y_int = y_true * y_pred_new
    dice_total = np.mean((2 * np.sum(y_int, axis=(1, 2, 3)) + eps) # harric deleted the smooth in the norminator
                         / (np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred_new, axis=(1, 2, 3)) + eps))


    # harric added to seperatively calculate dice scores
    dice_sep = []
    for ii in range(mask_num):
        y_true_sep = np.expand_dims(y_true[:,:,:,ii],axis=-1)
        y_pred_sep = np.expand_dims(y_pred_new[:, :, :, ii], axis=-1)
        # y_pred_sep = np.round(y_pred_sep) # Cast the prediction to binary 0 or 1
        this_y_int = y_true_sep * y_pred_sep
        #this_dice_sep = np.mean((2 * np.sum(this_y_int, axis=(1, 2, 3)) + smooth) # harric deleted the smooth in the norminator
        this_dice_sep = np.mean((2 * np.sum(this_y_int, axis=(1, 2, 3)) + eps)
                                / (np.sum(y_true_sep, axis=(1, 2, 3)) + np.sum(y_pred_sep, axis=(1, 2, 3)) + eps))
        dice_sep.append(this_dice_sep)

    return dice_total, dice_sep


def dice_coef(y_true, y_pred):
    '''
    DICE Loss.
    :param y_true: a tensor of ground truth data
    :param y_pred: a tensor of predicted data
    '''
    # Symbolically compute the intersection
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3))
    return K.mean((2 * intersection + eps) / (union + eps), axis=0)


# Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# harric added to enable different error for different segmentation
def make_mse_loss_func(restrict_chn):
    log.debug('Making MSE loss function for the first %d channels' % restrict_chn)
    def mse_func(y_true,y_pred):
        return mean_squared_error(y_true,y_pred)
    return mse_func
# def make_mae_loss_func():
#     log.debug('Making MAE loss function')
#     def mae_func(y_true,y_pred):
#         return mean_absolute_error(y_true,y_pred)
#     return mae_func

def make_weighted_mae_loss_func(args):
    y1,y2, pathology_label, rate, name = args
    mae_org = Lambda(lambda x: K.abs(x))(Subtract()([y1,y2]))
    mae_combined_list = []
    for ii in range(int(pathology_label.shape[-1])):
        current_pathology_label = Lambda(lambda x:x[:,:,:,ii:ii+1])(pathology_label)
        mae_pathology = Lambda(lambda x: K.abs(x))(Subtract()([Multiply()([y1, current_pathology_label]),
                                                               Multiply()([y2, current_pathology_label])]))
        mae_combined = Add()([mae_org, Lambda(lambda x: x*rate)(mae_pathology)])
        mae_combined_list.append(mae_combined)
    mae_combined = Add()(mae_combined_list)
    return Lambda(lambda x: K.expand_dims(K.mean(x,axis=(1,2,3)),axis=-1), name='Reconstructor'+name)(mae_combined)


def make_mse_loss_func_distributed(restrict_chn, infarction_weight=-1, loss_type='-1'):
    log.debug('Making MSE & MAE combinde loss function for the first %d channels' % restrict_chn)
    def mae_mse_combined_loss(y_true,y_pred):

        y_true_myo = relu(y_true - 1.0 / 3.0) + 1.0 / 3.0
        y_pred_myo = relu(y_pred - 1.0 / 3.0) + 1.0 / 3.0
        y_true_myi = relu(y_true - 2.0 / 3.0) + 2.0 / 3.0
        y_pred_myi = relu(y_pred - 2.0 / 3.0) + 2.0 / 3.0

        loss_types = loss_type.split('+')
        if loss_types[0]=='mse':
            loss1 = mean_squared_error(y_true_myo,y_pred_myo)
        elif loss_types[0]=='mae':
            loss1 = mean_absolute_error(y_true_myo, y_pred_myo)

        if loss_types[1] == 'mse':
            loss2 = mean_squared_error(y_true_myi,y_pred_myi)
        elif loss_types[1] == 'mae':
            loss2 = mean_absolute_error(y_true_myi,y_pred_myi)
        return (loss1 + loss2*infarction_weight) / restrict_chn

    return mae_mse_combined_loss


def weighted_softmax_cross_entropy(restrict_chn, calculate_true_background=True):
    """
    Define weighted cross-entropy function for classification tasks. Applies softmax on y_pred.
    :param y_pred: tensor [None, width, height, n_classes]
    :param y_true: tensor [None, width, height, n_classes]
    :param eps: (float) small value to avoid division by zero
    :param num_classes: (int) number of classes
    :return:
    """
    if calculate_true_background:
        restrict_chn = restrict_chn + 1
    log.debug('Making Weighted Cross Entropy loss function for the first %d channels' % restrict_chn)


    def cross_entropy(y_true,y_pred):
        if calculate_true_background:
            true_background = K.tf.ones_like(K.tf.expand_dims(y_true[...,0],axis=-1))
            for channel in range(restrict_chn-1):
                true_background = true_background -  K.tf.expand_dims(y_true[...,channel],axis=-1)
            y_true_valid= K.tf.concat([y_true,true_background],axis=-1)
        else:
            y_true_valid = y_true

        y_pred_valid = y_pred

        n = [K.tf.reduce_sum(K.tf.cast(y_true_valid[..., c], K.tf.float32)) for c in range(restrict_chn)]
        n_tot = K.tf.reduce_sum(n)

        weights = [n_tot / (n[c] + eps) for c in range(restrict_chn)]

        y_pred_valid = K.tf.reshape(y_pred_valid, (-1, restrict_chn))
        y_true_valid = K.tf.to_float(K.tf.reshape(y_true_valid, (-1, restrict_chn)))
        softmax = K.tf.nn.softmax(y_pred_valid)

        w_cross_entropy = -K.tf.reduce_sum(K.tf.multiply(y_true_valid * K.tf.log(softmax + eps), weights),
                                           reduction_indices=[1])
        loss = K.tf.reduce_mean(w_cross_entropy, name='weighted_softmax_cross_entropy')
        return loss

    return cross_entropy


def make_tversky_loss_func(restrict_chn=1, beta=0.5):
    log.debug('Making Tversky loss function for the first %d channels' % restrict_chn)

    def loss(y_true,y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        numerator = K.sum(y_true * y_pred_new, axis=(1,2,3))
        denominator = K.sum(y_true * y_pred_new +
                            beta * (1 - y_true) * y_pred_new +
                            (1 - beta) * y_true * (1 - y_pred_new),
                            axis=(1,2,3))
        return 1 - K.mean((numerator + eps) / (denominator + eps),axis=0)
    return loss


def make_focal_loss_func(restrict_chn=1, gamma=2):
    log.debug('Making Focal Loss function for the first %d channels' % (restrict_chn + 1))
    def loss(y_true,y_pred):
        true_background = K.tf.ones_like(K.tf.expand_dims(y_true[...,0],axis=-1))
        for channel in range(restrict_chn):
            true_background = true_background -  K.tf.expand_dims(y_true[...,channel],axis=-1)
        y_true_valid= K.tf.concat([y_true,true_background],axis=-1)

        pred_background = K.tf.ones_like(K.tf.expand_dims(y_pred[..., 0], axis=-1))
        for channel in range(restrict_chn):
            pred_background = pred_background - K.tf.expand_dims(y_pred[..., channel], axis=-1)
        y_pred_valid = K.tf.concat([y_pred, pred_background], axis=-1)
        # y_pred_valid = y_pred

        y_pred_valid = K.tf.nn.softmax(y_pred_valid, dim=-1)  # [batch_size,num_classes]
        # labels = K.tf.one_hot(y_true, depth=y_pred.shape[1])
        L = -y_true_valid * ((1 - y_pred_valid) ** gamma) * K.tf.log(y_pred_valid)
        L = K.tf.reduce_mean(K.tf.reduce_sum(L, axis=1))
        return L
    return loss

def make_dice_loss_fnc(restrict_chn=1):
    log.debug('Making DICE loss function for the first %d channels' % restrict_chn)

    def loss(y_true, y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        intersection = K.sum(y_true * y_pred_new, axis=(1, 2, 3))
        union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred_new, axis=(1, 2, 3))
        return 1 - K.mean((2 * intersection + eps) / (union + eps), axis=0)

    return loss

def kl(args):
    mean, log_var = args
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.reshape(kl_loss, (-1, 1))


def ypred(y_true, y_pred):
    return y_pred


def make_triplet_loss(args, name):
    def _calculate_triplet(inputs):
        p, n, a = inputs
        d_positive = K.mean(K.square(a - p), axis=(-1))
        d_negative = K.mean(K.square(a - n), axis=(-1))
        return  d_positive / d_negative - 1

    real, reconstructed, pseudo, maximum_dist_pctg = args
    anchor = reconstructed
    positive = real
    negative = pseudo
    comparison = Lambda(_calculate_triplet)([positive,negative,anchor])
    return Lambda(lambda x:K.expand_dims(K.maximum(0.,maximum_dist_pctg+x), axis=-1), name='Triplet_' + name)(comparison)

