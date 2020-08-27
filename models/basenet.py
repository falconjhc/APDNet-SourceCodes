import importlib
import os
from abc import abstractmethod

from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from callbacks.image_callback import SaveImage
from costs import make_dice_loss_fnc, weighted_softmax_cross_entropy,  make_mse_loss_func, make_mse_loss_func_distributed
import logging

from loaders import loader_factory

log = logging.getLogger('basenet')


class BaseNet(object):
    """
    Base model for segmentation neural networks
    """
    def __init__(self, conf):
        self.model = None
        self.conf = conf
        self.loader = None
        if hasattr(self.conf, 'dataset_name') and len(self.conf.dataset_name) > 0:
            self.loader = loader_factory.init_loader(self.conf.dataset_name)

    @abstractmethod
    def build(self):
        pass

    def load_models(self):
        if os.path.exists(self.conf.folder + '/model'):
            log.info('Loading trained model from file')
            self.model.load_weights(self.conf.folder + '/model')

    def save_models(self, postfix=''):
        log.debug('Saving trained model')
        self.model.save_weights(self.conf.folder + '/model' + postfix)

    def compile(self):
        assert self.model is not None, 'Model has not been built'

        ce = weighted_softmax_cross_entropy(self.conf.num_masks)
        if not self.conf.segmentation_option == '4':
            loss_fnc = 1

            # harric added the segmentation option argument
            # possibly useless ...
        else: # harric added to incorporate with segmentation_option=4: see it as a regression task
            if self.conf.loss_type == 'mse_calling':
                loss_fnc = ['mse']
            elif self.conf.loss_type == 'mae_calling':
                loss_fnc = ['mae']
            elif self.conf.loss_type == 'mse_keras_func':
                loss_fnc = make_mse_loss_func(self.conf.num_masks)
            # elif self.conf.loss_type == 'mae_keras_func':
            #     loss_fnc = make_mae_loss_func(self.conf.num_masks)
            elif '+' in self.conf.loss_type:
                loss_fnc = make_mse_loss_func_distributed(self.conf.num_masks,
                                                          infarction_weight=self.conf.infarction_weight,
                                                          loss_type = self.conf.loss_type)
        self.model.compile(optimizer=Adam(lr=self.conf.lr, decay=self.conf.decay), loss=[loss_fnc, ce],
                           loss_weights=[1,1])
    def load_data(self):
        train_data = self.loader.load_data(self.conf.split, 'training')
        valid_data = self.loader.load_data(self.conf.split, 'validation')

        num_l = int(train_data.num_volumes * self.conf.l_mix)
        num_l = num_l if num_l <= self.conf.data_len else self.conf.data_len
        print('Using %d labelled volumes.' % (num_l))
        train_data.sample(num_l)
        return train_data, valid_data

    # def fit(self):
    #     train_data, valid_data = self.load_data()
    #
    #     es = EarlyStopping(min_delta=0.001, patience=20)
    #     si = SaveImage(os.path.join(self.conf.folder, 'training_results'), train_data.images, train_data.masks)
    #     cl = CSVLogger(os.path.join(self.conf.folder, 'training_results') + '/training.csv')
    #
    #     if not os.path.exists(os.path.join(self.conf.folder, 'training_results')):
    #         os.mkdir(os.path.join(self.conf.folder, 'training_results'))
    #
    #     if self.conf.augment:
    #         datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    #         self.model.fit_generator(datagen.flow(train_data.images, train_data.masks, batch_size=self.conf.batch_size),
    #                                  steps_per_epoch=2 * len(train_data.images) / self.conf.batch_size, epochs=self.conf.epochs,
    #                                  validation_data=(valid_data.images, valid_data.masks))
    #     else:
    #         self.model.train(train_data.images, train_data.masks, validation_data=(valid_data.images, valid_data.masks),
    #                          epochs=self.conf.epochs, callbacks=[es, si, cl], batch_size=self.conf.batch_size)

    @abstractmethod
    def get_segmentor(self):
        """
        Create a model for segmentation
        :return: a keras model
        """
        return self.model
