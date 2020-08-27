from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, concatenate, \
    BatchNormalization, SpatialDropout2D, ZeroPadding2D, Permute, Add
from keras.layers import PReLU, Activation, Lambda
from keras.models import Model
# from tensorflow.python.keras import backend as K
from keras import backend as K
from keras.backend import expand_dims # because mi only in my
from models.basenet import BaseNet
import logging
log = logging.getLogger('enet')
from keras import regularizers


# In[5]:


class ENet(BaseNet):
    def __init__(self, conf):
        """
        Constructor.
        :param conf: the configuration object
        """
        super(ENet, self).__init__(conf) # inherent from the BaseNet Class
        self.input_shape  = conf.input_shape
        self.residual     = conf.residual
        self.out_channels = conf.out_channels
        self.normalise    = conf.normalise
        self.f            = conf.filters
        self.downsample   = conf.downsample
        self.regularizer = conf.regularizer
        assert self.downsample > 0, 'Unet downsample must be over 0.'

    def initial_block(self, tensor):
        conv = Conv2D(filters=16-self.input_shape[2], kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_conv',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(self.regularizer))(tensor)
        pool = MaxPooling2D(pool_size=(2, 2), name='initial_block_pool')(tensor)
        concat = concatenate([conv, pool], axis=-1, name='initial_block_concat')
        return concat


    def bottleneck_encoder(self,
                           tensor, nfilters, downsampling=False, dilated=False, asymmetric=False, normal=False, drate=0.1,
                           name=''):
        y = tensor
        skip = tensor
        stride = 1
        ksize = 1
        if downsampling:
            stride = 2
            ksize = 2
            skip = MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{name}')(skip)
            skip = Permute((1, 3, 2), name=f'permute_1_{name}')(skip)  # (B, H, W, C) -> (B, H, C, W)
            ch_pad = nfilters - K.int_shape(tensor)[-1]
            skip = ZeroPadding2D(padding=((0, 0), (0, ch_pad)), name=f'zeropadding_{name}')(skip)
            skip = Permute((1, 3, 2), name=f'permute_2_{name}')(skip)  # (B, H, C, W) -> (B, H, W, C)

        y = Conv2D(filters=nfilters // 4, kernel_size=(ksize, ksize), kernel_initializer='he_normal',
                   strides=(stride, stride), padding='same', use_bias=False, name=f'1x1_conv_{name}',
                   kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)
        y = PReLU(shared_axes=[1, 2], name=f'prelu_1x1_{name}')(y)

        if normal:
            y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same',
                       name=f'3x3_conv_{name}',
                       kernel_regularizer=regularizers.l2(self.regularizer))(y)
        elif asymmetric:
            y = Conv2D(filters=nfilters // 4, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same',
                       use_bias=False, name=f'5x1_conv_{name}',
                       kernel_regularizer=regularizers.l2(self.regularizer))(y)
            y = Conv2D(filters=nfilters // 4, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same',
                       name=f'1x5_conv_{name}',
                       kernel_regularizer=regularizers.l2(self.regularizer))(y)
        elif dilated:
            y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal',
                       dilation_rate=(dilated, dilated), padding='same', name=f'dilated_conv_{name}',
                       kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)
        y = PReLU(shared_axes=[1, 2], name=f'prelu_{name}')(y)

        y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   name=f'final_1x1_{name}',
                   kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)
        y = SpatialDropout2D(rate=drate, name=f'spatial_dropout_final_{name}')(y)

        y = Add(name=f'add_{name}')([y, skip])
        y = PReLU(shared_axes=[1, 2], name=f'prelu_out_{name}')(y)

        return y


    def bottleneck_decoder(self, tensor, nfilters, upsampling=False, normal=False, name=''):
        y = tensor
        skip = tensor
        if upsampling:
            skip = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                          padding='same', use_bias=False, name=f'1x1_conv_skip_{name}',
                          kernel_regularizer=regularizers.l2(self.regularizer))(skip)
            skip = UpSampling2D(size=(2, 2), name=f'upsample_skip_{name}')(skip)

        y = Conv2D(filters=nfilters // 4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                   padding='same', use_bias=False, name=f'1x1_conv_{name}',
                   kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_1x1_{name}')(y)
        y = PReLU(shared_axes=[1, 2], name=f'prelu_1x1_{name}')(y)

        if upsampling:
            y = Conv2DTranspose(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal', strides=(2, 2),
                                padding='same', name=f'3x3_deconv_{name}', kernel_regularizer=regularizers.l2(self.regularizer))(y)
        elif normal:
            Conv2D(filters=nfilters // 4, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                   padding='same', name=f'3x3_conv_{name}',
                   kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_main_{name}')(y)
        y = PReLU(shared_axes=[1, 2], name=f'prelu_{name}')(y)

        y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   name=f'final_1x1_{name}',
                   kernel_regularizer=regularizers.l2(self.regularizer))(y)
        y = BatchNormalization(momentum=0.1, name=f'bn_final_{name}')(y)

        y = Add(name=f'add_{name}')([y, skip])
        # y = ReLU(name=f'relu_out_{name}')(y)
        # return y
        return Activation('relu', name=f'relu_out_{name}')(y)


    # In[12]:
    def normal_seg_out(self, l):
        """
        Build output layer
        :param l: last layer from the upsampling path
        :return:  the final segmentation layer
        """
        out_activ = 'sigmoid' if self.out_channels == 1 else 'softmax'
        img_output = Conv2DTranspose(self.out_channels, kernel_size=(2, 2), strides=(2, 2),
                                     kernel_initializer='he_normal',
                                     padding='same', name='image_output',
                                     kernel_regularizer=regularizers.l2(self.regularizer),
                                     activation=out_activ)(l)
        # img_output = Activation(out_activ)(img_output)
        return img_output

    def out_regression_tenary(self, l):
        l1 = Conv2D(1,1, activation='sigmoid',
                    kernel_regularizer=regularizers.l2(self.regularizer))(l)
        return l1



    # harric added to incorporate with the segmentation_option=2 case
    # when the prediction is performed in a channel-wised manner
    # possibly useless
    def out_channel_wise_sigmoid(self,l):
        out_list = []
        for ii in range(self.out_channels):
            out_curt = Conv2D(1,1,activation='sigmoid',
                              kernel_regularizer=regularizers.l2(self.regularizer))(l)
            out_list.append(out_curt)
        output = concatenate(out_list,axis=3)
        return output

    # harric added to ensure the output infarction mask is within the myocardium region
    def infarction_mask_correction(self,output_mask):
        my_out = expand_dims(output_mask[..., 0], axis=-1)
        mi_out = expand_dims(output_mask[..., 1], axis=-1)
        back_out = expand_dims(output_mask[..., 2], axis=-1)
        mi_out = mi_out * my_out
        output_mask_corrected = concatenate([my_out, mi_out, back_out], axis=3)
        return output_mask_corrected

    def build(self, segmentation_option='-1'):
        if '6-' in segmentation_option:
            if segmentation_option == '6-All':
                self.input_shape = [self.input_shape[0],self.input_shape[1], 4]
            else:
                self.input_shape = [self.input_shape[0], self.input_shape[1], 1]

        img_height=self.input_shape[0]
        img_width=self.input_shape[1]
        img_channel=self.input_shape[2]
        nclasses=self.out_channels
        print('. . . . .Building ENet. . . . .')
        img_input = Input(shape=(img_height, img_width, img_channel), name='image_input')

        # Stage Initial:
        # input inception module
        x = self.initial_block(img_input)

        # Stage 1:
        # 5 bottleneck modules for downsample
        x = self.bottleneck_encoder(x, 64, downsampling=True, normal=True, name='1.0', drate=0.01)
        for _ in range(1, 5):
            x = self.bottleneck_encoder(x, 64, normal=True, name=f'1.{_}', drate=0.01)


        # Stage 2:
        # 9 bottle modules for downsample
        x = self.bottleneck_encoder(x, 128, downsampling=True, normal=True, name=f'2.0')
        x = self.bottleneck_encoder(x, 128, normal=True, name=f'2.1')
        x = self.bottleneck_encoder(x, 128, dilated=2, name=f'2.2')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name=f'2.3')
        x = self.bottleneck_encoder(x, 128, dilated=4, name=f'2.4')
        x = self.bottleneck_encoder(x, 128, normal=True, name=f'2.5')
        x = self.bottleneck_encoder(x, 128, dilated=8, name=f'2.6')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name=f'2.7')
        x = self.bottleneck_encoder(x, 128, dilated=16, name=f'2.8')

        # Stage 3:
        # 8 bottle modules for downsample
        x = self.bottleneck_encoder(x, 128, normal=True, name=f'3.0')
        x = self.bottleneck_encoder(x, 128, dilated=2, name=f'3.1')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name=f'3.2')
        x = self.bottleneck_encoder(x, 128, dilated=4, name=f'3.3')
        x = self.bottleneck_encoder(x, 128, normal=True, name=f'3.4')
        x = self.bottleneck_encoder(x, 128, dilated=8, name=f'3.5')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name=f'3.6')
        x = self.bottleneck_encoder(x, 128, dilated=16, name=f'3.7')

        # Stage 4:
        # 3 bottle modules for upsampling
        x = self.bottleneck_decoder(x, 64, upsampling=True, name='4.0')
        x = self.bottleneck_decoder(x, 64, normal=True, name='4.1')
        x = self.bottleneck_decoder(x, 64, normal=True, name='4.2')

        # Stage 5:
        # 2 bottle modules for upsampling
        x = self.bottleneck_decoder(x, 16, upsampling=True, name='5.0')
        x = self.bottleneck_decoder(x, 16, normal=True, name='5.1')

        # Stage output:
        # output layer
        # img_output = Conv2DTranspose(nclasses, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal',
        #                              padding='same', name='image_output')(x)
        # img_output = Activation('softmax')(img_output)
        if (not (self.conf.segmentation_option == '1' or self.conf.segmentation_option =='4'))\
                or '6' in self.conf.segmentation_option:
            out = self.normal_seg_out(x)
        elif self.conf.segmentation_option == '1':
            out = self.out_channel_wise_sigmoid(x)
        elif self.conf.segmentation_option=='4':
            out = self.out_regression_tenary(x)

        # harric added to ensure the output infarction mask is within the myocardium region
        if self.conf.segmentation_option == '1':
            out = Lambda(self.infarction_mask_correction)(out)

        naming_layer1 = Lambda(lambda x: x, name='dice')
        naming_layer2 = Lambda(lambda x: x, name='cret')
        out1 = naming_layer1(out)
        out2 = naming_layer2(out)



        self.model = Model(inputs=img_input, outputs=[out1,out2], name='ENET')
        print('. . . . .Build Compeleted. . . . .')

        self.model.summary(print_fn = log.info)
        self.load_models()
