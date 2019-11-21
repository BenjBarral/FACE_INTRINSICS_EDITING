## Adapted from Neural Face Editing : http://openaccess.thecvf.com/content_cvpr_2017/papers/Shu_Neural_Face_Editing_CVPR_2017_paper.pdf

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import math


"""
The definition of the intrinsics generation neural network.
"""

num_layers_encoder = 6
downscaling_ratio = math.pow(2,num_layers_encoder)
num_layers_decoder_env = 3
downscaling_ratio_env = math.pow(2,num_layers_decoder_env)

bottle_neck_layer_size = 3000 #2048
bottle_neck_albedo = 2000 #1024
bottle_neck_normal = 1024 #1024
bottle_neck_env = 2000 #1024

layer1_encoder_filters = 32
layer2_encoder_filters = 64
layer3_encoder_filters = 128
layer4_encoder_filters = 256
layer5_encoder_filters = 512
layer6_encoder_filters = 1024

skip_layers_albedo_mode = True
skip_layers_normal_mode = True

invGamma = tf.constant(1. / 2.2)
tone_power_env = tf.constant(1. / 3.2)

L1_ENV = False
LOG_LOSS_ENV = False
LINEAR_LOSS_ENV = False

std_dev = 2e-2
max_envMap_value = tf.constant(200000.,tf.float32)
min_envMap_value = tf.constant(1e-10,tf.float32)

strides_normal = 1
strides_envMap = 2

scale_tm_env = tf.constant(.3,dtype=tf.float32)

lambda_smooth_mask = 1e-6
lambda_mean_env = 100

spec_alpha = 30.
k_spec = 1.5
k_diff = 0.1



class GeneratorModel(tf.keras.Model):

    def __init__(self, input_shape, envOrientationTensor, halfOrientationsTensor, envNormalization, envMap_size, high_res_mode=False, smooth_mask_mode=False):
        super(GeneratorModel, self).__init__()
        self._input_shape = [-1,input_shape[0],input_shape[1],input_shape[2]]
        self.mIm = input_shape[0]
        self.nIm = input_shape[1]
        self.envNormalization = envNormalization
        self.envOrientationTensor = envOrientationTensor
        self.halfOrientationsTensor = halfOrientationsTensor
        self.smooth_mask_mode = smooth_mask_mode

        self.high_res_mode = high_res_mode

        self.conv1 = tf.layers.Conv2D(filters = layer1_encoder_filters, kernel_size= 3, strides=2,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=std_dev),
                                      activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv2 = tf.layers.Conv2D(filters = layer2_encoder_filters, kernel_size= 3, strides=2,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=std_dev),
                                      activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv3 = tf.layers.Conv2D(filters=layer3_encoder_filters, kernel_size=3, strides=2,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=std_dev),
                                      activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv4 = tf.layers.Conv2D(filters=layer4_encoder_filters, kernel_size=3, strides=2,
                                      padding='same', kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                                      activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.conv5 = tf.layers.Conv2D(filters=layer5_encoder_filters, kernel_size=3, strides=2,
                                      padding='same', kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                                      activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))




        last_layer_conv = layer5_encoder_filters
        if num_layers_encoder == 6:
            self.conv6 = tf.layers.Conv2D(filters=layer6_encoder_filters, kernel_size=3, strides=2,
                                          padding='same',
                                          kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
                                          activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))
            last_layer_conv = layer6_encoder_filters

        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(bottle_neck_layer_size, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.fc2 = tf.layers.Dense(bottle_neck_albedo, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.fc3 = tf.layers.Dense(bottle_neck_normal, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.fc_env = tf.layers.Dense(bottle_neck_env, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))

        layer_size = (self.mIm / downscaling_ratio) * (self.nIm / downscaling_ratio)
        self.fc4_normal = tf.layers.Dense(last_layer_conv * layer_size, activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.fc4_albedo = tf.layers.Dense(last_layer_conv * layer_size, activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.envMap_bottle_size = ((int)(envMap_size[0] / downscaling_ratio_env), (int)(envMap_size[1] / downscaling_ratio_env))
        envMap_layer_size = self.envMap_bottle_size[0] * self.envMap_bottle_size[1]
        self.fc4_env =  tf.layers.Dense(last_layer_conv * envMap_layer_size, activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))

        if num_layers_encoder == 6:
            self.deconv00_normal = tf.layers.Conv2DTranspose(filters=layer6_encoder_filters, kernel_size=3,
                                                            padding='same',  strides=strides_normal,
                                                            activation=tf.nn.relu,
                                                            kernel_initializer=tf.random_normal_initializer(
                                                                stddev=std_dev),
                                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.deconv00_albedo = tf.layers.Conv2DTranspose(filters=layer6_encoder_filters, kernel_size=3,
                                                            padding='same', strides=2,
                                                            activation=tf.nn.relu,
                                                            kernel_initializer=tf.random_normal_initializer(
                                                                stddev=std_dev),
                                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))


        self.deconv0_normal = tf.layers.Conv2DTranspose(filters=layer5_encoder_filters, kernel_size=3,
                                               padding='same', strides=strides_normal,
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv1_normal = tf.layers.Conv2DTranspose(filters=layer4_encoder_filters, kernel_size=3,
                                               padding='same', strides=strides_normal,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv2_normal = tf.layers.Conv2DTranspose(filters=layer3_encoder_filters, kernel_size=3,
                                               padding='same', strides=strides_normal,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv3_normal = tf.layers.Conv2DTranspose(filters=layer2_encoder_filters, kernel_size=3,
                                               padding='same', strides=strides_normal,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv4_normal = tf.layers.Conv2DTranspose(filters=layer1_encoder_filters, kernel_size=3,
                                                        padding='same', strides=strides_normal,
                                                        activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.deconv0_albedo = tf.layers.Conv2DTranspose(filters=layer5_encoder_filters, kernel_size=3,
                                               padding='same', strides=2,
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv1_albedo = tf.layers.Conv2DTranspose(filters=layer4_encoder_filters, kernel_size=3,
                                               padding='same', strides=2,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv2_albedo = tf.layers.Conv2DTranspose(filters=layer3_encoder_filters, kernel_size=3,
                                               padding='same', strides=2,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv3_albedo = tf.layers.Conv2DTranspose(filters=layer2_encoder_filters, kernel_size=3,
                                               padding='same', strides=2,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv4_albedo = tf.layers.Conv2DTranspose(filters=layer1_encoder_filters, kernel_size=3,
                                                        padding='same', strides=2,
                                                        activation=tf.nn.relu,
                                                        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),kernel_regularizer=tf.keras.regularizers.l2(0.01))


        if num_layers_encoder == 6:
            layer1_deconv_env = layer6_encoder_filters
            layer2_deconv_env = layer5_encoder_filters
            layer3_deconv_env = layer4_encoder_filters
            layer4_deconv_env = layer3_encoder_filters
        else:
            layer1_deconv_env = layer5_encoder_filters
            layer2_deconv_env = layer4_encoder_filters
            layer3_deconv_env = layer3_encoder_filters
            layer4_deconv_env = layer2_encoder_filters

        self.deconv0_env = tf.layers.Conv2DTranspose(filters=layer1_deconv_env, kernel_size=3,
                                            padding='same', strides=strides_envMap,
                                            activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv1_env = tf.layers.Conv2DTranspose(filters=layer2_deconv_env, kernel_size=3,
                                            padding='same', strides=strides_envMap,
                                            activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv2_env = tf.layers.Conv2DTranspose(filters=layer3_deconv_env, kernel_size=3,
                                               padding='same', strides=strides_envMap,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        if num_layers_decoder_env==4:
            self.deconv3_env = tf.layers.Conv2DTranspose(filters=layer4_deconv_env, kernel_size=3,
                                               padding='same', strides=strides_envMap,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))


        self.deconv_final_albedo = tf.layers.Conv2DTranspose(filters=3, kernel_size=3,
                                              padding='same', #strides = 2,
                                              activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv_final_normal = tf.layers.Conv2DTranspose(filters=3, kernel_size=3,
                                              padding='same', #strides = strides_normal,
                                              activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.deconv_final_env = tf.layers.Conv2DTranspose(filters=3, kernel_size=3,
                                               padding='same', #strides = strides_envMap,
                                               activation=tf.nn.relu, kernel_initializer = tf.random_normal_initializer(stddev=std_dev), kernel_regularizer=tf.keras.regularizers.l2(0.01))

        #self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.up_sampl2d = tf.keras.layers.UpSampling2D(size = (2, 2))

    def render(self, normalTensor, albedoTensor):
        #normalTensorSign = normalTensor
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        cosineTensor = tf.matmul(normalTensorSign, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0, 1)
        shadedBrightness = tf.matmul(cosineTensor, self.envMapTensor)
        shadedBrightness = tf.scalar_mul(1. / self.envNormalization,
                                         tf.reshape(shadedBrightness,
                                                [n_shape[0], n_shape[1], n_shape[2], 3]))
        resTensor = tf.multiply(albedoTensor, shadedBrightness)
        return resTensor

    def render_with_predicted_envMap(self, normalTensor, albedoTensor, predicted_envMap):
        #normalTensorSign = normalTensor
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        cosineTensor = tf.matmul(normalTensorSign, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0., 1.)
        cosineTensor = tf.reshape(cosineTensor, [n_shape[0],n_shape[1] * n_shape[2], cosineTensor.shape[1]])
        shadedBrightness = tf.matmul(cosineTensor, predicted_envMap)
        shadedBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(shadedBrightness,
                                                [n_shape[0], n_shape[1], n_shape[2], 3]))
        resTensor = tf.multiply(albedoTensor, shadedBrightness)
        return resTensor

    def render_with_predicted_envMap_Rot(self, normalTensor, albedoTensor, predicted_envMap,rotTensor):
        #normalTensorSign = normalTensor
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0], n_shape[1] * n_shape[2], 3])
        normalTensorCameraCS = tf.matmul(normalTensorSign, rotTensor)
        normalTensorCameraCS = tf.math.l2_normalize(normalTensorCameraCS, axis=2)
        normalTensorCameraCS = tf.reshape(normalTensorCameraCS, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        normalTensorCameraCS = tf.reverse(normalTensorCameraCS, axis=[-1])
        cosineTensor = tf.matmul(normalTensorCameraCS, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0., 1.)
        cosineTensor = tf.reshape(cosineTensor, [n_shape[0],n_shape[1] * n_shape[2], cosineTensor.shape[1]])
        shadedBrightness = tf.matmul(cosineTensor, predicted_envMap)
        shadedBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(shadedBrightness,
                                                [n_shape[0], n_shape[1], n_shape[2], 3]))
        resTensor = tf.multiply(albedoTensor, shadedBrightness)
        return resTensor

    def render_with_specularity(self, normalTensor, albedoTensor, predicted_envMap,rotTensor):
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0], n_shape[1] * n_shape[2], 3])
        normalTensorCameraCS = tf.matmul(normalTensorSign, rotTensor)
        normalTensorCameraCS = tf.math.l2_normalize(normalTensorCameraCS, axis=2)
        normalTensorCameraCS = tf.reshape(normalTensorCameraCS, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        normalTensorCameraCS = tf.reverse(normalTensorCameraCS, axis=[-1])
        cosineTensor = tf.matmul(normalTensorCameraCS, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0., 1.)
        cosineTensor = tf.reshape(cosineTensor, [n_shape[0], n_shape[1] * n_shape[2], cosineTensor.shape[1]])
        blinnPhongTensor = tf.matmul(normalTensorCameraCS, self.halfOrientationsTensor)
        blinnPhongTensor = tf.clip_by_value(blinnPhongTensor, 0., 1.)
        blinnPhongTensor = tf.pow(blinnPhongTensor,spec_alpha)
        blinnPhongTensor = tf.reshape(blinnPhongTensor, [n_shape[0], n_shape[1] * n_shape[2], blinnPhongTensor.shape[1]])
        diffuseBrightness = tf.matmul(cosineTensor, predicted_envMap)
        diffuseBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(diffuseBrightness,
                                                                                [n_shape[0], n_shape[1], n_shape[2],
                                                                                 3]))
        specularBrightness = tf.matmul(blinnPhongTensor, predicted_envMap)
        specularBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(specularBrightness,
                                                                                 [n_shape[0], n_shape[1], n_shape[2],
                                                                                  3]))
        diffuseTensor = tf.multiply(albedoTensor, diffuseBrightness)
        specularTensor = tf.multiply(albedoTensor, specularBrightness)

        resTensor = k_spec * specularTensor + k_diff * diffuseTensor

        return resTensor


    def call(self, x,rot):
        x1 = tf.reshape(x, self._input_shape)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        if num_layers_encoder==6:
            x7 = self.conv6(x6)
            code = self.flatten(x7)
            last_shape = x7.shape
        else:
            code = self.flatten(x6)
            last_shape = x6.shape
        # x is size 32x32

        #code = self.flatten(x6)

        code = self.fc1(code)  # code


        y = self.fc2(code)
        z = self.fc3(code)
        w = self.fc_env(code)
        y = self.fc4_albedo(y)
        z = self.fc4_normal(z)
        w = self.fc4_env(w)

        y = tf.reshape(y,last_shape)
        if num_layers_encoder == 6:
            if skip_layers_albedo_mode:
                y = tf.concat([y, x7], axis=3)
            y = self.deconv00_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x6], axis=3)
        y = self.deconv0_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y,x5],axis = 3)
        y = self.deconv1_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y,x4], axis=3)
        y= self.deconv2_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y,x3], axis=3)
        y = self.deconv3_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x2], axis=3)
        y = self.deconv4_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x1], axis=3)

        z = tf.reshape(z, last_shape)
        if strides_normal==2:
            if num_layers_encoder==6:
                if skip_layers_normal_mode:
                    z = tf.concat([z, x7], axis=3)
                z = self.deconv00_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x6], axis=3)
            z = self.deconv0_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x5], axis=3)
            z = self.deconv1_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x4], axis=3)
            z = self.deconv2_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x3], axis=3)
            z = self.deconv3_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x2], axis=3)
            z = self.deconv4_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x1], axis=3)
        else:
            if num_layers_encoder==6:
                if skip_layers_normal_mode:
                    z = tf.concat([z, x7], axis=3)
                z = self.up_sampl2d(self.deconv00_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x6], axis=3)
            z = self.up_sampl2d(self.deconv0_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x5], axis=3)
            z = self.up_sampl2d(self.deconv1_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x4], axis=3)
            z = self.up_sampl2d(self.deconv2_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x3], axis=3)
            z = self.up_sampl2d(self.deconv3_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x2], axis=3)
            z = self.up_sampl2d(self.deconv4_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x1], axis=3)

        w = tf.reshape(w, [x.shape[0], self.envMap_bottle_size[0], self.envMap_bottle_size[1], last_shape[3]])
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv0_env(w))
        else:
            w = self.deconv0_env(w)
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv1_env(w))
        else:
            w = self.deconv1_env(w)
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv2_env(w))
        else :
            w = self.deconv2_env(w)
        if num_layers_decoder_env==4:
            if strides_envMap == 1:
                w = self.up_sampl2d(self.deconv3_env(w))
            else:
                w = self.deconv3_env(w)

        albedo = self.deconv_final_albedo(y)
        normal = self.deconv_final_normal(z)
        env_pred = tf.clip_by_value(self.deconv_final_env(w), 0., max_envMap_value)


        res = self.render_with_predicted_envMap_Rot(normal, albedo,
                                                tf.reshape(env_pred, [env_pred.shape[0],
                                                                      env_pred.shape[1] *
                                                                      env_pred.shape[2], 3]),rot)
        res = tf.pow(res, invGamma)

        albedo = tf.pow(albedo, invGamma)
        return (albedo, normal, res, env_pred)

    def compute_code(self, x):
        x1 = tf.reshape(x, self._input_shape)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        if num_layers_encoder==6:
            x7 = self.conv6(x6)
            code = self.flatten(x7)
            last_shape = x7.shape
        else:
            code = self.flatten(x6)
            last_shape = x6.shape

        code = self.fc1(code)  # code

        return (code, x1,x2,x3,x4,x5,x6,x7)

    def generate_from_code(self, code, x1,x2,x3,x4,x5,x6,x7,rot):
        if num_layers_encoder == 6:
            last_shape = x7.shape
        else:
            last_shape = x6.shape

        y = self.fc2(code)
        z = self.fc3(code)
        w = self.fc_env(code)
        y = self.fc4_albedo(y)
        z = self.fc4_normal(z)
        w = self.fc4_env(w)

        y = tf.reshape(y, last_shape)
        if num_layers_encoder == 6:
            if skip_layers_albedo_mode:
                y = tf.concat([y, x7], axis=3)
            y = self.deconv00_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x6], axis=3)
        y = self.deconv0_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x5], axis=3)
        y = self.deconv1_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x4], axis=3)
        y = self.deconv2_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x3], axis=3)
        y = self.deconv3_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x2], axis=3)
        y = self.deconv4_albedo(y)
        if skip_layers_albedo_mode:
            y = tf.concat([y, x1], axis=3)

        z = tf.reshape(z, last_shape)
        if strides_normal == 2:
            if num_layers_encoder == 6:
                if skip_layers_normal_mode:
                    z = tf.concat([z, x7], axis=3)
                z = self.deconv00_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x6], axis=3)
            z = self.deconv0_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x5], axis=3)
            z = self.deconv1_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x4], axis=3)
            z = self.deconv2_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x3], axis=3)
            z = self.deconv3_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x2], axis=3)
            z = self.deconv4_normal(z)
            if skip_layers_normal_mode:
                z = tf.concat([z, x1], axis=3)
        else:
            if num_layers_encoder == 6:
                if skip_layers_normal_mode:
                    z = tf.concat([z, x7], axis=3)
                z = self.up_sampl2d(self.deconv00_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x6], axis=3)
            z = self.up_sampl2d(self.deconv0_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x5], axis=3)
            z = self.up_sampl2d(self.deconv1_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x4], axis=3)
            z = self.up_sampl2d(self.deconv2_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x3], axis=3)
            z = self.up_sampl2d(self.deconv3_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x2], axis=3)
            z = self.up_sampl2d(self.deconv4_normal(z))
            if skip_layers_normal_mode:
                z = tf.concat([z, x1], axis=3)

        w = tf.reshape(w, [last_shape[0], self.envMap_bottle_size[0], self.envMap_bottle_size[1], last_shape[3]])
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv0_env(w))
        else:
            w = self.deconv0_env(w)
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv1_env(w))
        else:
            w = self.deconv1_env(w)
        if strides_envMap == 1:
            w = self.up_sampl2d(self.deconv2_env(w))
        else:
            w = self.deconv2_env(w)
        if num_layers_decoder_env == 4:
            if strides_envMap == 1:
                w = self.up_sampl2d(self.deconv3_env(w))
            else:
                w = self.deconv3_env(w)

        albedo = self.deconv_final_albedo(y)
        normal = self.deconv_final_normal(z)
        env_pred = tf.clip_by_value(self.deconv_final_env(w), 0., max_envMap_value)

        res = self.render_with_predicted_envMap_Rot(normal, albedo,
                                                    tf.reshape(env_pred, [env_pred.shape[0],
                                                                          env_pred.shape[1] *
                                                                          env_pred.shape[2], 3]), rot)
        res = tf.pow(res, invGamma)

        albedo = tf.pow(albedo, invGamma)
        return (albedo, normal, res, env_pred)

