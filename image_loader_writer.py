import cv2
import tensorflow as tf
import numpy as np

#from face_intrinsics_processing import equalizeConstant, gamma, invGamma
from FACE_INTRINSICS_EDITING.env_map_processor import calculate_envMap_scale

"""
Abstraction module of OpenCV functions for loading and writing images
"""

# Rendering parameters
gamma = tf.constant(2.2)
invGamma = tf.constant(1./2.2)
equalizeConstant = 0.48/0.39

def load_rgba(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(temp)  # get b,g,r
    res = cv2.merge([r, g, b, a])
    return res

def load_rgb(path, code= 1):
    if (code==-1):
        temp = cv2.imread(path,-1)
    else:
        temp = cv2.imread(path)
    b, g, r = cv2.split(temp)  # get b,g,r
    res = cv2.merge([r, g, b])
    return res

def convert_rgb_to_cv2(im):
    r, g, b = cv2.split(im)
    res = cv2.merge([b, g, r])
    return res

def convert_rgba_to_cv2(im):
    r, g, b,a = cv2.split(im)
    res = cv2.merge([b, g, r,a])
    return res

def convert_cv2_to_rgb(im):
    b, g, r = cv2.split(im)
    res = cv2.merge([r, g, b])
    return res

def convert_cv2_to_rgb_without(im):
    b, g, r = cv2.split(im)
    res = cv2.merge([r, g, b])
    return res

def tonemap(imTensor, gamma):
    #temp = tf.multiply(1./tf.reduce_max(imTensor),imTensor)
    temp = tf.pow(imTensor,gamma)
    res = tf.clip_by_value(temp, 0, 1)
    return res



def save_tensor_cv2(tensor, path, envMap = -1,toneMap=False):
    if (envMap == -1):
        if toneMap:
            tensor=tonemap(tensor,gamma=1./2.2)
        res_save = 255. * np.array(tensor)
        res_save = convert_rgb_to_cv2(res_save)
    else: # write HDR
        res_save = np.array(tensor)
        res_save = convert_rgb_to_cv2(res_save)
    cv2.imwrite(path, res_save)
    return

def load_envMap(envFile,model, nEnv, mEnv, mRes = -1, nRes = -1):
    envMapOrig = load_rgb(envFile, -1)
    envMap = cv2.resize(envMapOrig, (nEnv, mEnv), interpolation=cv2.INTER_LINEAR)
    if mRes != - 1 :
        envMapOrig = cv2.resize(envMapOrig, (nRes, mRes), interpolation=cv2.INTER_LINEAR)
        # envMapOrig = tf.constant(envMapOrig, dtype=tf.float32)
        # envMapOrig = tf.reshape(envMapOrig, [1, mRes , nRes, 3])
    envMapTensor = tf.constant(envMap, dtype=tf.float32)
    envMapTensor = tf.reshape(envMapTensor, [1, mEnv , nEnv, 3])
    perc = calculate_envMap_scale(model, envMapTensor, mEnv=mEnv, nEnv=nEnv)
    percGamma = tf.pow(tf.constant(equalizeConstant / perc, dtype=tf.float32), gamma)
    envMapTensorBis = tf.scalar_mul(percGamma, envMapTensor)
    return envMapTensorBis,envMapOrig

def toneMap(image):
    image_nump = np.power(image,invGamma)
    res = 255. * image_nump
    res = np.clip(res,0,255)
    return res.astype(int)