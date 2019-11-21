import numpy as np
import math
import tensorflow as tf

#from face_intrinsics_processing import invGamma

""" Module for precomputing envMap parameters (direction images and tensors for rendering pipeline).
"""
# Rendering parameters
gamma = tf.constant(2.2)
invGamma = tf.constant(1./2.2)
equalizeConstant = 0.48/0.39

def envMapOrientation(mEnv,nEnv, sphere_probe):
    orientationArray = np.zeros((3,mEnv, nEnv))
    for q in range(mEnv):
        for r in range(nEnv):
            if (sphere_probe):
                viewVec = orientationVectorFromSpherEnvMap(q, r, mEnv, nEnv)
            else:
                viewVec = viewVectorFrom360EnvMap(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[:,q, r] = viewVec
    return orientationArray

def halfOrientations(mEnv,nEnv,orientationArray):
    halfVectorsArray = np.zeros((3, mEnv, nEnv))
    for q in range(mEnv):
        for r in range(nEnv):
            viewVec = orientationArray[:,q,r]
            halfVec = np.array([0.,0.,-1.]) + viewVec
            halfVec /= np.linalg.norm(halfVec)
            halfVectorsArray[:, q, r] = halfVec
    return halfVectorsArray


def normalSphereOrientation(mEnv,nEnv):
    orientationArray = np.zeros((mEnv, nEnv,3))
    for q in range(mEnv):
        for r in range(nEnv):
            viewVec = orientationVectorFromSpherEnvMapDebug(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[q, r,:] = viewVec
    return orientationArray

def sphereWhiteAlbedo(mEnv,nEnv):
    orientationArray = np.zeros((mEnv, nEnv,3))
    for q in range(mEnv):
        for r in range(nEnv):
            colorVec = colorSphere(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[q, r,:] = colorVec
    return orientationArray

def orientationVectorFromSpherEnvMap(i,j,m,n):
    view = np.array([0,0,1.])
    x0 = (n-1)/2
    x = (j-x0)/x0
    y = -(i-((m-1)/2))/((m-1)/2)
    sqr = x**2 + y**2
    if (sqr > 1):
        return np.array([0,0,0])
    else:
        z = math.sqrt(1-sqr)
        normal = np.array([x,y,z])
        res = 2. * np.dot(view,normal) * normal - view
        res = np.array([res[0], res[1], res[2]])
        return res

def colorSphere(i,j,m,n):
    view = np.array([0, 0, 1.])
    x0 = (n - 1) / 2
    x = (j - x0) / x0
    y = -(i - ((m - 1) / 2)) / ((m - 1) / 2)
    sqr = x ** 2 + y ** 2
    if (sqr > 1):
        return np.array([0, 0, 0])
    else:
        return np.array([1.,1.,1.])

def orientationVectorFromSpherEnvMapDebug(i,j,m,n):
    view = np.array([0,0,1.])
    x0 = (n-1)/2
    x = (j-x0)/x0
    y = -(i-((m-1)/2))/((m-1)/2)
    sqr = x**2 + y**2
    if (sqr > 1):
        return np.array([0,0,0])
    else:
        z = math.sqrt(1-sqr)
        normal = np.array([x,y,z])
        res = 2. * np.dot(view,normal) * normal - view
        res = np.array([res[0], res[1], res[2]])

    res_debug = (res + np.array([1,1,1])) / 2.
    return res_debug

def viewVectorFrom360EnvMap(i,j,m,n):
    phi = (math.pi) * i / (m-1)
    #theta = math.pi * (i-((n-1)/2))/((n-1)/2)
    #theta = 2. * math.pi * i / (n - 1)
    #theta = math.pi / 2. + 2. * math.pi * i / (n - 1)
    theta = (2. * math.pi) * (j - (n-1)/2) / (n-1)
    y = math.cos(phi)
    x = math.sin(phi) * math.sin(theta)
    z = math.sin(phi) * math.cos(theta)

    return np.array([x,y,z])

def viewVectorFrom360EnvMapDebug(i,j,m,n):
    phi = (math.pi) * i / (m-1)
    #theta = math.pi * (i-((n-1)/2))/((n-1)/2)
    #theta = 2. * math.pi * i / (n - 1)
    #theta = math.pi / 2. + 2. * math.pi * i / (n - 1)
    theta = (2. * math.pi) * (j - (n-1)/2) / (n-1)
    y = math.cos(phi)
    x = math.sin(phi) * math.sin(theta)
    z = math.sin(phi) * math.cos(theta)

    res = np.array([x,y,z])
    res_debug = (res + np.array([1.,1.,1.])) / 2.
    return res_debug

def testInSphere(i,j,m,n):
    x0 = (n - 1) / 2.
    x = (j - x0) / x0
    y = -(i - ((m - 1) / 2.)) / ((m - 1) / 2.)
    sqr = x ** 2 + y ** 2
    if (sqr > 1):
        return False
    return True

def getMaskSphereMap(mEnv,nEnv):
    maskArray = np.zeros((mEnv, nEnv,3))
    for q in range(mEnv):
        for r in range(nEnv):
            if (testInSphere(q, r, mEnv, nEnv)):
                for k in range(3):
                    maskArray[q,r,k] = 1

    return maskArray

def envMapOrientationDebug(mEnv,nEnv, sphere_probe):
    orientationArray = np.zeros((mEnv, nEnv,3))
    for q in range((mEnv)):
        for r in range(nEnv):
            if (sphere_probe):
                viewVec = orientationVectorFromSpherEnvMapDebug(q, r, mEnv, nEnv)
            else:
                viewVec = viewVectorFrom360EnvMapDebug(q, r, mEnv, nEnv)
            # viewVec = viewVectorFromEnvMap(q, r, nIllum, mIllum)
            orientationArray[q, r,:] = viewVec
    return orientationArray

def rotate_envMap(envMap, ratio, m,n,tensorMode = 1):
    if tensorMode != 1:
        res = np.zeros([m, n, 3])
    else:
        res = np.zeros([1, m, n, 3])
    deltaJ = ratio*n
    for j in range(n):
        jOrig = (j - deltaJ)
        jOrig0 = math.floor(jOrig)
        jOrig1 = jOrig0 + 1
        lambdaJ = jOrig - jOrig0
        assert lambdaJ >= 0 and lambdaJ <= 1

        if (jOrig0 < 0):
            jOrig0 = jOrig0 + n
        if (jOrig0 >= n):
            jOrig0 = jOrig0 - n

        if (jOrig1 < 0):
            jOrig1 = jOrig1 + n
        if (jOrig1 >= n):
            jOrig1 = jOrig1 - n

        if tensorMode != 1:
            resPix = lambdaJ * envMap[:, jOrig1, :] + (1. - lambdaJ) * envMap[:, jOrig0, :]
            res[:, j, :] = resPix[:]
        else:
            resPix = lambdaJ * envMap[0,: , jOrig1, :] + (1. - lambdaJ) * envMap[0, :, jOrig0, :]
            res[0, :, j, :] = resPix[:]



    return res

def rotate_envMapBis(envMap, ratio, m,n,tensorMode = 1):
    if tensorMode != 1:
        res = np.zeros([m, n, 3])
    else:
        res = np.zeros([1, m, n, 3])
    deltaJ = ratio*n
    for i in range(m):
        for j in range(n):
            jOrig = (j-deltaJ)
            jOrig0 = (int)(jOrig)
            jOrig1 = jOrig0 + 1
            lambdaJ = jOrig - jOrig0
            assert lambdaJ >=0 and lambdaJ <=1

            if (jOrig0 < 0):
                jOrig0 = jOrig0 + n
            if (jOrig0 >= n):
                jOrig0 = jOrig0 - n

            if (jOrig1 < 0):
                jOrig1 = jOrig1 + n
            if (jOrig1 >= n):
                jOrig1 = jOrig1 - n

            if tensorMode != 1:
                resPix = lambdaJ * envMap[0,i,jOrig1,:] + (1. - lambdaJ) * envMap[i,jOrig0,:]
            else:
                resPix = lambdaJ * envMap[0, i, jOrig1, :] + (1. - lambdaJ) * envMap[0, i, jOrig0, :]

            res[0,i,j,:] = resPix[:]
    return res

def initialize_envMap_parameters(envMap_resolution):
    mEnv = envMap_resolution[0]
    nEnv = envMap_resolution[1]
    envVectors = envMapOrientation(mEnv, nEnv, False)
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])

    halfVectors = halfOrientations(mEnv, nEnv, envVectors)
    halfOrientationsTensor = tf.constant(halfVectors, dtype=tf.float32)
    halfOrientationsTensor = tf.reshape(halfOrientationsTensor, [3, mEnv * nEnv])

    envNormalization = tf.constant(float(mEnv * nEnv), dtype=tf.float32)

    return (envOrientationTensor, halfOrientationsTensor, envNormalization)

def calculate_envMap_scale(model, envMapTensor, mEnv, nEnv):
    mIm = 100
    nIm = 100
    white_Sphere_Albedo = tf.constant(sphereWhiteAlbedo(mIm, nIm), dtype=tf.float32)
    white_Sphere_Normals = tf.constant(normalSphereOrientation(mIm, nIm), dtype=tf.float32)

    white_Sphere_Albedo = tf.reshape(white_Sphere_Albedo, [1, mIm, nIm, 3])
    white_Sphere_Normals = tf.reshape(white_Sphere_Normals, [1, mIm, nIm, 3])
    resTensor = model.render_with_predicted_envMap(white_Sphere_Normals, white_Sphere_Albedo,
                                                               tf.reshape(envMapTensor,
                                                                          [1, mEnv * nEnv, 3]))

    resTensorGamma = tf.pow(resTensor, invGamma)
    resNump = np.array(resTensorGamma[0])
    perc = np.percentile(resNump, 85)
    return perc