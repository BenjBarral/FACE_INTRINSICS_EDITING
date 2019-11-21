
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import cv2
import eos
import os
import matplotlib.pyplot as plt
import math

from FACE_INTRINSICS_EDITING.image_loader_writer import *
from FACE_INTRINSICS_EDITING.texture_extraction_eos import extract_texture
import FACE_INTRINSICS_EDITING.env_map_processor as envOr
from FACE_INTRINSICS_EDITING.generator_model import GeneratorModel
from FACE_INTRINSICS_EDITING.video_synthesis import VideoSynthesizer

### Path to the pretrained weights : to get the weights, contact me at benjybarral@gmail.com.
model_path = "/Users/benjaminbarral/Documents/Academic/UCL/Research Project/TensorFlowExperimentations/TrainedModel2/"

## PARAMETERS AND OBJECTS
texture_resolution = 256
envMap_resolution = (16, 32)
mEnv = envMap_resolution[0]
nEnv = envMap_resolution[1]
# Parameters for visualization of albedo and envMap
average_envMap_scale = tf.constant(0.13)
albedo_rescale = 1.1
# Rendering parameters
gamma = tf.constant(2.2)
invGamma = tf.constant(1./2.2)
equalizeConstant = 0.48/0.39
# Initialize envMap variables
(envOrientationTensor, halfOrientationsTensor, envNormalization) = envOr.initialize_envMap_parameters(envMap_resolution)
# Initialize the intrinsics generator model
generator_model = GeneratorModel((texture_resolution, texture_resolution, 3), envOrientationTensor, halfOrientationsTensor, envNormalization,
                                     envMap_resolution)
# Load the pretrained weights
root = tf.train.Checkpoint(generator_model=generator_model)
checkpoint_manager = tf.train.CheckpointManager(root, model_path, max_to_keep=2)
root.restore(checkpoint_manager.latest_checkpoint)


### AUXILIARY FUNCTIONS
def save_image(output_folder, input_image_name, image, extension):
    save_folder = output_folder + input_image_name[:-4] + "/"
    if not (os.path.isdir(save_folder)):
        os.mkdir(save_folder)
    cv2.imwrite(save_folder + extension, convert_rgb_to_cv2(image))

def make_white_image_background(input_image):
    input_shape = np.shape(input_image)
    mIm = input_shape[0]
    nIm = input_shape[1]
    res = np.zeros((mIm, nIm, 3))
    for i in range(mIm):
        for j in range(nIm):
            for q in range(3):
                res[i, j, q] = 255
    return res.astype(int)

def convert_images_to_tensors(rotation_matrix, input_texture,texture_mask):
    rotation_matrix_save = np.zeros(shape=(3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            rotation_matrix_save[i, j] = -rotation_matrix[i, j]
    for j in range(3):
        rotation_matrix_save[0, j] = rotation_matrix[0, j]

    rotation_matrix_save = rotation_matrix_save.transpose()

    input_texture_tensor = tf.constant(input_texture[:, :, :3], dtype=tf.float32)
    input_texture_tensor = tf.scalar_mul(1. / 255., input_texture_tensor)
    input_texture_tensor = tf.reshape(input_texture_tensor, [1, texture_resolution, texture_resolution, 3])

    mask_tensor = tf.constant(texture_mask / 255., dtype=tf.float32)
    mask_tensor = tf.reshape(mask_tensor, [1,texture_resolution, texture_resolution, 3])

    return input_texture_tensor,rotation_matrix_save,mask_tensor

def convert_tensor_to_image(tensor_texture):
    numpy_texture = 255. * tensor_texture[0].numpy()
    res = np.zeros((texture_resolution, texture_resolution, 4))
    lim = 255
    for i in range(texture_resolution):
        for j in range(texture_resolution):
            for q in range(3):
                res[i, j, q] = numpy_texture[i, j, q]
            res[i, j, 3] = 255
    return res.astype(int)

def create_transparent_mask(input, mask,normalize=False):
    res = np.concatenate((input,mask),axis=2)
    return res[:,:,:4]

def write_result_texture(texture,mask,output_res_base,extension):
    res_save = ((255. * texture[0]).numpy()).astype(int)
    res_save = create_transparent_mask(res_save, mask)
    res_save = convert_rgba_to_cv2(res_save)
    cv2.imwrite(output_res_base + extension, res_save)

def project_texture_on_image(mesh, pose, background_image, texture):
    edit_image = eos.render.render_with_texture(mesh, pose, cv2.convertScaleAbs(background_image), cv2.convertScaleAbs(texture),
                                                isomap_resolution=texture_resolution)
    cv2.transpose(edit_image, edit_image)
    return edit_image


def in_network_render(normalTensor,albedo_render,envMapTensor,autoencoder_model,rotationTensor, specular_mode = False):
    albedo_renderGamma = tf.pow(albedo_render,gamma)
    if specular_mode :
        resTensor = autoencoder_model.render_with_specularity(
            tf.reshape(normalTensor, [1, texture_resolution, texture_resolution, 3]),
            tf.reshape(albedo_renderGamma, [1, texture_resolution, texture_resolution, 3]),
            tf.reshape(envMapTensor,
                       [1, mEnv * nEnv, 3]),
            tf.reshape(rotationTensor, [1, 3, 3]))
    else :
        resTensor = autoencoder_model.render_with_predicted_envMap_Rot(
            tf.reshape(normalTensor, [1, texture_resolution, texture_resolution, 3]),
            tf.reshape(albedo_renderGamma, [1, texture_resolution, texture_resolution, 3]),
            tf.reshape(envMapTensor,
                       [1, mEnv * nEnv, 3]),
            tf.reshape(rotationTensor, [1, 3, 3]))
    resTensorGamma = tf.pow(resTensor, invGamma)
    return (resTensorGamma[0]).numpy()

def render_new_texture(albedo, normal, envMap, rotation, model,clamp=False, specular_mode = False):
    # all inputs are textures (1,M,N,3), in [0,1]
    rotationTensor = tf.constant(rotation.transpose(), dtype=tf.float32)
    new_texture = in_network_render(normal, albedo, envMap,model, rotationTensor, specular_mode=specular_mode)
    #if clamp:
        #new_texture = np.power(new_texture,gamma)
    new_texture = 255. * new_texture
    res = np.zeros((texture_resolution, texture_resolution, 4))
    lim = 255
    if clamp:
        lim = 225
    for i in range(texture_resolution):
        for j in range(texture_resolution):
            for q in range(3):
                if new_texture[i, j, q] > lim:
                    res[i, j, q] = lim
                else:
                    res[i, j, q] = new_texture[i, j, q]
            res[i, j, 3] = 255
    return res.astype(int)


## OBJECTS FOR IMAGE AND INTRINSICS INFO
class ImageInfo():
    def __init__(self, input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation):
        self.input_image = input_image
        self.image_texture = image_texture
        self.normal_map = normal_map
        self.mesh = mesh
        self.pose = pose
        self.rot = rot
        self.texture_mask = texture_mask
        self.rotation = rotation

class NeuralLayersAndImageInfo(ImageInfo):
    def __init__(self, input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation,
                 neural_layers_result):
        super().__init__(input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation)
        self.neural_layers_result = neural_layers_result

class IntrinsicsAndImageInfo(ImageInfo):
    def __init__(self, input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation, albedo_texture,
                 normal_texture, rendered_appearance, envMap):
        super().__init__(input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation)
        self.albedo_texture = albedo_texture
        self.normal_texture = normal_texture
        self.rendered_appearance = rendered_appearance
        self.envMap = envMap


### DIFFERENT FEATURE MODES
def extract_texture_and_generate_intrinsics(input_image):
    (image_texture, normal_map, mesh, pose, rot, texture_mask) = extract_texture(input_image)

    if image_texture[0, 0, 0] == -1:
        # Face fitting failure
        return -1
    input_texture_tensor, rotation, mask_tensor = convert_images_to_tensors(rot, image_texture, texture_mask)
    (albedo_texture, normal_texture, rendered_appearance, envMap) = generator_model(input_texture_tensor, rotation)

    return IntrinsicsAndImageInfo(input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation, albedo_texture, normal_texture, rendered_appearance, envMap)

def extract_texture_and_generate_network_layers(input_image, save_mode=False, output_folder=-1, debug_mode = False):
    (image_texture, normal_map, mesh, pose, rot, texture_mask) = extract_texture(input_image)
    if debug_mode :
        plt.imshow(image_texture)
        plt.show()
    if image_texture[0, 0, 0] == -1:
        # Face fitting failure
        return -1
    input_texture_tensor, rotation, mask_tensor = convert_images_to_tensors(rot, image_texture, texture_mask)

    result = generator_model.compute_code(input_texture_tensor)

    return NeuralLayersAndImageInfo(input_image, image_texture, normal_map, mesh, pose, rot, texture_mask, rotation, result)

def compute_intrinsics(input_folder, input_image_name, save_mode=False, output_folder=-1):
    input_image_path = input_folder + input_image_name
    input_image = load_rgb(input_image_path)
    int_im_info = extract_texture_and_generate_intrinsics(input_image)

    if save_mode:
        save_folder = output_folder + input_image_name[:-4] + "/"
        if not (os.path.isdir(save_folder)):
            os.mkdir(save_folder)
        cv2.imwrite(save_folder + "_InputImage.png", convert_rgb_to_cv2(int_im_info.input_image))
        cv2.imwrite(save_folder + "_Appearance_gt.png", convert_rgba_to_cv2(int_im_info.image_texture))
        write_result_texture(int_im_info.normal_texture, int_im_info.texture_mask, save_folder, extension="_Normal_map.png")
        write_result_texture(int_im_info.rendered_appearance, int_im_info.texture_mask, save_folder, extension="_Appearance_map.png")
        write_result_texture(albedo_rescale * int_im_info.albedo_texture, int_im_info.texture_mask, save_folder,
                             extension="_Albedo_map.png")
        envMap_write = int_im_info.envMap[0] * average_envMap_scale
        save_tensor_cv2(envMap_write, save_folder + "_EnvMap_result.png", envMap=-1, toneMap=True)
    else:
        plt.imshow(int_im_info.image_texture)
        plt.show()
        for tensor in [int_im_info.albedo_texture, int_im_info.normal_texture, int_im_info.rendered_appearance]:
            plt.imshow(tensor[0].numpy())
            plt.show()

    return (int_im_info.albedo_texture, int_im_info.normal_texture, int_im_info.rendered_appearance, int_im_info.envMap)

def visualize_intrinsics_on_image(input_folder, input_image_name, save_mode=False, output_folder=-1):
    input_image_path = input_folder + input_image_name
    input_image = load_rgb(input_image_path)
    int_im_info = extract_texture_and_generate_intrinsics(input_image)

    image_normal = project_texture_on_image(int_im_info.mesh, int_im_info.pose, int_im_info.input_image, convert_tensor_to_image(int_im_info.normal_texture))
    image_albedo = project_texture_on_image(int_im_info.mesh, int_im_info.pose, int_im_info.input_image, convert_tensor_to_image(int_im_info.albedo_texture))

    if save_mode:
        save_image(output_folder, input_image_name, image_normal, "_imageNormal.png")
        save_image(output_folder, input_image_name, image_albedo, "_imageAlbedo.png")

    else:
        plt.imshow(image_albedo)
        plt.show()
        plt.imshow(image_normal)
        plt.show()
    return

def relight_with_intrinsics(envMap,rotation, relight_ratio, albedo_texture, normal_texture,mesh, pose,
                            white_background, specular_mode=False):
    envMapRot = tf.constant(envOr.rotate_envMap(envMap, -relight_ratio, mEnv, nEnv), dtype=tf.float32)
    new_textureRelit = render_new_texture(albedo_texture, normal_texture, envMapRot, rotation,
                                          generator_model, specular_mode=specular_mode)
    edit_image = project_texture_on_image(mesh, pose, white_background, new_textureRelit)
    return edit_image

def relight_image(input_folder, input_image_name, relight_ratio = 0.5, save_mode=False, output_folder=-1):
    input_image_path = input_folder + input_image_name
    input_image = load_rgb(input_image_path)
    int_im_info = extract_texture_and_generate_intrinsics(input_image)

    white_background = make_white_image_background(int_im_info.input_image)

    edit_image = relight_with_intrinsics(int_im_info.envMap, int_im_info.rotation, relight_ratio,
                                         int_im_info.albedo_texture, int_im_info.normal_texture, int_im_info.mesh,
                                         int_im_info.pose, white_background)

    if save_mode:
        save_image(output_folder,input_image_name,edit_image, extension="_imageRelit.png")
    else:
        plt.imshow(edit_image)
        plt.show()

    return edit_image

def morph_intrinsics_with_neural_layers(neur_im_info1, neur_im_info2, ratio):
    res1 = neur_im_info1.neural_layers_result
    res2 = neur_im_info2.neural_layers_result
    average_code = ratio * res1[0] + (1. - ratio) * res2[0]
    x = [0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 8):
        x[i - 1] = ratio * res1[i] + (1. - ratio) * res2[i]

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]

    rot = neur_im_info1.rotation
    pose = neur_im_info1.pose
    (albedo_texture, normal_texture, rendered_appearance, envMap) = generator_model.generate_from_code(average_code, x1,
                                                                                                       x2, x3, x4, x5,
                                                                                                       x6, x7, rot)

    return (albedo_texture, normal_texture, rendered_appearance, envMap)

def morph_faces(input_folder, input_image_name1, input_image_name2, morph_ratio = 0.5, save_mode=False, output_folder=-1):
    input_image_path1 = input_folder + input_image_name1
    input_image1 = load_rgb(input_image_path1)
    input_image_path2 = input_folder + input_image_name2
    input_image2 = load_rgb(input_image_path2)
    neur_im_info1 = extract_texture_and_generate_network_layers(input_image1)
    neur_im_info2 = extract_texture_and_generate_network_layers(input_image2)
    mesh1 = neur_im_info1.mesh
    mesh2 = neur_im_info2.mesh
    mesh = mesh1
    numvVertices1 = np.asarray(mesh1.vertices)
    numvVertices2 = np.asarray(mesh2.vertices)

    (albedo_texture, normal_texture, rendered_appearance, envMap) = morph_intrinsics_with_neural_layers(neur_im_info1, neur_im_info2, morph_ratio)

    white_background = make_white_image_background(neur_im_info1.input_image)

    numpVertices = morph_ratio * numvVertices1 + (1. - morph_ratio) * numvVertices2
    mesh.vertices = [numpVertices[i, :] for i in range(np.shape(numpVertices)[0])]

    edit_image = project_texture_on_image(mesh, neur_im_info1.pose, white_background,
                                   convert_tensor_to_image(rendered_appearance))

    if save_mode:
        save_folder = output_folder + input_image_name1[:-4] + "/"
        envMap_write = envMap[0] * average_envMap_scale
        save_tensor_cv2(envMap_write, save_folder + "_EnvMap_morphed.png", envMap=-1, toneMap=True)
        save_image(output_folder,input_image_name1,edit_image, extension="_morphed_{}.png".format(input_image_name2))
        write_result_texture(albedo_rescale * albedo_texture, neur_im_info1.texture_mask, save_folder,
                             extension="_Albedo_morphed.png")
        write_result_texture(normal_texture, neur_im_info1.texture_mask, save_folder,
                             extension="_Normal_morphed.png")
        write_result_texture(rendered_appearance, neur_im_info1.texture_mask, save_folder,
                             extension="_Appearance_morphed.png")
    else:
        plt.imshow(edit_image)
        plt.show()

    return

