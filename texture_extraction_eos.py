import eos
import numpy as np
import cv2
import dlib
from imutils import face_utils
import copy

"""
The texture extraction from image module : exploiting EOS (https://github.com/patrikhuber/eos) 
and DLIB (https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).
"""

### EOS : h
eos_data_dir = "/Users/benjaminbarral/Documents/Academic/UCL/Research Project/EOS_Tests/eos/"
model_file = eos_data_dir + "share/sfm_shape_3448.bin"
blendshapes_file = eos_data_dir + "share/expression_blendshapes_3448.bin"
landmark_mapper_file = eos_data_dir + 'share/ibug_to_sfm.txt'
edge_topology_file = eos_data_dir + 'share/sfm_3448_edge_topology.json'
contour_landmarks_file = eos_data_dir + 'share/ibug_to_sfm.txt'
model_contour_file = eos_data_dir + 'share/sfm_model_contours.json'
landmark_detector_file = '/Users/benjaminbarral/Documents/CODING/FACES/shape_predictor_68_face_landmarks.dat'

model = eos.morphablemodel.load_model(model_file)
blendshapes = eos.morphablemodel.load_blendshapes(blendshapes_file)
# Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                    color_model=eos.morphablemodel.PcaModel(),
                                                                    vertex_definitions=None,
                                                                    texture_coordinates=model.get_texture_coordinates())
landmark_mapper = eos.core.LandmarkMapper(landmark_mapper_file)
edge_topology = eos.morphablemodel.load_edge_topology(edge_topology_file)
contour_landmarks = eos.fitting.ContourLandmarks.load(contour_landmarks_file)
model_contour = eos.fitting.ModelContour.load(model_contour_file)


### DLIB landmark detection
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(landmark_detector_file)

def make_texture_mask(texture,texture_resolution = 256):
    res = np.zeros((texture_resolution,texture_resolution,3))
    for i in range(3):
        res[:, :, i] = texture[:, :, 3]
    return res


def detectlandmarks(detector, predictor, image, draw_landmarks = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    res_image = -1
    if draw_landmarks:
        res_image = copy.copy(image)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if draw_landmarks:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(res_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(res_image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(res_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(res_image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        landmarks = []
        ibug_index = 1
        for (x, y) in shape:
            landmarks.append(eos.core.Landmark(str(ibug_index), [x, y]))
            ibug_index = ibug_index + 1
            if draw_landmarks:
                cv2.circle(res_image, (x, y), 5, (0, 0, 255), -1)

        if landmarks:
            return (landmarks,res_image)
        else: # bad detection
            return (-1,-1)



def extract_texture(input_image_load, texture_resolution = 256):
    input_image = input_image_load
    (image_height,image_width,image_channels) = input_image.shape

    image_resize = 1000

    # DETECT LANDMARKS
    landmarks_pair = detectlandmarks(dlib_detector, dlib_predictor, input_image, image_resize)
    if landmarks_pair:
        landmarks = landmarks_pair[0]
        image_with_landmarks = landmarks_pair[1]
    else:
        return (-1 * np.ones((texture_resolution,texture_resolution,3)),-1,-1,-1,-1,-1)

    (image_height,image_width,image_channels) = input_image.shape

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
        landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour)


    # Extract the texture map, like this:
    image_texture = eos.render.extract_texture(mesh, pose, input_image, isomap_resolution = texture_resolution)

    normal_map = eos.render.extract_normal_map(mesh, pose, input_image, isomap_resolution = texture_resolution)

    rotation_matrix = eos.render.get_rotation_matrix(pose, input_image)

    texture_mask = make_texture_mask(image_texture)

    cv2.transpose(image_texture, image_texture)
    cv2.transpose(normal_map, normal_map)
    cv2.transpose(texture_mask, texture_mask)

    return (image_texture,normal_map,mesh,pose,rotation_matrix,texture_mask)


def read_pts(filename):
    """A helper function to read the 68 ibug landmarks from a .pts file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for l in lines:
        coords = l.split()
        landmarks.append(eos.core.Landmark(str(ibug_index), [float(coords[0]), float(coords[1])]))
        ibug_index = ibug_index + 1

    return landmarks
