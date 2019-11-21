import matplotlib.pyplot as plt
import math
from enum import Enum
import random


import FACE_INTRINSICS_EDITING.face_intrinsics_processing as fip
from FACE_INTRINSICS_EDITING.image_loader_writer import *
import FACE_INTRINSICS_EDITING.env_map_processor as envOr

duration_blur = 2
num_envMaps = 5
prob_change_envMap = 0.01

class VideoEditingMode(Enum):
    SUPERPOSE_NORMALS = 1
    RELIGHT_INTRINSICS = 2
    RELIGHT_SYNTHETIC_ENVMAP = 3

class ResynthesisAttributes():
    def __init__(self, input_image, num_frames, fps, first_pose, envMap_dir, mIm, nIm, m1, m2,n1,n2):
        self.background = fip.make_white_image_background(input_image)
        self.num_frames = num_frames
        self.fps = fps
        self.ind_frame_lighting = 0
        self.ind_frame_pose = 0
        self.ratio = 0

        self.rotation_initial = first_pose.get_rotation()
        self.pose_render = first_pose

        # Load and save envMaps
        self.envMaps = []
        self.envMapBackgrounds = []
        self.mIm = mIm
        self.nIm = nIm
        self.m1 = m1
        self.m2 = m2
        self.n1 = n1
        self.n2 = n2
        for i in range(num_envMaps):
            envName = "env{}.hdr".format(i+1)
            (envMap, envMapBackground) = load_envMap(envMap_dir + envName, fip.generator_model,
                                                     mEnv=fip.mEnv, nEnv=fip.nEnv, mRes=mIm, nRes=nIm)
            envMapBackground = toneMap(envMapBackground)
            self.envMaps.append(envMap)
            self.envMapBackgrounds.append(envMapBackground)

        self.current_envMap = self.envMaps[0]
        self.current_envMapBackground = self.envMapBackgrounds[0]
        self.ratio_envMap_rot = 0
        return

    def update_ratio(self):
        self.ratio = 0.5 * (1. - math.cos(math.pi * 0.5 * self.ind_frame_lighting / (self.num_frames - 1)))
        self.ind_frame_lighting += 1
        return self.ratio

    def update_pose(self):
        # argCos = 2. * math.pi * self.ind_frame_pose / self.num_frames
        # a_render = self.rotation_initial[0] * math.cos(argCos)
        # b_render = self.rotation_initial[1] * math.cos(argCos)
        # c_render = self.rotation_initial[2] * math.cos(argCos)
        # self.pose_render.set_rotation(np.array([a_render, b_render, c_render, self.rotation_initial[3]]))

        self.ind_frame_pose += 1
        return self.pose_render

    def get_current_envMap(self):
        rd = random.random()
        if rd < prob_change_envMap:
            ind_env = (int)(random.random() * num_envMaps)
            self.current_envMap = self.envMaps[ind_env]
            self.current_envMapBackground = self.envMapBackgrounds[ind_env]

        self.ratio_envMap_rot += 0.25 / self.fps
        self.ratio_envMap_rot = self.ratio_envMap_rot - (int)(self.ratio_envMap_rot)
        if self.ratio_envMap_rot < 0:
            self.ratio_envMap_rot = self.ratio_envMap_rot + 1

        envMap_rot = tf.constant(envOr.rotate_envMap(self.current_envMap, -self.ratio_envMap_rot, fip.mEnv, fip.nEnv), dtype=tf.float32)
        envMap_background_video = (envOr.rotate_envMap(self.current_envMapBackground, -self.ratio_envMap_rot, self.mIm,
                                                       self.nIm, tensorMode=-1)).astype(int)

        envMap_background_proc = envMap_background_video[self.m1:self.m2, self.n1:self.n2, :]


        return envMap_rot, envMap_background_proc, envMap_background_video

def synthesize_loop_video(output_folder, input_image_name, frames, input_image, fps, extension):
    input_shape = np.shape(input_image)
    mIm = input_shape[0]
    nIm = input_shape[1]
    save_folder = output_folder + input_image_name[:-4] + "/"
    video_synthesizer = VideoSynthesizer(save_folder + extension, fps, mIm, nIm)
    video_synthesizer.create_loop_video_from_frames(frames,convert_rgb_to_cv2(input_image))
    return

def relight_image_and_create_video(input_folder, input_image_name, output_folder, num_frames, fps):
    input_image_path = input_folder + input_image_name
    input_image = load_rgb(input_image_path)
    int_im_info = fip.extract_texture_and_generate_intrinsics(input_image)

    white_background = fip.make_white_image_background(int_im_info.input_image)

    frames = []
    for ind in range(num_frames):
        relight_ratio = 0.5 * (1. - math.cos(math.pi * 0.5 * ind / (num_frames - 1)))
        edit_image = fip.relight_with_intrinsics(int_im_info.envMap, int_im_info.rotation, relight_ratio,
                                             int_im_info.albedo_texture, int_im_info.normal_texture, int_im_info.mesh,
                                             int_im_info.pose, white_background)
        frames.append(convert_rgb_to_cv2(edit_image))
        print("Created frame number {}".format(ind))
    print("FINISHED creating frames.")

    synthesize_loop_video(output_folder,input_image_name, frames, int_im_info.input_image, fps=fps, extension="relighting.mp4")

    return

def create_morphed_frames(num_frames, neur_im_info1, neur_im_info2, debug_mode=False):
    mesh1 = neur_im_info1.mesh
    mesh2 = neur_im_info2.mesh
    mesh = mesh1
    numvVertices1 = np.asarray(mesh1.vertices)
    numvVertices2 = np.asarray(mesh2.vertices)

    white_background = fip.make_white_image_background(neur_im_info1.input_image)

    frames = []
    for ind in range(num_frames):
        morph_ratio = math.cos(math.pi * 0.5 * ind / (num_frames - 1))
        (albedo_texture, normal_texture, rendered_appearance, envMap) = \
            fip.morph_intrinsics_with_neural_layers(neur_im_info1, neur_im_info2, morph_ratio)

        numpVertices = morph_ratio * numvVertices1 + (1. - morph_ratio) * numvVertices2
        mesh.vertices = [numpVertices[i, :] for i in range(np.shape(numpVertices)[0])]

        edit_image = fip.project_texture_on_image(mesh, neur_im_info1.pose, white_background,
                                                  fip.convert_tensor_to_image(rendered_appearance))

        if debug_mode:
            plt.imshow(edit_image)
            plt.show()

        frames.append(fip.convert_rgb_to_cv2(edit_image))
        print("Created frame number {}".format(ind))
    print("FINISHED creating frames.")
    return (frames, neur_im_info1.input_image)

def morph_faces_and_create_video(input_folder, input_image_name1, input_image_name2, output_folder, num_frames, fps):
    # Preprocessing : load, extract intrinsics layers, intialize objects
    input_image_path1 = input_folder + input_image_name1
    input_image1 = load_rgb(input_image_path1)
    input_image_path2 = input_folder + input_image_name2
    input_image2 = load_rgb(input_image_path2)


    neur_im_info1 = fip.extract_texture_and_generate_network_layers(input_image1)
    neur_im_info2 = fip.extract_texture_and_generate_network_layers(input_image2)

    (frames,image_shape) = create_morphed_frames(num_frames, neur_im_info1, neur_im_info2)

    synthesize_loop_video(output_folder, input_image_name1, frames, image_shape, fps=fps,
                          extension="_morphed_{}.mp4".format(input_image_name2))
    return


def create_blur_effect_frames(frames, input_image, fps):
    firstFrame = frames[0]
    num_frames = (int)(duration_blur * fps)
    for i in range(num_frames):
        lambd = i / (num_frames - 1)
        blur = lambd * input_image + (1. - lambd) * firstFrame
        blur = cv2.convertScaleAbs(blur.astype(int))
        frames = [blur] + frames
    # BLUR
    for i in range(int(fps)):
        frames = [input_image] + frames
    return frames

class VideoSynthesizer():

    """ Class for synthesizing videos"""

    def __init__(self, output_file, FPS, N, M):
        self.video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), FPS, (N, M))
        self.FPS = FPS

    def create_loop_video_from_frames(self, frames, input_image):
        frames = create_blur_effect_frames(frames, input_image, self.FPS)

        for frame in frames:
            self.video_writer.write(frame)

        frames.reverse()

        for frame in frames:
            self.video_writer.write(frame)

class VideoProcessor():
    def __init__(self, input_folder, input_video, input_static_image, output_folder,
                 debug_mode=False, num_frames=-1, envMap_dir=-1):
        video_input_file = input_folder + input_video
        self.cap = cv2.VideoCapture(video_input_file)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS = {}".format(self.fps))

        self.video_reader = cv2.VideoCapture(video_input_file)
        # Read first frame to get video resolution
        ret, frame = self.video_reader.read()
        input_image = convert_cv2_to_rgb(frame)
        input_shape = np.shape(input_image)
        mIm = input_shape[0]
        nIm = input_shape[1]


        # Video writer
        self.video_writer = cv2.VideoWriter(output_folder + input_video[:-4] + "_edited.mp4",
                                       cv2.VideoWriter_fourcc(*'MP4V'), self.fps, (nIm, mIm))
        self.frames_write = []

        # Precompute variables for frame resizing for EOS
        halfM = int(mIm / 2)
        halfN = int(nIm / 2)

        if mIm < nIm:
            self.m1 = 0
            self.m2 = mIm
            self.n1 = int(nIm / 2) - halfM
            self.n2 = int(nIm / 2) + halfM
            width = mIm
        else:
            self.n11 = 0
            self.n2 = nIm
            self.m1 = int(mIm / 2) - halfN
            self.m2 = int(mIm / 2) + halfN
            width = nIm


        # Precompute intrinsics for static face
        input_image_path = input_folder + input_static_image
        input_image = load_rgb(input_image_path)
        input_image = cv2.resize(input_image, (width,width))
        self.static_face_info = fip.extract_texture_and_generate_network_layers(input_image,debug_mode=debug_mode)

        # Reynthesis attributes
        self.resynthesis_attributes = ResynthesisAttributes(input_image, num_frames, self.fps, self.static_face_info.pose,
                                                            envMap_dir=envMap_dir, mIm=mIm, nIm=nIm,
                                                            m1=self.m1, m2=self.m2, n1=self.n1, n2=self.n2)
        self.white_background_video = fip.make_white_image_background(frame)
        self.envMap_dir = envMap_dir


    def process_frame(self, frame, edit_mode, debug_mode=False, resynthesis_attributes=-1, specularity=False):
        original_frame = convert_cv2_to_rgb(frame)
        input_image = original_frame[self.m1:self.m2, self.n1:self.n2, :]

        int_im_info = fip.extract_texture_and_generate_intrinsics(input_image)
        if int_im_info == -1:
            return None

        if edit_mode == VideoEditingMode.SUPERPOSE_NORMALS:
            edited_frame = fip.project_texture_on_image(int_im_info.mesh, int_im_info.pose, int_im_info.input_image,
                                                    fip.convert_tensor_to_image(int_im_info.normal_texture))
            result_image = convert_rgb_to_cv2(original_frame)
        elif edit_mode == VideoEditingMode.RELIGHT_INTRINSICS:
            edited_frame = fip.relight_with_intrinsics(int_im_info.envMap, int_im_info.rotation,
                                                       resynthesis_attributes.update_ratio(),
                                                       int_im_info.albedo_texture, int_im_info.normal_texture,
                                                       int_im_info.mesh,
                                                       resynthesis_attributes.update_pose(),
                                                       resynthesis_attributes.background)
            result_image = convert_rgb_to_cv2(self.white_background_video)
        elif edit_mode == VideoEditingMode.RELIGHT_SYNTHETIC_ENVMAP:
            envMap_rot, envMap_background_proc, envMap_background_video = resynthesis_attributes.get_current_envMap()
            edited_frame = fip.relight_with_intrinsics(envMap_rot, int_im_info.rotation,
                                                       0,
                                                       int_im_info.albedo_texture, int_im_info.normal_texture,
                                                       int_im_info.mesh,
                                                       resynthesis_attributes.update_pose(),
                                                       envMap_background_proc, specular_mode=specularity)
            result_image = convert_rgb_to_cv2(envMap_background_video)


        result = convert_rgb_to_cv2(edited_frame)
        result_image[self.m1:self.m2, self.n1:self.n2, :] = result[:, :, :]

        result_image = cv2.convertScaleAbs(result_image.astype(int))

        if debug_mode:
            plt.imshow(result_image)
            plt.show()

        return result_image

    def process_video(self, debug_mode, edit_mode, num_frames = 1e5, specularity=False):
        if (self.video_reader.isOpened() == False):
            print("Error opening video stream or file")

        frame_counter = 0

        while (self.video_reader.isOpened() and frame_counter < num_frames):
            ret, frame = self.video_reader.read()
            frame_counter += 1
            if ret == True:
                # LOAD IMAGE
                result_image = self.process_frame(frame, edit_mode, debug_mode,
                                                  resynthesis_attributes=self.resynthesis_attributes, specularity=specularity)
                if not (result_image is None):
                    self.frames_write.append(result_image)
                    print("Frame {}".format(frame_counter))
            else:
                break

    def morph_static_faces(self, num_frames, debug_mode=False):
        if (self.video_reader.isOpened() == False):
            print("Error opening video stream or file")

        ret, frame = self.video_reader.read()
        original_frame = convert_cv2_to_rgb(frame)
        input_image = original_frame[self.m1:self.m2, self.n1:self.n2, :]

        int_im_info = fip.extract_texture_and_generate_network_layers(input_image,debug_mode=debug_mode)


        (frames, frame_shape) = create_morphed_frames(num_frames, self.static_face_info, int_im_info, debug_mode)

        frames = create_blur_effect_frames(frames, self.static_face_info.input_image, self.fps)

        self.frames_write = self.frames_write + frames

    def write_video(self):
        for frame in self.frames_write:
            self.video_writer.write(frame)
        return

def visualize_intrinsics_on_video(input_folder, input_video, input_static_image, output_folder, debug_mode=False):
    video_processor = VideoProcessor(input_folder, input_video, input_static_image, output_folder)
    video_processor.process_video(debug_mode, edit_mode=VideoEditingMode.SUPERPOSE_NORMALS)
    video_processor.write_video()
    return


def create_face_animation(input_folder, input_video, input_static_image, envMap_dir,
                          output_folder, num_frames, debug_mode=False):
    video_processor = VideoProcessor(input_folder, input_video, input_static_image, output_folder,
                                     debug_mode=debug_mode, num_frames = num_frames, envMap_dir=envMap_dir)

    # STEP ONE : morph static face to first frame's face
    video_processor.morph_static_faces(num_frames=num_frames, debug_mode=debug_mode)

    # STEP TWO : READ VIDEO, MOVE CAMERA AND RELIGHT WITH INTRINSICS
    video_processor.process_video(debug_mode, edit_mode=VideoEditingMode.RELIGHT_INTRINSICS, num_frames=num_frames)

    # STEP THREE : READ FRAMES, MOVE CAMERA AND RELIGHT WITH SYNTHETIC ENVMAPS
    video_processor.process_video(debug_mode, edit_mode=VideoEditingMode.RELIGHT_SYNTHETIC_ENVMAP,
                                  num_frames=num_frames,specularity=True)

    # WRITE VIDEO
    video_processor.write_video()
    video_processor.video_reader.release()

    return


# argCos = 2. * math.pi * sm_frame / duration_stop_motion
# a_render = rotation_initial[0] * math.cos(argCos)
# b_render = rotation_initial[1] * math.cos(argCos)
# c_render = rotation_initial[2] * math.cos(argCos)
# pose_render.set_rotation(np.array([a_render, b_render, c_render, rotation_initial[3]]))