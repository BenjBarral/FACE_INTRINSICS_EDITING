from enum import Enum, auto

import FACE_INTRINSICS_EDITING.face_intrinsics_processing as fip
import FACE_INTRINSICS_EDITING.video_synthesis as vs


# Different functionalities:
# - Get intrinsics of one face : show or save
# - Visualize intrinsics on top of image
# - Relighting : video
# - FACE MORPHING : get image (show or save) or video : morph background to white
# - VIDEO EDITING :
# 1- other face transforming to mine. Then my face animating (static pose, camera moving). Background changing and illumination.
# First : albedo and relighting. Then specular and relighting.
# Show normals. Then morph geometry to that of somebody else. Than normals to albedo.
# Animate other face : normal and mesh from me and albedo from them.

## Define different features
class ApplicationMode(Enum):
     INTRINSICS = 1
     VISUALIZE_INTRINSICS_ON_IMAGE = 2
     RELIGHTING = 3
     FACE_MORPHING = 4
     CREATE_ANIMATION = 5

class Media(Enum):
    IMAGE = 1
    VIDEO = 2


## USER : pick the mode and media types
mode = ApplicationMode.FACE_MORPHING  # the feature type
input_media = Media.IMAGE  # the type of input
output_media = Media.IMAGE  # the type of output
save_mode = True  # if you want to save the result in a folder

## USER : define the file locations
input_folder = "/Users/benjaminbarral/Documents/CODING/FACES/Experiments_Input/"
output_folder = "/Users/benjaminbarral/Documents/CODING/FACES/Experiment_Results/"
envMap_dir = "/Users/benjaminbarral/Documents/CODING/FACES/Media/EnvMaps/"
input_image1 = "benj.jpg"
input_image2 = "obama2.jpg"
input_video = "video_banger3.mov"

## VIDEO SYNTHESIS PARAMETERS FOR VIDEO OUT OF IMAGE AND FOR ANIMATION
DURATION_ANIMATION = 7
FPS = 15
num_frames = FPS * DURATION_ANIMATION


if mode == ApplicationMode.INTRINSICS:
    # Generate texture intrinsics
    fip.compute_intrinsics(input_folder, input_image1, save_mode=save_mode, output_folder=output_folder)

elif mode == ApplicationMode.VISUALIZE_INTRINSICS_ON_IMAGE:
    # Generate texture intrinsics and project them onto face image
    if input_media == Media.IMAGE:
        fip.visualize_intrinsics_on_image(input_folder, input_image1, save_mode, output_folder)
    else:
        vs.visualize_intrinsics_on_video(input_folder, input_video, input_image2, output_folder, debug_mode=False)

elif mode == ApplicationMode.RELIGHTING:
    if output_media == Media.IMAGE :
        fip.relight_image(input_folder, input_image1,save_mode=save_mode, output_folder=output_folder)
    elif output_media == Media.VIDEO :
        vs.relight_image_and_create_video(input_folder, input_image1, output_folder=output_folder, num_frames=num_frames, fps=FPS)

elif mode == ApplicationMode.FACE_MORPHING:
    if output_media == Media.IMAGE :
        fip.morph_faces(input_folder, input_image1, input_image2, save_mode=save_mode, output_folder=output_folder)
    elif output_media == Media.VIDEO :
        vs.morph_faces_and_create_video(input_folder, input_image1, input_image2, output_folder=output_folder, num_frames=num_frames, fps=FPS)

elif mode == ApplicationMode.CREATE_ANIMATION:
    vs.create_face_animation(input_folder, input_video, input_image2, envMap_dir, output_folder, num_frames,debug_mode=False)


