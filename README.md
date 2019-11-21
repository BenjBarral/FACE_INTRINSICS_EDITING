# A Python program to edit videos and images of faces, using Tensorflow and a pretrained neural network for face intrinsics decomposition

## Usage
### Dependencies

You need to build the python binding from the [EOS](https://github.com/libigl/libigl) library  (try `pip install eos-py`).

Other packages : 
- Tensorflow
- OpenCV 
- Numpy

You will need the pretrained weights of the face texture intrinsics deep neural network : contact me at benjybarral@gmail.com.

### Running the code
Run the ```face_app.py``` module.
You will need to modify the paths to the input video or image to edit : `input_folder`, `input_image1` etc...

## References

_A Multiresolution 3D Morphable Face Model and Fitting Framework_, P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler, International Conference on Computer Vision Theory and Applications (VISAPP) 2016, Rome, Italy [[PDF]](http://www.patrikhuber.ch/files/3DMM_Framework_VISAPP_2016.pdf).

Dlib facial landmark detection : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
