import pyximport
from pyximport import install

old_get_distutils_extension = pyximport.pyximport.get_distutils_extension

def new_get_distutils_extension(modname, pyxfilename, language_level=None):
    extension_mod, setup_args = old_get_distutils_extension(modname, pyxfilename, language_level)
    extension_mod.language='c++'
    extension_mod.include_dirs=['C:/shared_cpp/OpenCV/opencv/build/include',
                                'C:/shared_cpp/Eigen/eigen-3.3.8', 
                                'C:/shared_cpp/pupil-detectors-master/src/shared_cpp/include',
                                'C:/shared_cpp/pupil-detectors-master/src/singleeyefitter',
                                'C:/shared_cpp/ceres-solver-1.14.0/include',
                                'C:/shared_cpp/ceres-solver-1.14.0/config']
    extension_mod.define_macros=[('CERES_USE_CXX11_THREADS', 1)]
    return extension_mod,setup_args

pyximport.pyximport.get_distutils_extension = new_get_distutils_extension