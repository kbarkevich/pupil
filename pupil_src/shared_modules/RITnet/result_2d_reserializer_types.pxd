# distutils: language = c++
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from '<opencv2/core.hpp>' namespace 'cv':
	cdef cppclass Rect_[T]:
		Rect_() except +
		Rect_( T x, T y, T width, T height ) except +
		T x, y, width, height
		
	cdef cppclass Point_[T]:
		Point_() except +
		
		
cdef extern from '<Eigen/Eigen>' namespace 'Eigen':
	cdef cppclass Matrix21d "Eigen::Matrix<double,2,1>": # eigen defaults to column major layout
		Matrix21d() except +
		double * data()
		double& operator[](size_t)


cdef extern from 'common/types.h':
	cdef cppclass Ellipse2D[T]:
		Ellipse2D()
		Ellipse2D(T x, T y, T major_radius, T minor_radius, T angle) except +
		Matrix21d center
		T major_radius
		T minor_radius
		T angle
		
	# typedefs
	ctypedef vector[Point_[int]] Edges2D
	ctypedef Ellipse2D[double] Ellipse

	cdef struct Detector2DResult:
		double confidence
		Ellipse ellipse
		Edges2D final_edges
		Edges2D raw_edges
		Rect_[int] current_roi
		double timestamp
		int image_width
		int image_height
		string serialize()
