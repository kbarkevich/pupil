# distutils: language = c++
from libcpp.string cimport string

from result_2d_reserializer cimport (
	Detector2DResult
)


cdef string cget_serialized_string(object data):
	return "Hello, World!"
	
def get_serialized_string(data):
	return cget_serialized_string(data)