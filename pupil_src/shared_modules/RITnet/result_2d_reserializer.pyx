# distutils: language = c++
from libcpp.string cimport string

from result_2d_reserializer_types cimport (
	Detector2DResult
)

cdef string cget_serialized_string(object data):
	cdef Detector2DResult result
	#result.confidence = 0.99
	#cdef Detector2DResult *result = <Detector2DResult *> malloc(sizeof(Detector2DResult))
	#result.confidence = 0.99
	# if data["confidence"]:
	#	result.confidence = data["confidence"]
	# else:
	#	result.confidence = 0.99
	
	# result.ellipse.center[0] = data["ellipse"]["center"][0]
	# result.ellipse.center[1] = data["ellipse"]["center"][1]
	# result.ellipse.minor_radius = data["ellipse"]["axes"][0]
	# result.ellipse.major_radius = data["ellipse"]["axes"][1]
	# result.ellipse.angle = data["ellipse"]["angle"]
	# if data["confidence"]:
	#	result.confidence = data["confidence"]
	# else:
	#	result.confidence = 0.99
	return "Hello, World!"
	#return result.serialize()
	
def get_serialized_string(data):
	return cget_serialized_string(data)