# ----------------------------------------------------------------------------
# <summary> CMakeLists.txt for OpenCV and CUDA                  </summary>
# <date>    2014-04-25                                          </date>
# <author>  Tran Minh Quan                                      </author>
# <email>   quantm@unist.ac.kr                                  </email>
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(renderer)
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
# ----------------------------------------------------------------------------
if(CUDA_FOUND)
    message("CUDA is found on this system.")
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_35,code=sm_35; 
		-gencode;arch=compute_30,code=sm_30; 
		-gencode;arch=compute_20,code=sm_20; 
		")

	# add -Wextra compiler flag for gcc compilations
	if (UNIX)
		# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--disable-warnings;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	endif (UNIX)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OPENCV is found on this system.")
	include_directories(${OpenCV_INCLUDE_DIR})
else(OpenCV_FOUND)
    message("OPENCV is not installed on this system.")
endif()


# ----------------------------------------------------------------------------
cuda_add_executable(renderer main.cu)
target_link_libraries(renderer ${OpenCV_LIBS})