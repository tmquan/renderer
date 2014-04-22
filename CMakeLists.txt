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
	include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
	# message("${CUDA_INCLUDE_DIRS}")
	# message("${CUDA_SDK_ROOT_DIR}")
	message("${CUDA_SDK_ROOT_DIR}/common/inc")

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
    # message("" ${OpesnCV_INCLUDE_DIR})
	include_directories(${OpenCV_INCLUDE_DIR})
else(OpenCV_FOUND)
    message("OPENCV is not installed on this system.")
endif()





# ----------------------------------------------------------------------------
# cuda_add_executable(renderer main.cu)
cuda_add_executable(renderer volumeRender_kernel.cu volumeRender.cpp)
target_link_libraries(renderer ${OpenCV_LIBS})




find_package(GLEW)
find_package(GLUT)
find_package(OpenGL)
############################################################################
if(GLEW_FOUND)
    message("GLEW is found on this system.")
else(GLEW_FOUND)
    message("GLEW is not installed on this system.")
	# set( GLEW_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/glew/include	)
endif()
############################################################################
if(GLUT_FOUND)
    message("GLUT is found on this system.")
else(GLUT_FOUND)
    message("GLUT is not installed on this system.")
	# set( GLUT_INCLUDE_DIR  ${CMAKE_SOURCE_DIR}/freeglut/include)
endif()
############################################################################
if(OPENGL_FOUND)
    message("OPENGL is found on this system.")
else(OPENGL_FOUND)
    message("OPENGL is not installed on this system.")
endif()
############################################################################
target_link_libraries(renderer ${OPENGL_LIBRARIES})
target_link_libraries(renderer ${GLEW_LIBRARIES})
target_link_libraries(renderer ${GLUT_LIBRARIES})
# ----------------------------------------------------------------------------
add_test(Test_Renderer renderer)