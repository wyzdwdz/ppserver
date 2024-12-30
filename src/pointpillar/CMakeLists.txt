cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME pointpillar)

set(TARGET_SOURCES_PRIVATE pointpillar/lidar-backbone.cu
                           pointpillar/lidar-postprocess.cu
                           pointpillar/lidar-voxelization.cu
                           pointpillar/pointpillar.cpp
                           common/tensor.cu
                           common/tensorrt.cpp)

set(TARGET_SOURCES_PUBLIC  pointpillar/pillarscatter-kernel.cu
                           pointpillar/pointpillar-scatter.cpp)

set(THIRD_PARTY CUDA::toolkit 
                libnvinfer.so)

add_library(${TARGET_NAME} STATIC)

target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES_PRIVATE}
                              PUBLIC ${TARGET_SOURCES_PUBLIC})

target_include_directories(${TARGET_NAME} PUBLIC .)

target_link_libraries(${TARGET_NAME} PRIVATE ${THIRD_PARTY})

set(COMPILE_OPTIONS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    list(APPEND COMPILE_OPTIONS -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-declarations)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    list(APPEND COMPILE_OPTIONS -Wall)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    list(APPEND COMPILE_OPTIONS -Werror=all-warnings -Xcompiler -Wextra -Wall -Wno-deprecated-declarations)
endif()

target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)


#############################################################################################################


set(TARGET_NAME pluginpp)

set(TARGET_SOURCES pointpillar/pillarscatter-kernel.cu
                   pointpillar/pointpillar-scatter.cpp)

set(THIRD_PARTY CUDA::toolkit 
                libnvinfer.so)

add_library(${TARGET_NAME} SHARED)

target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE ${THIRD_PARTY})

set(COMPILE_OPTIONS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    list(APPEND COMPILE_OPTIONS -Wall -Wextra -Wno-missing-field-initializers -Wno-deprecated-declarations)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    list(APPEND COMPILE_OPTIONS -Wall)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    list(APPEND COMPILE_OPTIONS -Werror=all-warnings -Xcompiler -Wextra -Wall -Wno-deprecated-declarations)
endif()

target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

install(TARGETS ${TARGET_NAME}
        CONFIGURATIONS Release
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
