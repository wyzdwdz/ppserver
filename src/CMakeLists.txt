cmake_minimum_required(VERSION 3.22)

add_subdirectory(pointpillar)

set(TARGET_NAME ppserver)

set(TARGET_SOURCES main.cpp
                   predictor.cpp)

set(THIRD_PARTY Boost::boost
                spdlog::spdlog_header_only
                CLI11::CLI11
                pointpillar
                CUDA::toolkit)

set(COMPILE_OPTIONS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    list(APPEND COMPILE_OPTIONS -Wall -Wextra -Wno-unknown-pragmas)
    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
        list(APPEND COMPILE_OPTIONS -Wpedantic -Wno-static-in-inline)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    list(APPEND COMPILE_OPTIONS -Wall)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    list(APPEND COMPILE_OPTIONS -Werror=all-warnings -Xcompiler -Wextra -Wall -Wno-deprecated-declarations)
endif()

add_executable(${TARGET_NAME})

target_sources(${TARGET_NAME} PRIVATE ${TARGET_SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE ${THIRD_PARTY})

target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})

target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

install(TARGETS ${TARGET_NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
