cmake_minimum_required (VERSION 3.28)
project(OptixPathTracer CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.3 REQUIRED)
message("CUDAToolkit_INCLUDE_DIRS    = " ${CUDAToolkit_INCLUDE_DIRS})
message("CUDAToolkit_LIBRARY_DIR     = " ${CUDAToolkit_LIBRARY_DIR})
message("CUDAToolkit_NVCC_EXECUTABLE = " ${CUDAToolkit_NVCC_EXECUTABLE})

find_package(OpenEXR REQUIRED)
find_package(Imath REQUIRED)

message("CMAKE_CURRENT_SOURCE_DIR    = " ${CMAKE_CURRENT_SOURCE_DIR})
message("CMAKE_CURRENT_BINARY_DIR    = " ${CMAKE_CURRENT_BINARY_DIR})
message("CMAKE_SOURCE_DIR            = " ${CMAKE_SOURCE_DIR})
message("CMAKE_BINARY_DIR            = " ${CMAKE_BINARY_DIR})

set(SOURCES
    src/main.cpp
    src/camera.cpp
    src/gltf_loader.cpp
    src/gl_utils.cpp
    src/gl_window.cpp
    src/light.cpp
    src/main_window.cpp
    src/mesh.cpp
    src/renderer.cpp
    src/scene_desc.cpp
    src/tests.cpp
    src/stb_image_impl.cpp
    src/tiny_gltf_impl.cpp
    src/utils.cpp
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

#set_source_files_properties(src/test_cuda.cpp PROPERTIES LANGUAGE CUDA)

set(EXECUTABLE_NAME optix_pt)

add_executable(${EXECUTABLE_NAME} ${SOURCES})

target_compile_definitions(${EXECUTABLE_NAME} PRIVATE
    GLM_ENABLE_EXPERIMENTAL
    NOMINMAX)

target_include_directories(${EXECUTABLE_NAME} PRIVATE
    include
    external
    external/glm
    external/imgui
    external/imgui/backends
    external/rapidjson
    external/rapidjson/include
    external/tinygltf
    external/tinyobjloader
    ${CUDAToolkit_INCLUDE_DIRS}
    $ENV{OPTIX_PATH_7_1_0}/include
    $ENV{OPTIX_PATH_7_1_0}/SDK
)

target_link_directories(${EXECUTABLE_NAME} PRIVATE
    external/lib
    ${CUDAToolkit_LIBRARY_DIR}
)

target_link_libraries(${EXECUTABLE_NAME}
    cuda
    cudart
    glew32
    OpenEXR::OpenEXR
)

# May be needed if static libraries are used.
#set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/glfw EXCLUDE_FROM_ALL)
target_link_libraries(${EXECUTABLE_NAME} glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} OpenGL::GL)

#
# Create symbolic links to the scenes directory.
#

add_custom_target(symlinks_target
    COMMAND ${CMAKE_COMMAND} -E echo "Creating symbolic links to scenes directory ..."
    COMMAND ${CMAKE_COMMAND} -E create_symlink $ENV{SCENES_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/scenes

    # Create a symbolic link in the current binary directory.
    COMMAND ${CMAKE_COMMAND} -E create_symlink $ENV{SCENES_DIR} scenes
)

add_dependencies(${EXECUTABLE_NAME} symlinks_target)

#
# CUDA kernels
#

set(CUDA_LIBRARY_SOURCES
    src/stb_image_impl.cpp
    cu/kernels.cu
)

add_library(cuda_kernels SHARED ${CUDA_LIBRARY_SOURCES})

target_include_directories(cuda_kernels PRIVATE
    include
    external
    $ENV{OPTIX_PATH_7_1_0}/include
    $ENV{OPTIX_PATH_7_1_0}/SDK
    $ENV{OpenEXR_DIR}/include/Imath
    $ENV{OpenEXR_DIR}/include/OpenEXR
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    #CUDA_SEPARABLE_COMPILATION ON
)

target_link_directories(cuda_kernels PRIVATE
    $ENV{OpenEXR_DIR}/lib
)

target_link_libraries(cuda_kernels PRIVATE
    OpenEXR-3_2
)

target_link_libraries(${EXECUTABLE_NAME} cuda_kernels)

#
# Compile CUDA to PTX
#

set(CUDA_SOURCES
    cu/device_programs.cu
    cu/light_sampling.cu
    cu/path_tracer.cu
    cu/path_tracer_naive.cu
)

# + Strategy 1

# Optional
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#set(CMAKE_CUDA_FLAGS "-rdc=true -cudart=none -m64")

#add_library(generate_ptx OBJECT
#    ${CUDA_SOURCES}
#)

#set_target_properties(generate_ptx PROPERTIES CUDA_ARCHITECTURES 52)
#set_target_properties(generate_ptx PROPERTIES CUDA_PTX_COMPILATION ON)

#target_include_directories(generate_ptx PRIVATE
#    include
#    $ENV{OPTIX_PATH_7_1_0}/include
#    $ENV{OPTIX_PATH_7_1_0}/SDK
#)

#add_custom_target(copy_ptx_files
#    COMMAND ${CMAKE_COMMAND} -E echo "Copying PTX files to ptx subdirectory ..."
#    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/generate_ptx.dir/cu ${CMAKE_CURRENT_SOURCE_DIR}/ptx
#)

#add_dependencies(copy_ptx_files generate_ptx)
#add_dependencies(${EXECUTABLE_NAME} copy_ptx_files)

# Copy resources to build directory.

#add_custom_target(copy_files
#    COMMAND ${CMAKE_COMMAND} -E echo "Copying files to output directory ..."
#    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/configs ${CMAKE_CURRENT_BINARY_DIR}/configs
#    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
#    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/ptx ${CMAKE_CURRENT_BINARY_DIR}/ptx
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/external/bin/glew32.dll ${CMAKE_CURRENT_BINARY_DIR}
#)

#add_dependencies(${EXECUTABLE_NAME} copy_files)

# - Strategy 1

# + Strategy 2

set(CUDA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{OPTIX_PATH_7_1_0}/include
    $ENV{OPTIX_PATH_7_1_0}/SDK
)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})

# Optional
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 52)

# Set compilation flags common to all builds.
set(NVCC_COMMAND_LINE_PARAMS "-rdc=true;-cudart=none;-m64;-arch=sm_${CMAKE_CUDA_ARCHITECTURES};-std=c++${CMAKE_CUDA_STANDARD};-ptx")

# Add specific debug/release compilation flags.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(NVCC_COMMAND_LINE_PARAMS "${NVCC_COMMAND_LINE_PARAMS};-lineinfo")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(NVCC_COMMAND_LINE_PARAMS "${NVCC_COMMAND_LINE_PARAMS}")
endif()

macro(add_dep_target target_name in_cu out_d target_ptx)
    # Regenerate the dependency file if the cu file changes.
    add_custom_command(
        OUTPUT ${out_d}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating ${out_d}"
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -M -MF ${out_d} -MT ${target_ptx} "$<LIST:TRANSFORM,${CUDA_INCLUDE_DIRS},PREPEND,-I>" ${in_cu}
        DEPENDS ${in_cu}
        VERBATIM
        COMMAND_EXPAND_LISTS
    )
    
    add_custom_target(${target_name} DEPENDS ${out_d})
    add_dependencies(${EXECUTABLE_NAME} ${target_name})
endmacro()

macro(add_target target_name in_cu in_d out_ptx)
    # Regenerate the ptx file if either the cu or any file listed in the dependency file changes.
    add_custom_command(
        OUTPUT ${out_ptx}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating ${out_ptx}"
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -odir ${CMAKE_CURRENT_BINARY_DIR}/ptx ${NVCC_COMMAND_LINE_PARAMS} "$<LIST:TRANSFORM,${CUDA_INCLUDE_DIRS},PREPEND,-I>" ${in_cu}
        DEPENDS ${in_cu}
        DEPFILE ${in_d}
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(${target_name} DEPENDS ${out_ptx})
    add_dependencies(${EXECUTABLE_NAME} ${target_name})
endmacro()

foreach(FILE ${CUDA_SOURCES})
    cmake_path(GET FILE STEM file_stem)

    # Generate dependency file.
    add_dep_target(generate_${file_stem}_d ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${CMAKE_CURRENT_BINARY_DIR}/ptx_deps/${file_stem}.d ${CMAKE_CURRENT_BINARY_DIR}/ptx/${file_stem}.ptx)

    # Generate ptx file.
    add_target(generate_${file_stem}_ptx ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${CMAKE_CURRENT_BINARY_DIR}/ptx_deps/${file_stem}.d ${CMAKE_CURRENT_BINARY_DIR}/ptx/${file_stem}.ptx)
endforeach()

add_custom_target(copy_ptx_files
    COMMAND ${CMAKE_COMMAND} -E echo "Copying PTX files to ptx subdirectory ..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_BINARY_DIR}/ptx ${CMAKE_CURRENT_SOURCE_DIR}/ptx
)

foreach(FILE ${CUDA_SOURCES})
    cmake_path(GET FILE STEM file_stem)
    add_dependencies(copy_ptx_files generate_${file_stem}_ptx)
endforeach()

add_dependencies(${EXECUTABLE_NAME} copy_ptx_files)

# Copy resources to build directory.

add_custom_target(copy_files
    COMMAND ${CMAKE_COMMAND} -E echo "Copying files to output directory ..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/configs ${CMAKE_CURRENT_BINARY_DIR}/configs
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/resources ${CMAKE_CURRENT_BINARY_DIR}/resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${CMAKE_CURRENT_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/external/bin/glew32.dll ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $ENV{OpenEXR_DIR}/bin/OpenEXRCore-3_2.dll ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $ENV{OpenEXR_DIR}/bin/OpenEXR-3_2.dll ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $ENV{OpenEXR_DIR}/bin/Iex-3_2.dll ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $ENV{OpenEXR_DIR}/bin/IlmThread-3_2.dll ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $ENV{OpenEXR_DIR}/bin/Imath-3_1.dll ${CMAKE_CURRENT_BINARY_DIR}
)

#foreach(FILE ${CUDA_SOURCES})
#    cmake_path(GET FILE STEM file_stem)
#    add_dependencies(copy_files generate_${file_stem}_ptx)
#endforeach()

add_dependencies(${EXECUTABLE_NAME} copy_files)

# - Strategy 2

#
# Clean
#

set_property(TARGET ${EXECUTABLE_NAME} APPEND PROPERTY ADDITIONAL_CLEAN_FILES configs)
set_property(TARGET ${EXECUTABLE_NAME} APPEND PROPERTY ADDITIONAL_CLEAN_FILES shaders)
set_property(TARGET ${EXECUTABLE_NAME} APPEND PROPERTY ADDITIONAL_CLEAN_FILES glew32.dll)
set_property(TARGET ${EXECUTABLE_NAME} APPEND PROPERTY ADDITIONAL_CLEAN_FILES scenes)
#set_property(TARGET ${EXECUTABLE_NAME} APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ptx)

#
# Install
#

install(TARGETS ${EXECUTABLE_NAME} DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/configs DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ptx DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION .)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/external/bin/glew32.dll DESTINATION .)

cmake_path(SET out_path $ENV{SCENES_DIR})
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink ${out_path} ${CMAKE_INSTALL_PREFIX}/scenes)")
