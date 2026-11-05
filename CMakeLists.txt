cmake_minimum_required(VERSION 3.20)
project(sfm_snr_mfsr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
if(MSVC)
    add_compile_options(/O2 /W4 /EHsc /utf-8)
else()
    add_compile_options(-O3 -Wall -Wextra -pedantic)
endif()

# Threading
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional dynamic loader library on POSIX systems
set(EXTRA_LIBS Threads::Threads)
if(UNIX AND NOT APPLE)
    find_library(LIB_DL dl)
    if(LIB_DL)
        list(APPEND EXTRA_LIBS ${LIB_DL})
    endif()
endif()

# Header-only library interface
add_library(sfm_sr_core INTERFACE)
target_include_directories(sfm_sr_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sfm_sr_core INTERFACE ${EXTRA_LIBS})

# Main executable
add_executable(sfm_mvs_sr src/main.cpp)
target_link_libraries(sfm_mvs_sr PRIVATE sfm_sr_core)

# Single-file standalone distribution target (for convenience)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sfm_mvs_sr_single_file.cpp")
    add_executable(sfm_mvs_sr_single sfm_mvs_sr_single_file.cpp)
    target_link_libraries(sfm_mvs_sr_single PRIVATE ${EXTRA_LIBS})
endif()

# Install rules
install(TARGETS sfm_mvs_sr DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
