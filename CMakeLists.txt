cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(social_bot)

add_subdirectory(pybind11)

# Find packages
find_package(gazebo REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
message("python include ${PYTHON_INCLUDE_DIRS}")
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

# include appropriate directories
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

add_subdirectory(pygazebo)
