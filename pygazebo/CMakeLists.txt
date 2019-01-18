# Create libraries and executables

pybind11_add_module(pygazebo MODULE pygazebo.cc)
target_link_libraries(pygazebo PRIVATE ${GAZEBO_LIBRARIES} ${PYTHON_LIBRARIES})
