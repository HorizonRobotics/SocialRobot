# Create libraries and executables

pybind11_add_module(social_bot MODULE social_bot.cc)
target_link_libraries(social_bot PRIVATE ${GAZEBO_LIBRARIES} ${PYTHON_LIBRARIES})

add_library(animated_box SHARED animated_box.cc)
target_link_libraries(animated_box ${GAZEBO_LIBRARIES})

add_executable(integrated_main integrated_main.cc)
target_link_libraries(integrated_main ${GAZEBO_LIBRARIES} pthread)

add_executable(independent_listener independent_listener.cc)
target_link_libraries(independent_listener ${GAZEBO_LIBRARIES} pthread)
