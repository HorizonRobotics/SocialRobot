// Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/SensorsIface.hh>
#include <mutex>  // NOLINT

namespace py = pybind11;

namespace social_bot {

class Observation;
class Action;

class Agent {
 public:
  Observation* Sense();
  void TackAction(Action* action);
};

// Create a new agent is world.
Agent* NewAgent(const std::string& world_name = "default");

class World {
  gazebo::physics::WorldPtr world_;

 public:
  explicit World(gazebo::physics::WorldPtr world) : world_(world) {}
  Agent* GetAgent(const std::string& name);
  void Step(int num_steps) { gazebo::runWorld(world_, num_steps); }
  void InsertModelFile(const std::string& fileName) {
    world_->InsertModelFile(fileName);
  }

  void InsertModelFromSdfString(const std::string& sdfString) {
    sdf::SDF sdf;
    sdf.SetFromString(sdfString);
    world_->InsertModelSDF(sdf);
  }
};

void Initialize(const std::vector<std::string>& args) {
  gazebo::common::Console::SetQuiet(false);
  static std::once_flag flag;
  std::call_once(flag, [&args]() { gazebo::setupServer(args); });
}

void StartSensors() {
  static std::once_flag flag;
  std::call_once(flag, []() { gazebo::sensors::run_threads(); });
}

std::unique_ptr<World> NewWorldFromString(const std::string& std_string);

std::unique_ptr<World> NewWorldFromFile(const std::string& world_file) {
  gazebo::physics::WorldPtr world = gazebo::loadWorld(world_file);
  gazebo::sensors::run_once(true);
  StartSensors();
  return std::make_unique<World>(world);
}

std::unique_ptr<World> NewWorld();

PYBIND11_MODULE(social_bot, m) {
  m.doc() = "social_bot python API";

  m.def("initialize",
        &Initialize,
        "Initialize",
        py::arg("args") = std::vector<std::string>());

  // Global functions
  m.def("new_world_from_file",
        &NewWorldFromFile,
        "Create a world from .world file",
        py::arg("world_file"));

  // World class
  py::class_<World>(m, "World")
      .def("step", &World::Step, "Run world for steps", py::arg("steps") = 1)
      .def("insertModelFromSdfString",
           &World::InsertModelFromSdfString,
           "Insert model from sdf string",
           py::arg("sdfString"))
      .def("insertModelFile",
           &World::InsertModelFile,
           "Insert model from file",
           py::arg("fileName"));
}

}  // namespace social_bot
