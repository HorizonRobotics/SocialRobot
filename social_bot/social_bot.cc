// Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/SensorsIface.hh>

#include <mutex>  // NOLINT

namespace py = pybind11;

namespace social_bot {

class Observation;
class Action;

class Agent {
  gazebo::physics::ModelPtr model_;

 public:
  explicit Agent(gazebo::physics::ModelPtr model) : model_(model) {}
  Observation* Sense();
  std::vector<std::string> GetJointNames() {
    std::vector<std::string> names;
    names.reserve(model_->GetJointCount());
    for (auto joint : model_->GetJoints()) {
      names.push_back(joint->GetScopedName());
      std::cout << " joint name: " << names.back() << std::endl;
    }
    return names;
  }
  bool TakeAction(const std::map<std::string, double>& forces) {
    auto controller = model_->GetJointController();
    for (const auto& name2force : forces) {
      bool ret = controller->SetForce(name2force.first, name2force.second);
      if (!ret) {
        std::cout << "Cannot find joint '" << name2force.first << "'"
                  << " in  Agent '" << model_->GetName() << "'" << std::endl;
        return false;
      }
    }
  }

  // return ((x,y,z), (roll, pitch, yaw))
  std::tuple<std::tuple<double, double, double>,
             std::tuple<double, double, double>>
  GetPose() const {
    auto pose = model_->WorldPose();
    auto euler = pose.Rot().Euler();
    return std::make_tuple(
        std::make_tuple(pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()),
        std::make_tuple(euler.X(), euler.Y(), euler.Z()));
  }
};

class World {
  gazebo::physics::WorldPtr world_;

 public:
  explicit World(gazebo::physics::WorldPtr world) : world_(world) {}
  std::unique_ptr<Agent> GetAgent(const std::string& name) {
    if (name.empty()) {
      for (auto model : world_->Models()) {
        if (model->GetJointCount() > 0) {
          return std::make_unique<Agent>(model);
        }
      }
      return nullptr;
    }
    return std::make_unique<Agent>(world_->ModelByName(name));
  }
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
  static std::once_flag flag;
  std::call_once(flag, [&args]() {
    gazebo::common::Console::SetQuiet(false);
    gazebo::setupServer(args);
  });
}

void StartSensors() {
  static std::once_flag flag;
  std::call_once(flag, []() { gazebo::sensors::run_threads(); });
}

std::unique_ptr<World> NewWorldFromString(const std::string& std_string);

std::unique_ptr<World> NewWorldFromFile(const std::string& world_file) {
  gazebo::physics::WorldPtr world = gazebo::loadWorld(world_file);
  for (auto model : world->Models()) {
    std::cout << "Model: " << model->GetName() << std::endl;
    for (auto joint : model->GetJoints()) {
      std::cout << "Joint: " << joint->GetScopedName() << std::endl;
    }
  }
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
           py::arg("fileName"))
      .def("get_agent",
           &World::GetAgent,
           "Get an agent by name",
           py::arg("name") = "");

  py::class_<Agent>(m, "Agent")
      .def("get_joint_names",
           &Agent::GetJointNames,
           "Get the names of all the joints of this agent")
      .def("take_action",
           &Agent::TakeAction,
           "Take action for this agent, forces is a dictionary from joint name "
           "to force."
           " Return false if some joint name cannot be found",
           py::arg("forces"))
      .def("get_pose",
           &Agent::GetPose,
           "Get ((x,y,z), (roll, pitch, yaw)) of the agent");
}

}  // namespace social_bot
