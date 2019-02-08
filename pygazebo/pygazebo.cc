// Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/WorldState.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/sensors/Sensor.hh>
#include <gazebo/sensors/SensorManager.hh>
#include <gazebo/sensors/SensorsIface.hh>
#include <gazebo/util/LogRecord.hh>

#include <mutex>  // NOLINT

namespace py = pybind11;

namespace social_bot {

class Observation;

class CameraObservation {
 public:
  CameraObservation(size_t width, size_t height, size_t depth, uint8_t* data)
      : width_(width), height_(height), depth_(depth), data_(data) {}

  uint8_t* data() const { return data_; }
  size_t width() const { return width_; }
  size_t height() const { return height_; }
  size_t depth() const { return depth_; }

 private:
  size_t width_, height_, depth_;
  uint8_t* data_;
};

class JointState {
 public:
  // degree of freedom
  unsigned int GetDOF() const { return dof_; }
  const std::vector<double>& GetPositions() const { return positions_; }
  const std::vector<double>& GetVelocities() const { return velocities_; }

  explicit JointState(unsigned int dof) : dof_(dof) {}
  void SetVelocities(const std::vector<double>& v) {
    velocities_.assign(v.begin(), v.end());
  }
  void SetPositions(const std::vector<double>& pos) {
    positions_.assign(pos.begin(), pos.end());
  }

 private:
  unsigned int dof_;
  std::vector<double> positions_;
  std::vector<double> velocities_;
};

class Action;

class Model {
 protected:
  gazebo::physics::ModelPtr model_;

 public:
  typedef std::tuple<std::tuple<double, double, double>,
                     std::tuple<double, double, double>>
      Pose;

  explicit Model(gazebo::physics::ModelPtr model) : model_(model) {}

  // return ((x,y,z), (roll, pitch, yaw))
  Pose GetPose() const {
    auto pose = model_->WorldPose();
    auto euler = pose.Rot().Euler();
    return std::make_tuple(
        std::make_tuple(pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()),
        std::make_tuple(euler.X(), euler.Y(), euler.Z()));
  }
  void SetPose(const Pose& pose) {
    auto loc = std::get<0>(pose);
    auto rot = std::get<1>(pose);
    ignition::math::Pose3d pose3d(std::get<0>(loc),
                                  std::get<1>(loc),
                                  std::get<2>(loc),
                                  std::get<0>(rot),
                                  std::get<1>(rot),
                                  std::get<2>(rot));
    model_->SetWorldPose(pose3d);
  }
};

class Agent : public Model {
 private:
  std::map<std::string, gazebo::sensors::CameraSensorPtr> cameras_;

 public:
  explicit Agent(gazebo::physics::ModelPtr model) : Model(model) {}
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

  JointState GetJointState(const std::string& joint_name) {
    auto joint = model_->GetJoint(joint_name);
    if (!joint) {
      std::cerr << "unable to find joint: " << joint_name << std::endl;
    }

    unsigned int dof = joint->DOF();
    JointState state(dof);
    std::vector<double> positions, velocities;
    positions.reserve(dof);
    velocities.reserve(dof);

    for (unsigned int i = 0; i < dof; i++) {
      positions.push_back(joint->Position(i));
      velocities.push_back(joint->GetVelocity(i));
    }

    state.SetPositions(positions);
    state.SetVelocities(velocities);
    return state;
  }

  void SetJointState(const std::string& joint_name,
                     const JointState& joint_state) {
    auto joint = model_->GetJoint(joint_name);
    if (!joint) {
      std::cerr << "unable to find joint: " << joint_name << std::endl;
    }

    if (joint->DOF() != joint_state.GetDOF()) {
      std::cerr << "joint degree of freedom not match" << std::endl;
    }

    for (int idx = 0; idx < joint->DOF(); idx++) {
      joint->SetPosition(idx, joint_state.GetPositions()[idx]);
      joint->SetVelocity(idx, joint_state.GetVelocities()[idx]);
    }
  }

  CameraObservation GetCameraObservation(const std::string& sensor_scope_name) {
    auto it = cameras_.find(sensor_scope_name);

    if (it == cameras_.end()) {
      gazebo::sensors::SensorManager* mgr =
          gazebo::sensors::SensorManager::Instance();

      gazebo::sensors::CameraSensorPtr sensor =
          std::dynamic_pointer_cast<gazebo::sensors::CameraSensor>(
              mgr->GetSensor(sensor_scope_name));

      if (!sensor) {
        std::cerr << "unable to find sensor: " << sensor_scope_name
                  << std::endl;
      }

      auto ret = cameras_.insert(std::make_pair(sensor_scope_name, sensor));
      it = ret.first;
    }

    auto sensor = it->second;
    auto camera = sensor->Camera();

    return CameraObservation(
        sensor->ImageWidth(),
        sensor->ImageHeight(),
        camera->ImageDepth(),
        // const char* => uint8_t*
        const_cast<uint8_t*>(
            reinterpret_cast<const uint8_t*>(sensor->ImageData())));
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

  void Reset() { model_->Reset(); }
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

  std::unique_ptr<Model> GetModel(const std::string& name) {
    return std::make_unique<Model>(world_->ModelByName(name));
  }

  void Step(int num_steps) {
    gazebo::runWorld(world_, num_steps);
    gazebo::sensors::run_once();
  }

  void InsertModelFile(const std::string& fileName) {
    world_->InsertModelFile(fileName);
  }

  void Info() const {
    std::cout << " ==== world info ==== " << std::endl;
    gazebo::physics::WorldState world_state(world_);
    std::cout << " world state: " << world_state << std::endl;
    for (auto model : world_->Models()) {
      std::cout << "Model: " << model->GetName() << std::endl;
      for (auto link : model->GetLinks()) {
        std::cout << "Link: " << link->GetScopedName() << std::endl;
      }

      for (auto joint : model->GetJoints()) {
        std::cout << "Joint: " << joint->GetScopedName() << std::endl;
        std::cout << "DOF: " << joint->DOF() << std::endl;

        for (int i = 0; i < joint->DOF(); i++) {
          std::cout << "Position " << i << ":" << joint->Position(i)
                    << std::endl;
          std::cout << "Velocity " << i << ":" << joint->GetVelocity(i)
                    << std::endl;
        }
      }
    }

    gazebo::sensors::SensorManager* mgr =
        gazebo::sensors::SensorManager::Instance();

    for (auto sensor : mgr->GetSensors()) {
      std::cout << " sensor name: " << sensor->Name()
                << ", scoped name: " << sensor->ScopedName() << std::endl;
    }

    std::cout << " === the end of world === " << std::endl;
  }

  void InsertModelFromSdfString(const std::string& sdfString) {
    sdf::SDF sdf;
    sdf.SetFromString(sdfString);
    world_->InsertModelSDF(sdf);
  }

  void Reset() { world_->Reset(); }
};

void Initialize(const std::vector<std::string>& args) {
  static std::once_flag flag;
  std::call_once(flag, [&args]() {
    gazebo::common::Console::SetQuiet(false);
    gazebo::setupServer(args);
    // gazebo::runWorld uses World::RunLoop(). RunLoop() starts LogWorker()
    // every time. LogWorker
    // will always store something in the buffer when it is started. But we
    // don't have a running
    // LogRecord to collect all these data. This cause memory usage keep
    // increasing.
    // So we need to start LogRecord to collect data generated by RunLoop() (via
    // LogWorker)
    {
      gazebo::util::LogRecordParams params;
      params.period = 1e300;  // In fact, we don't need to do logging.
      gazebo::util::LogRecord::Instance()->Start(params);
    }
  });
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

PYBIND11_MODULE(pygazebo, m) {
  m.doc() = "Gazebo python API";

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
           py::arg("name") = "")
      .def(
          "get_model", &World::GetModel, "Get a model by name", py::arg("name"))
      .def("reset", &World::Reset, "Reset the world")
      .def("info", &World::Info, "show debug info for the world");

  py::class_<Model>(m, "Model")
      .def("get_pose",
           &Model::GetPose,
           "Get ((x,y,z), (roll, pitch, yaw)) of the agent")
      .def("set_pose",
           &Model::SetPose,
           "Set ((x,y,z), (roll, pitch, yaw)) of the agent");

  py::class_<JointState>(m, "JointState")
      .def(py::init<unsigned int>())
      .def("get_positions", &JointState::GetPositions, "get joint positions")
      .def("get_velocities", &JointState::GetVelocities, "get joint velocities")
      .def("set_positions",
           &JointState::SetPositions,
           "set joint positions",
           py::arg("pos"))
      .def("set_velocities",
           &JointState::SetVelocities,
           "set joint velocities",
           py::arg("v"))
      .def("get_dof", &JointState::GetDOF, "get degree of freedoms");

  py::class_<CameraObservation>(m, "CameraObservation", py::buffer_protocol())
      .def_buffer([](CameraObservation& m) -> py::buffer_info {
        return py::buffer_info(m.data(),
                               sizeof(uint8_t),
                               py::format_descriptor<uint8_t>::format(),
                               3,
                               {m.height(), m.width(), m.depth()},
                               // memory layout for image
                               {sizeof(uint8_t) * m.width() * m.depth(),
                                sizeof(uint8_t) * m.depth(),
                                sizeof(uint8_t)});
      });

  py::class_<Agent, Model>(m, "Agent")
      .def("get_joint_names",
           &Agent::GetJointNames,
           "Get the names of all the joints of this agent")
      .def("take_action",
           &Agent::TakeAction,
           "Take action for this agent, forces is a dictionary from joint name "
           "to force."
           " Return false if some joint name cannot be found",
           py::arg("forces"))
      .def("get_camera_observation",
           &Agent::GetCameraObservation,
           py::arg("sensor_scope_name"))
      .def("get_joint_state", &Agent::GetJointState, py::arg("joint_name"))
      .def("set_joint_state",
           &Agent::SetJointState,
           py::arg("joint_name"),
           py::arg("joint_state"))
      .def("reset", &Agent::Reset, "Reset to the initial state");
}

}  // namespace social_bot
