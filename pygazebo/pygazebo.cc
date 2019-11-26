// Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <gazebo/common/PID.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/WorldState.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/rendering/DepthCamera.hh>
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/sensors/ContactSensor.hh>
#include <gazebo/sensors/DepthCameraSensor.hh>
#include <gazebo/sensors/Sensor.hh>
#include <gazebo/sensors/SensorManager.hh>
#include <gazebo/sensors/SensorsIface.hh>
#include <gazebo/util/LogRecord.hh>
#include <mutex>  // NOLINT
#include <sstream>

namespace py = pybind11;

namespace social_bot {

bool gazebo_initialized = false;
bool gazebo_sensor_initialized = false;

class Observation;

class CameraObservation {
 public:
  CameraObservation(size_t width,
                    size_t height,
                    size_t depth,
                    uint8_t* img_data,
                    float* depth_data)
      : width_(width),
        height_(height),
        depth_(depth),
        img_data_(img_data),
        depth_data_(depth_data) {
    if (depth_data_) {
      float* buf = new float[height_ * width_ * (depth_ + 1)];
      for (size_t i = 0; i < height_ * width_; i++) {
        for (size_t j = 0; j < depth_; j++) {
          buf[i * (depth_ + 1) + j] = img_data_[i * depth_ + j];
        }
        buf[i * (depth_ + 1) + depth_] = depth_data_[i];
      }
      this->data_ = buf;
    } else {
      this->data_ = NULL;
    }
  }

  uint8_t* img_data() const { return img_data_; }
  float* depth_data() const { return depth_data_; }
  size_t width() const { return width_; }
  size_t height() const { return height_; }
  size_t depth() const { return depth_; }
  float* data() const { return data_; }

  ~CameraObservation() {
    if (this->data_) {
      delete[] this->data_;
      this->data_ = NULL;
    }
  }

 private:
  size_t width_, height_, depth_;
  uint8_t* img_data_;
  float* depth_data_;
  float* data_;
};

class JointState {
 public:
  // degree of freedom
  unsigned int GetDOF() const { return dof_; }
  const std::vector<double>& GetPositions() const { return positions_; }
  const std::vector<double>& GetVelocities() const { return velocities_; }
  const std::vector<double>& GetEffortLimits() const { return effort_limits_; }

  explicit JointState(unsigned int dof) : dof_(dof) {}
  void SetVelocities(const std::vector<double>& v) {
    velocities_.assign(v.begin(), v.end());
  }
  void SetPositions(const std::vector<double>& pos) {
    positions_.assign(pos.begin(), pos.end());
  }
  void SetEffortLimits(const std::vector<double> limits) {
    effort_limits_.assign(limits.begin(), limits.end());
  }

 private:
  unsigned int dof_;
  std::vector<double> positions_;
  std::vector<double> velocities_;
  std::vector<double> effort_limits_;
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
  // return ((Linear velocity x, Linear velocity y, Linear velocity z),
  //         (Angular velocity x, Angular velocity y, Angular velocity z))
  auto GetVelocities() const {
    auto lin_ver = model_->WorldLinearVel();
    auto ang_ver = model_->WorldAngularVel();
    return std::make_tuple(
        std::make_tuple(lin_ver.X(), lin_ver.Y(), lin_ver.Z()),
        std::make_tuple(ang_ver.X(), ang_ver.Y(), ang_ver.Z()));
  }
  void Reset() { model_->ResetPhysicsStates(); }
};

class Agent : public Model {
 private:
  std::map<std::string, gazebo::sensors::CameraSensorPtr> cameras_;
  std::map<std::string, gazebo::sensors::ContactSensorPtr> contacts_;
  std::map<std::string, int> joints_control_type_;
  enum joints_control_type_def_ {
    control_type_force_ =
        0,  // value in std::map<std::string, int> is 0 by default
    control_type_velocity_ = 1,
    control_type_position_ = 2
  };

 public:
  explicit Agent(gazebo::physics::ModelPtr model) : Model(model) {}
  Observation* Sense();
  std::vector<std::string> GetJointNames() {
    std::vector<std::string> names;
    names.reserve(model_->GetJointCount());
    for (auto joint : model_->GetJoints()) {
      names.push_back(joint->GetScopedName());
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
    std::vector<double> positions, velocities, limits;
    positions.reserve(dof);
    velocities.reserve(dof);
    limits.reserve(dof);

    for (unsigned int i = 0; i < dof; i++) {
      positions.push_back(joint->Position(i));
      velocities.push_back(joint->GetVelocity(i));
      limits.push_back(joint->GetEffortLimit(i));
    }

    state.SetPositions(positions);
    state.SetVelocities(velocities);
    state.SetEffortLimits(limits);
    return state;
  }

  Pose GetLinkPose(const std::string& link_name) const {
    auto link = model_->GetLink(link_name);

    if (!link) {
      std::cerr << "unable to find link: " << link_name << std::endl;
    }

    auto pose = link->WorldPose();
    auto euler = pose.Rot().Euler();
    return std::make_tuple(
        std::make_tuple(pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()),
        std::make_tuple(euler.X(), euler.Y(), euler.Z()));
  }

  void SetLinkPose(const std::string& link_name, const Pose& pose) const {
    auto loc = std::get<0>(pose);
    auto rot = std::get<1>(pose);
    auto link = model_->GetLink(link_name);

    if (!link) {
      std::cerr << "unable to find link: " << link_name << std::endl;
    }

    ignition::math::Pose3d pose3d(std::get<0>(loc),
                                  std::get<1>(loc),
                                  std::get<2>(loc),
                                  std::get<0>(rot),
                                  std::get<1>(rot),
                                  std::get<2>(rot));
    link->SetWorldPose(pose3d);
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

  // given a sensor names, it will return all pairs of link names
  // which are in collisions detected by the touch sensor at that instant.
  std::set<std::tuple<std::string, std::string>> GetCollisions(
      const std::string& contact_sensor_name) {
    std::set<std::tuple<std::string, std::string>> collisions;

    auto it = contacts_.find(contact_sensor_name);

    if (it == contacts_.end()) {
      gazebo::sensors::ContactSensorPtr sensor =
          std::dynamic_pointer_cast<gazebo::sensors::ContactSensor>(
              gazebo::sensors::get_sensor(contact_sensor_name));

      if (!sensor) {
        std::cerr << "unable to find sensor: " << contact_sensor_name
                  << std::endl;
      }

      auto ret = contacts_.insert(std::make_pair(contact_sensor_name, sensor));
      it = ret.first;
    }

    auto contacts = it->second->Contacts();

    for (int i = 0; i < contacts.contact_size(); i++) {
      auto contact = contacts.contact(i);

      collisions.insert(
          std::make_tuple(contact.collision1(), contact.collision2()));
    }

    return collisions;
  }

  CameraObservation GetCameraObservation(const std::string& sensor_scope_name) {
    auto it = cameras_.find(sensor_scope_name);

    if (it == cameras_.end()) {
      gazebo::sensors::CameraSensorPtr sensor =
          std::dynamic_pointer_cast<gazebo::sensors::CameraSensor>(
              gazebo::sensors::get_sensor(sensor_scope_name));

      if (!sensor) {
        std::cerr << "unable to find sensor: " << sensor_scope_name
                  << std::endl;
      }

      auto ret = cameras_.insert(std::make_pair(sensor_scope_name, sensor));
      it = ret.first;
    }

    auto sensor = it->second;
    auto camera = sensor->Camera();

    const float* depthData = NULL;
    auto sensorType = sensor->Type();
    if (sensorType == "depth") {
      // gazebo::sensors::DepthCameraSensorPtr depthSensor =
      //   std::dynamic_pointer_cast<gazebo::sensors::DepthCameraSensor>(sensor);
      // depthData = depthSensor->DepthData(); // it's always null
      gazebo::rendering::DepthCamera* depthCamera =
          dynamic_cast<gazebo::rendering::DepthCamera*>(camera.get());
      depthData = depthCamera->DepthData();
      if (depthData == NULL) {
        std::cerr << "Depth data null" << std::endl;
      }
    } else {
      // ignore logical_camera, multicamera, wideanglecamera
    }

    return CameraObservation(
        camera->ImageWidth(),
        camera->ImageHeight(),
        camera->ImageDepth(),
        const_cast<uint8_t*>(
            reinterpret_cast<const uint8_t*>(sensor->ImageData())),
        const_cast<float*>(depthData));
  }

  bool TakeAction(const std::map<std::string, double>& controls) {
    auto controller = model_->GetJointController();
    for (const auto& name2control : controls) {
      bool ret;
      int control_type = joints_control_type_[name2control.first];
      switch (control_type) {
        case control_type_force_:
          ret = controller->SetForce(name2control.first, name2control.second);
          break;
        case control_type_velocity_:
          ret = controller->SetVelocityTarget(name2control.first,
                                              name2control.second);
          break;
        case control_type_position_:
          ret = controller->SetPositionTarget(name2control.first,
                                              name2control.second);
          break;
        default:
          std::cerr << "Unknown control type '" << control_type << "'"
                    << " in joint '" << name2control.first << "'" << std::endl;
          return false;
      }
      if (!ret) {
        std::cerr << "Cannot find joint '" << name2control.first << "'"
                  << " in  Agent '" << model_->GetName() << "'" << std::endl;
        return false;
      }
    }
  }

  void SetPIDController(const std::string& joint_name,
                        const std::string& pid_control_type,
                        double p,
                        double i,
                        double d,
                        double i_max,
                        double max_force) {
    auto pid = gazebo::common::PID(p * max_force, i * max_force, d * max_force);
    pid.SetCmdMax(max_force);
    pid.SetCmdMin(-max_force);
    pid.SetIMax(max_force * i_max);  // limit i term
    auto controller = model_->GetJointController();
    if (pid_control_type == "force") {
      joints_control_type_[joint_name] = control_type_force_;
    } else if (pid_control_type == "velocity") {
      joints_control_type_[joint_name] = control_type_velocity_;
      controller->SetVelocityPID(joint_name, pid);
    } else if (pid_control_type == "position") {
      joints_control_type_[joint_name] = control_type_position_;
      controller->SetPositionPID(joint_name, pid);
    } else {
      std::cerr << "Unknown PID type '" << pid_control_type << "'" << std::endl;
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

  std::string ModelListInfo() {
    std::stringstream ss;
    for (auto model : world_->Models()) {
      ss << "Model: " << '"' << model->GetName() << '"' << std::endl;
    }
    return ss.str();
  }

  std::string Info() const {
    std::stringstream ss;
    ss << " ==== world info ==== " << std::endl;
    gazebo::physics::WorldState world_state(world_);
    ss << " world state: " << world_state << std::endl;
    for (auto model : world_->Models()) {
      ss << "Model: " << model->GetName() << std::endl;
      for (auto link : model->GetLinks()) {
        ss << "Link: " << link->GetScopedName() << std::endl;
      }

      for (auto joint : model->GetJoints()) {
        ss << "Joint: " << joint->GetScopedName() << std::endl;
        ss << "DOF: " << joint->DOF() << std::endl;

        for (int i = 0; i < joint->DOF(); i++) {
          ss << "Position " << i << ":" << joint->Position(i) << std::endl;
          ss << "Velocity " << i << ":" << joint->GetVelocity(i) << std::endl;
        }
      }
    }

    gazebo::sensors::SensorManager* mgr =
        gazebo::sensors::SensorManager::Instance();

    for (auto sensor : mgr->GetSensors()) {
      ss << " sensor name: " << sensor->Name()
         << ", scoped name: " << sensor->ScopedName() << std::endl;
    }

    ss << " === the end of world === " << std::endl;
    return ss.str();
  }

  void InsertModelFromSdfString(const std::string& sdfString) {
    sdf::SDF sdf;
    sdf.SetFromString(sdfString);
    world_->InsertModelSDF(sdf);
  }

  void Reset() { world_->Reset(); }
};

void Initialize(const std::vector<std::string>& args,
                int port = 0,
                bool quiet = false) {
  if (port != 0) {
    std::string uri = "localhost:" + std::to_string(port);
    setenv("GAZEBO_MASTER_URI", uri.c_str(), 1);
  }
  if (!gazebo_initialized) {
    gazebo::common::Console::SetQuiet(quiet);
    gazebo::setupServer(args);
    // gazebo::runWorld uses World::RunLoop(). RunLoop() starts LogWorker()
    // every time. LogWorker will always store something in the buffer when
    // it is started. But we don't have a running LogRecord to collect all
    // these data. This cause memory usage keep increasing.
    // So we need to start LogRecord to collect data generated by RunLoop()
    // via LogWorker
    {
      gazebo::util::LogRecordParams params;
      params.period = 1e300;  // In fact, we don't need to do logging.
      gazebo::util::LogRecord::Instance()->Init("pygazebo");
      gazebo::util::LogRecord::Instance()->Start(params);
      gazebo::util::LogRecord::Instance()->Stop();
    }
  }
  gazebo_initialized = true;
}

void CloseWithoutModelbaseFini() {
  gazebo::physics::stop_worlds();
  gazebo::sensors::stop();
  gazebo::util::LogRecord::Instance()->Stop();
  gazebo::transport::stop();

  gazebo::transport::fini();
  gazebo::physics::fini();
  gazebo::sensors::fini();

  gazebo_initialized = false;
  gazebo_sensor_initialized = false;
}

void Close() {
  gazebo::shutdown();
  gazebo_initialized = false;
  gazebo_sensor_initialized = false;
}

void StartSensors() {
  if (!gazebo_sensor_initialized) {
    gazebo::sensors::run_threads();
    gazebo_sensor_initialized = true;
  }
}

std::string WorldSDF(const std::string& world_file) {
  sdf::SDFPtr worldSDF(new sdf::SDF());
  sdf::init(worldSDF);
  sdf::readFile(world_file, worldSDF);
  return worldSDF->ToString();
}

std::unique_ptr<World> NewWorldFromString(const std::string& std_string) {
  gazebo::physics::WorldPtr world;
  sdf::SDFPtr worldSDF(new sdf::SDF);
  worldSDF->SetFromString(std_string);
  sdf::ElementPtr worldElem = worldSDF->Root()->GetElement("world");
  world = gazebo::physics::create_world();
  gazebo::physics::load_world(world, worldElem);
  gazebo::physics::init_world(world);
  gazebo::sensors::run_once(true);
  StartSensors();
  return std::make_unique<World>(world);
}

std::unique_ptr<World> NewWorldFromFile(const std::string& world_file) {
  gazebo::physics::WorldPtr world = gazebo::loadWorld(world_file);
  gazebo::sensors::run_once(true);
  StartSensors();
  return std::make_unique<World>(world);
}

std::unique_ptr<World> NewWorldFromFileWithAgent(
    const std::string& world_file, const std::string& agent_name) {
  gazebo::physics::WorldPtr world = gazebo::loadWorld(world_file);
  world->InsertModelFile(agent_name);
  world->UpdateStateSDF();
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
        py::arg("args") = std::vector<std::string>(),
        py::arg("port") = 0,
        py::arg("quiet") = false);

  m.def("close", &Close, "Shutdown everything of gazebo");

  m.def("close_without_model_base_fini",
        &CloseWithoutModelbaseFini,
        "A customized close function without execute ModelbaseFini"
        "For some unknwon reason, ModelbaseFini() in the gazebo.shutdown() "
        "makes the"
        "process fail to exit when the environment is wrapped with process");

  m.def("world_sdf",
        &WorldSDF,
        "Read a world sdf from .world file",
        py::arg("world_file"));

  // Global functions
  m.def("new_world_from_string",
        &NewWorldFromString,
        "Create a world from sdf string",
        py::arg("std_string"));

  m.def("new_world_from_file",
        &NewWorldFromFile,
        "Create a world from .world file",
        py::arg("world_file"));

  m.def("new_world_from_file_with_agent",
        &NewWorldFromFileWithAgent,
        "Create a world from .world file, and insert the agent model in the "
        "mean time",
        py::arg("world_file"),
        py::arg("agent_name"));

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
      .def("model_list_info", &World::ModelListInfo, "get model list")
      .def("info", &World::Info, "get info for the world");

  py::class_<Model>(m, "Model")
      .def("get_pose",
           &Model::GetPose,
           "Get ((x,y,z), (roll, pitch, yaw)) of the agent")
      .def("set_pose",
           &Model::SetPose,
           "Set ((x,y,z), (roll, pitch, yaw)) of the agent")
      .def("get_velocities",
           &Model::GetVelocities,
           "Get ((Linear velocity x, Linear velocity y, Linear velocity z),"
           "(Angular velocity x, Angular velocity y, Angular velocity z))"
           "of the model")
      .def("reset",
           &Model::Reset,
           "Resets the pose and velocities of the model");

  py::class_<JointState>(m, "JointState")
      .def(py::init<unsigned int>())
      .def("get_positions", &JointState::GetPositions, "get joint positions")
      .def("get_velocities", &JointState::GetVelocities, "get joint velocities")
      .def("get_effort_limits",
           &JointState::GetEffortLimits,
           "get joint effort limits")
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
        if (m.depth_data()) {
          return py::buffer_info(m.data(),
                                 sizeof(float),
                                 py::format_descriptor<float>::format(),
                                 3,
                                 {m.height(), m.width(), m.depth() + 1},
                                 {sizeof(float) * m.width() * (m.depth() + 1),
                                  sizeof(float) * (m.depth() + 1),
                                  sizeof(float)});
        } else {
          return py::buffer_info(m.img_data(),
                                 sizeof(uint8_t),
                                 py::format_descriptor<uint8_t>::format(),
                                 3,
                                 {m.height(), m.width(), m.depth()},
                                 // memory layout for image
                                 {sizeof(uint8_t) * m.width() * m.depth(),
                                  sizeof(uint8_t) * m.depth(),
                                  sizeof(uint8_t)});
        }
      });

  py::class_<Agent, Model>(m, "Agent")
      .def("get_joint_names",
           &Agent::GetJointNames,
           "Get the names of all the joints of this agent")
      .def("take_action",
           &Agent::TakeAction,
           "Take action for this agent"
           "controls is a dictionary from joint name to control variables"
           "Action type is force by default. If pid controller is needed, you "
           "should call set_pid_controller first"
           "Return false if some joint name or control type cannot be found",
           py::arg("controls"))
      .def("set_pid_controller",
           &Agent::SetPIDController,
           "Set PID parameters for a joint if PID controleer are used"
           "joint_name is the name for the joint"
           "pid_control_type is the type of pid controller, either 'velocity', "
           "or 'position' "
           "p, i, and d are the parameters for the controller, i_max is the"
           "max value of i term. They will be scaled by max_force"
           "max_force is the limit of PID contorller output",
           py::arg("joint_name"),
           py::arg("pid_control_type") = "velocity",
           py::arg("p") = 0.02,
           py::arg("i") = 0.00,
           py::arg("d") = 0.01,
           py::arg("i_max") = 0.1,
           py::arg("max_force") = 2.0)
      .def("get_camera_observation",
           &Agent::GetCameraObservation,
           "Get observation from this camera sensor "
           "The shape is (H, W, C) and dtype is uint8 by default "
           "If it's a depth camera, then the last channel represents the depth "
           "data "
           "and its dtype is float32",
           py::arg("sensor_scope_name"))
      .def("get_joint_state", &Agent::GetJointState, py::arg("joint_name"))
      .def("get_link_pose", &Agent::GetLinkPose, py::arg("link_name"))
      .def("set_link_pose",
           &Agent::SetLinkPose,
           py::arg("link_name"),
           py::arg("link_pose"))
      .def("get_collisions",
           &Agent::GetCollisions,
           "return a set of tuples of collided links detected by the contact "
           "sensor",
           py::arg("contact_sensor_name"))
      .def("set_joint_state",
           &Agent::SetJointState,
           py::arg("joint_name"),
           py::arg("joint_state"))
      .def("reset", &Agent::Reset, "Reset to the initial state");
}

}  // namespace social_bot
