
/*
 * Copyright (C) 2020 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "DataGeneration.hh"

#include <ignition/gazebo/components/Pose.hh>
#include <ignition/plugin/Register.hh>

IGNITION_ADD_PLUGIN(
    data_generation::DataGeneration,
    ignition::gazebo::System,
    data_generation::DataGeneration::ISystemConfigure,
    data_generation::DataGeneration::ISystemPreUpdate)
using namespace data_generation;

//////////////////////////////////////////////////
DataGeneration::DataGeneration()
{
}

//////////////////////////////////////////////////
DataGeneration::~DataGeneration()
{
}

double fRand(double fMax)
{
    double f = (double)rand() / RAND_MAX;
    if(rand()%2 == 0){
      f *= -1;
    }
    return f * fMax;
}

ignition::math::Pose3d generateError(double pos_epsilon, double ang_epsilon)
{
  return ignition::math::Pose3d(fRand(pos_epsilon), fRand(pos_epsilon), fRand(pos_epsilon/4), fRand(ang_epsilon), fRand(ang_epsilon), fRand(ang_epsilon));
}

//////////////////////////////////////////////////
void DataGeneration::Configure(const ignition::gazebo::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    ignition::gazebo::EntityComponentManager &/*_ecm*/,
    ignition::gazebo::EventManager &/*_eventMgr*/)
{
    this->entity = _entity;
    auto sdfClone = _sdf->Clone();
}

//////////////////////////////////////////////////
void DataGeneration::PreUpdate(const ignition::gazebo::UpdateInfo &_info,
    ignition::gazebo::EntityComponentManager &_ecm){

      auto sec = std::chrono::duration_cast<std::chrono::seconds>(_info.simTime).count();
      if(sec > this->lastPositionChange){
        auto poseComp = _ecm.Component<ignition::gazebo::components::Pose>(this->entity);



        *poseComp = ignition::gazebo::components::Pose(this->handrailInspectPose + generateError(0.1, 0.05));

        _ecm.SetChanged(this->entity, ignition::gazebo::components::Pose::typeId,
          ignition::gazebo::ComponentState::OneTimeChange);
        this->lastPositionChange = sec;
      }

}
