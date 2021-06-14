// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "SceneAttributes.h"
#include "esp/physics/RigidBase.h"
namespace esp {
namespace metadata {
namespace attributes {
// All keys must be lowercase
const std::map<std::string, esp::physics::MotionType>
    SceneObjectInstanceAttributes::MotionTypeNamesMap = {
        {"static", esp::physics::MotionType::STATIC},
        {"kinematic", esp::physics::MotionType::KINEMATIC},
        {"dynamic", esp::physics::MotionType::DYNAMIC},
};

SceneObjectInstanceAttributes::SceneObjectInstanceAttributes(
    const std::string& handle,
    const std::string& type)
    : AbstractAttributes(type, handle) {
  // default to unknown for object instances, to use attributes-specified
  // defaults
  setShaderType(static_cast<int>(ObjectInstanceShaderType::Unknown));

  // defaults to unknown/undefined
  setMotionType(static_cast<int>(esp::physics::MotionType::UNDEFINED));
  // set to no rotation
  setQuat("rotation", Mn::Quaternion(Mn::Math::IdentityInit));
  // defaults to unknown so that obj instances use scene instance setting
  setTranslationOrigin(
      static_cast<int>(managers::SceneInstanceTranslationOrigin::Unknown));
  // set default multiplicative scaling values
  setUniformScale(1.0f);
  setMassScale(1.0f);
}

SceneAOInstanceAttributes::SceneAOInstanceAttributes(const std::string& handle)
    : SceneObjectInstanceAttributes(handle, "SceneAOInstanceAttributes") {
  // set default fixed base value (only used for articulated object)
  setFixedBase(false);
}

const std::map<std::string, managers::SceneInstanceTranslationOrigin>
    SceneAttributes::InstanceTranslationOriginMap = {
        {"asset_local", managers::SceneInstanceTranslationOrigin::AssetLocal},
        {"com", managers::SceneInstanceTranslationOrigin::COM},
};

SceneAttributes::SceneAttributes(const std::string& handle)
    : AbstractAttributes("SceneAttributes", handle) {
  // defaults to no lights
  setLightingHandle(NO_LIGHT_KEY);
  // defaults to asset local
  setTranslationOrigin(
      static_cast<int>(managers::SceneInstanceTranslationOrigin::AssetLocal));
}

}  // namespace attributes
}  // namespace metadata
}  // namespace esp
