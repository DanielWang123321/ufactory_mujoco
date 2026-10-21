#!/usr/bin/env python3
"""Build and standardize MJCF models (model.xml & scene.xml) for all UFACTORY arms and grippers.

Generates complete, verified MJCF models with:
- Multi-material high-fidelity visual OBJs (group 2) and dedicated collision STLs (group 3)
- Accurate kinematics, mass, and inertial tensors from official URDF definitions
- Menagerie-compliant actuators, damping, and joint limits
- Contact exclusions for adjacent links to eliminate bearing collar overlaps
- Keyframes ('home' and 'ready') and tool flange attachment sites
- Ready-to-view scene.xml with skybox, groundplane, and cameras

Usage:
    python scripts/build_all_mjcf.py
"""

from __future__ import annotations

from pathlib import Path
import mujoco

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


SCENE_TEMPLATE = """<mujoco model="{name} scene">
  <include file="{xml_file}"/>

  <statistic center="{center}" extent="{extent}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="front" pos="1.2 0.8 0.9" quat="0.72 0 0.19 0.67"/>
    <camera name="side" pos="-0.3 1.6 0.8" quat="0.66 0.24 0.24 0.67"/>
  </worldbody>
</mujoco>
"""


def write_scene_file(target_dir: Path, name: str, xml_file: str, center="0.3 0 0.3", extent="0.8") -> None:
    scene_path = target_dir / "scene.xml"
    content = SCENE_TEMPLATE.format(name=name, xml_file=xml_file, center=center, extent=extent)
    scene_path.write_text(content, encoding="utf-8")
    print(f"  Wrote {scene_path}")


def build_xarm7() -> None:
    print("Building xArm 7...")
    target_dir = MODELS_DIR / "xarm7"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="xarm7">
  <!-- UFACTORY xArm 7 (1305 variant), high-fidelity multi-material visual OBJs
       with dedicated collision STLs. Actuator parameters follow Menagerie conventions. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="black" rgba="0.25 0.25 0.25 1" specular="0.2" shininess="0.1"/>
    <material name="gray" rgba="0.753 0.753 0.753 1"/>

    <!-- High-fidelity visual OBJs -->
    <mesh name="link_base_vis" file="xarm7/visual/link_base.obj"/>
    <mesh name="link1_white_vis" file="xarm7/visual/link1_white.obj"/>
    <mesh name="link1_silver_vis" file="xarm7/visual/link1_silver.obj"/>
    <mesh name="link2_white_vis" file="xarm7/visual/link2_white.obj"/>
    <mesh name="link2_silver_vis" file="xarm7/visual/link2_silver.obj"/>
    <mesh name="link3_white_vis" file="xarm7/visual/link3_white.obj"/>
    <mesh name="link3_silver_vis" file="xarm7/visual/link3_silver.obj"/>
    <mesh name="link4_white_vis" file="xarm7/visual/link4_white.obj"/>
    <mesh name="link4_silver_vis" file="xarm7/visual/link4_silver.obj"/>
    <mesh name="link5_white_vis" file="xarm7/visual/link5_white.obj"/>
    <mesh name="link5_silver_vis" file="xarm7/visual/link5_silver.obj"/>
    <mesh name="link6_white_vis" file="xarm7/visual/link6_white.obj"/>
    <mesh name="link6_silver_vis" file="xarm7/visual/link6_silver.obj"/>
    <mesh name="link7_white_vis" file="xarm7/visual/link7_white.obj"/>
    <mesh name="link7_silver_vis" file="xarm7/visual/link7_silver.obj"/>
    <mesh name="link7_black_vis" file="xarm7/visual/link7_black.obj"/>

    <!-- Collision STLs -->
    <mesh name="link_base_col" file="xarm7/collision/link_base.stl"/>
    <mesh name="link1_col" file="xarm7/collision/link1.stl"/>
    <mesh name="link2_col" file="xarm7/collision/link2.stl"/>
    <mesh name="link3_col" file="xarm7/collision/link3.stl"/>
    <mesh name="link4_col" file="xarm7/collision/link4.stl"/>
    <mesh name="link5_col" file="xarm7/collision/link5.stl"/>
    <mesh name="link6_col" file="xarm7/collision/link6.stl"/>
    <mesh name="link7_col" file="xarm7/collision/link7.stl"/>
  </asset>

  <default>
    <default class="xarm7">
      <joint axis="0 0 1" armature="0.1" range="-6.28319 6.28319" frictionloss="1"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator -->
      <default class="size1"><!-- shoulder joints -->
        <joint damping="10"/>
        <general gainprm="1500" biasprm="0 -1500 -150"/>
      </default>
      <default class="size2"><!-- elbow / arm joints -->
        <joint damping="5"/>
        <general gainprm="1000" biasprm="0 -1000 -100"/>
      </default>
      <default class="size3"><!-- wrist / flange joints -->
        <joint damping="2"/>
        <general gainprm="800" biasprm="0 -800 -80"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="xarm7">
      <geom class="visual" mesh="link_base_vis" material="silver"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.267">
        <inertial pos="-0.0002 0.02905 -0.01233" quat="0.978953 -0.202769 -0.00441617 -0.0227264" mass="2.382"
          diaginertia="0.00569127 0.00533384 0.00293865"/>
        <joint name="joint1" class="size1"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_silver_vis" material="silver"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="0.707105 -0.707108 0 0">
          <inertial pos="0.00022 -0.12856 0.01735" quat="0.50198 0.86483 -0.00778841 0.00483285" mass="1.869"
            diaginertia="0.00959898 0.00937717 0.00201315"/>
          <joint name="joint2" range="-2.059 2.0944" class="size1"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_silver_vis" material="silver"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0 -0.293 0" quat="0.707105 0.707108 0 0">
            <inertial pos="0.0466 -0.02463 -0.00768" quat="0.913819 0.289775 0.281481 -0.0416455" mass="1.6383"
              diaginertia="0.00351721 0.00294089 0.00195868"/>
            <joint name="joint3" class="size2"/>
            <geom class="visual" mesh="link3_white_vis" material="white"/>
            <geom class="visual" mesh="link3_silver_vis" material="silver"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.0525 0 0" quat="0.707105 0.707108 0 0">
              <inertial pos="0.07047 -0.11575 0.012" quat="0.422108 0.852026 -0.126025 0.282832" mass="1.7269"
                diaginertia="0.00657137 0.00647948 0.00186763"/>
              <joint name="joint4" range="-0.19198 3.927" class="size2"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_silver_vis" material="silver"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" pos="0.0775 -0.3425 0" quat="0.707105 0.707108 0 0">
                <inertial pos="-0.00032 0.01604 -0.026" quat="0.999311 -0.0304457 0.000577067 0.0212082" mass="1.3203"
                  diaginertia="0.00534729 0.00499076 0.0013489"/>
                <joint name="joint5" class="size2"/>
                <geom class="visual" mesh="link5_white_vis" material="white"/>
                <geom class="visual" mesh="link5_silver_vis" material="silver"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" quat="0.707105 0.707108 0 0">
                  <inertial pos="0.06469 0.03278 0.02141" quat="-0.217672 0.772419 0.16258 0.574069" mass="1.325"
                    diaginertia="0.00245421 0.00221646 0.00107273"/>
                  <joint name="joint6" range="-1.69297 3.14159" class="size3"/>
                  <geom class="visual" mesh="link6_white_vis" material="white"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="collision" mesh="link6_col"/>
                  <body name="link7" pos="0.076 0.097 0" quat="0.707105 -0.707108 0 0">
                    <inertial pos="0 -0.00677 -0.01098" quat="0.487612 0.512088 -0.512088 0.487612" mass="0.17"
                      diaginertia="0.000132176 9.3e-05 5.85236e-05"/>
                    <joint name="joint7" class="size3"/>
                    <geom class="visual" mesh="link7_white_vis" material="white"/>
                    <geom class="visual" mesh="link7_silver_vis" material="silver"/>
                    <geom class="visual" mesh="link7_black_vis" material="black"/>
                    <geom class="collision" mesh="link7_col"/>
                    <site name="attachment_site" pos="0 0 0"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link6" body2="link7"/>
  </contact>

  <actuator>
    <general name="act1" joint="joint1" class="size1" forcerange="-179.4445 179.4445"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.059 2.0944" forcerange="-179.4445 179.4445"/>
    <general name="act3" joint="joint3" class="size2" forcerange="-92 92"/>
    <general name="act4" joint="joint4" class="size2" ctrlrange="-0.19198 3.927" forcerange="-92 92"/>
    <general name="act5" joint="joint5" class="size2" forcerange="-81.558 81.558"/>
    <general name="act6" joint="joint6" class="size3" ctrlrange="-1.69297 3.14159" forcerange="-30.6 30.6"/>
    <general name="act7" joint="joint7" class="size3" forcerange="-30.6 30.6"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
    <key name="ready" qpos="0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "xarm7.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "xarm7", "xarm7.xml")


def build_xarm5() -> None:
    print("Building xArm 5...")
    target_dir = MODELS_DIR / "xarm5"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="xarm5">
  <!-- UFACTORY xArm 5 (1305 variant), high-fidelity multi-material visual OBJs
       with dedicated collision STLs. Actuator parameters follow Menagerie conventions. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="black" rgba="0.25 0.25 0.25 1" specular="0.2" shininess="0.1"/>

    <!-- High-fidelity visual OBJs -->
    <mesh name="link_base_vis" file="xarm5/visual/link_base.obj"/>
    <mesh name="link1_white_vis" file="xarm5/visual/link1_white.obj"/>
    <mesh name="link1_silver_vis" file="xarm5/visual/link1_silver.obj"/>
    <mesh name="link2_white_vis" file="xarm5/visual/link2_white.obj"/>
    <mesh name="link2_silver_vis" file="xarm5/visual/link2_silver.obj"/>
    <mesh name="link3_white_vis" file="xarm5/visual/link3_white.obj"/>
    <mesh name="link3_silver_vis" file="xarm5/visual/link3_silver.obj"/>
    <mesh name="link4_white_vis" file="xarm5/visual/link4_white.obj"/>
    <mesh name="link4_silver_vis" file="xarm5/visual/link4_silver.obj"/>
    <mesh name="link5_white_vis" file="xarm5/visual/link5_white.obj"/>
    <mesh name="link5_silver_vis" file="xarm5/visual/link5_silver.obj"/>
    <mesh name="link5_black_vis" file="xarm5/visual/link5_black.obj"/>

    <!-- Collision STLs -->
    <mesh name="link_base_col" file="xarm5/collision/link_base.stl"/>
    <mesh name="link1_col" file="xarm5/collision/link1.stl"/>
    <mesh name="link2_col" file="xarm5/collision/link2.stl"/>
    <mesh name="link3_col" file="xarm5/collision/link3.stl"/>
    <mesh name="link4_col" file="xarm5/collision/link4.stl"/>
    <mesh name="link5_col" file="xarm5/collision/link5.stl"/>
  </asset>

  <default>
    <default class="xarm5">
      <joint axis="0 0 1" armature="0.1" range="-6.28319 6.28319" frictionloss="1"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator -->
      <default class="size1"><!-- shoulder joints -->
        <joint damping="10"/>
        <general gainprm="1500" biasprm="0 -1500 -150"/>
      </default>
      <default class="size2"><!-- elbow joint -->
        <joint damping="5"/>
        <general gainprm="1000" biasprm="0 -1000 -100"/>
      </default>
      <default class="size3"><!-- wrist / flange joints -->
        <joint damping="2"/>
        <general gainprm="800" biasprm="0 -800 -80"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="xarm5">
      <geom class="visual" mesh="link_base_vis" material="silver"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.267">
        <inertial pos="0.00013 0.0294 -0.01239" quat="0.976897 -0.212169 0.00886257 0.0240422" mass="2.3849"
          diaginertia="0.00570618 0.00530019 0.00294197"/>
        <joint name="joint1" class="size1"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_silver_vis" material="silver"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="0.707105 -0.707108 0 0">
          <inertial pos="0.03788 -0.2254 0.03447" quat="0.632994 0.750527 0.187997 -0.0261719" mass="2.164"
            diaginertia="0.0289206 0.0277808 0.00354667"/>
          <joint name="joint2" range="-2.059 2.0944" class="size1"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_silver_vis" material="silver"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.0535 -0.2845 0">
            <inertial pos="0.06883 0.22985 0.00102" quat="0.75411 0.568954 -0.247614 0.215165" mass="2.121"
              diaginertia="0.0382682 0.0373731 0.00349086"/>
            <joint name="joint3" range="-3.927 0.19198" class="size2"/>
            <geom class="visual" mesh="link3_white_vis" material="white"/>
            <geom class="visual" mesh="link3_silver_vis" material="silver"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.0775 0.3425 0">
              <inertial pos="0.06489 0.03122 0.00319" quat="-0.224519 0.773365 0.156356 0.571884" mass="1.3169"
                diaginertia="0.00240435 0.00212066 0.00106127"/>
              <joint name="joint4" range="-1.69297 3.14159" class="size3"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_silver_vis" material="silver"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" pos="0.076 0.097 0" quat="0.707105 -0.707108 0 0">
                <inertial pos="0 -0.00677 -0.01098" quat="0.487612 0.512088 -0.512088 0.487612" mass="0.17"
                  diaginertia="0.000132176 9.3e-05 5.85236e-05"/>
                <joint name="joint5" class="size3"/>
                <geom class="visual" mesh="link5_white_vis" material="white"/>
                <geom class="visual" mesh="link5_silver_vis" material="silver"/>
                <geom class="visual" mesh="link5_black_vis" material="black"/>
                <geom class="collision" mesh="link5_col"/>
                <site name="attachment_site" pos="0 0 0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
  </contact>

  <actuator>
    <general name="act1" joint="joint1" class="size1" forcerange="-179.4445 179.4445"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.059 2.0944" forcerange="-179.4445 179.4445"/>
    <general name="act3" joint="joint3" class="size2" ctrlrange="-3.927 0.19198" forcerange="-92 92"/>
    <general name="act4" joint="joint4" class="size3" ctrlrange="-1.69297 3.14159" forcerange="-30.6 30.6"/>
    <general name="act5" joint="joint5" class="size3" forcerange="-30.6 30.6"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0" ctrl="0 0 0 0 0"/>
    <key name="ready" qpos="0 0 0 0 0" ctrl="0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "xarm5.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "xarm5", "xarm5.xml")


def build_uf850() -> None:
    print("Building UF 850...")
    target_dir = MODELS_DIR / "uf850"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="uf850">
  <!-- UFACTORY UF850 robot arm, high-fidelity multi-material visual OBJs
       with dedicated collision STLs. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>

    <!-- High-fidelity visual OBJs -->
    <mesh name="link_base_vis" file="uf850/visual/link_base.obj"/>
    <mesh name="link1_white_vis" file="uf850/visual/link1_white.obj"/>
    <mesh name="link1_silver_vis" file="uf850/visual/link1_silver.obj"/>
    <mesh name="link2_white_vis" file="uf850/visual/link2_white.obj"/>
    <mesh name="link2_silver_vis" file="uf850/visual/link2_silver.obj"/>
    <mesh name="link3_white_vis" file="uf850/visual/link3_white.obj"/>
    <mesh name="link3_silver_vis" file="uf850/visual/link3_silver.obj"/>
    <mesh name="link4_white_vis" file="uf850/visual/link4_white.obj"/>
    <mesh name="link4_silver_vis" file="uf850/visual/link4_silver.obj"/>
    <mesh name="link5_vis" file="uf850/visual/link5.obj"/>
    <mesh name="link6_vis" file="uf850/visual/link6.obj"/>

    <!-- Collision STLs -->
    <mesh name="link_base_col" file="uf850/collision/link_base.stl"/>
    <mesh name="link1_col" file="uf850/collision/link1.stl"/>
    <mesh name="link2_col" file="uf850/collision/link2.stl"/>
    <mesh name="link3_col" file="uf850/collision/link3.stl"/>
    <mesh name="link4_col" file="uf850/collision/link4.stl"/>
    <mesh name="link5_col" file="uf850/collision/link5.stl"/>
    <mesh name="link6_col" file="uf850/collision/link6.stl"/>
  </asset>

  <default>
    <default class="uf850">
      <joint axis="0 0 1" armature="0.1" range="-6.28319 6.28319" frictionloss="2"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator -->
      <default class="size1"><!-- base / shoulder -->
        <joint damping="15"/>
        <general gainprm="3000" biasprm="0 -3000 -300"/>
      </default>
      <default class="size2"><!-- elbow / forearm -->
        <joint damping="6"/>
        <general gainprm="1500" biasprm="0 -1500 -150"/>
      </default>
      <default class="size3"><!-- wrist / flange -->
        <joint damping="3"/>
        <general gainprm="800" biasprm="0 -800 -80"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="uf850">
      <geom class="visual" mesh="link_base_vis" material="silver"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.364">
        <inertial pos="5.7e-05 0.013842 -0.02055" quat="0.999624 -0.0273411 -0.000364635 -0.00185749" mass="4.807"
          diaginertia="0.0220018 0.0189206 0.0122503"/>
        <joint name="joint1" class="size1"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_silver_vis" material="silver"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="0.5 0.5 -0.5 0.5">
          <inertial pos="0.323425 9.5e-05 0.007586" quat="0.463878 0.532622 0.539126 0.458775" mass="3.28"
            diaginertia="0.0642088 0.0624144 0.00637301"/>
          <joint name="joint2" range="-2.30383 2.30383" class="size1"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_silver_vis" material="silver"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.39 0 0" quat="0 -0.707107 0.707107 0">
            <inertial pos="0.128834 0.11341 0.00978" quat="0.781477 0.318832 -0.496505 0.202789" mass="2.237"
              diaginertia="0.0155188 0.0144822 0.00312732"/>
            <joint name="joint3" range="-4.2237 0.061087" class="size2"/>
            <geom class="visual" mesh="link3_white_vis" material="white"/>
            <geom class="visual" mesh="link3_silver_vis" material="silver"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.15 0.426 0" quat="0.707105 -0.707108 0 0">
              <inertial pos="0.000266 -0.03833 -0.101915" quat="0.983546 0.180626 -0.00118669 -0.00323727" mass="2.041"
                diaginertia="0.0199188 0.0197497 0.00204912"/>
              <joint name="joint4" class="size2"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_silver_vis" material="silver"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 -0.707108 0 0">
                <inertial pos="5.7e-05 -0.015267 -0.001015" quat="0.709629 0.704516 0.00436679 -0.0080782" mass="1.385"
                  diaginertia="0.00207749 0.00203574 0.00110911"/>
                <joint name="joint5" range="-2.1642 2.1642" class="size3"/>
                <geom class="visual" mesh="link5_vis" material="white"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0 -0.09 0" quat="0.707105 0.707108 0 0">
                  <inertial pos="0.000484 -0.000334 -0.014217" quat="0.63763 0.308332 -0.636717 0.30488" mass="0.225"
                    diaginertia="0.000232503 0.000133671 0.000124886"/>
                  <joint name="joint6" class="size3"/>
                  <geom class="visual" mesh="link6_vis" material="silver"/>
                  <geom class="collision" mesh="link6_col"/>
                  <site name="attachment_site" pos="0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
  </contact>

  <actuator>
    <general name="act1" joint="joint1" class="size1" forcerange="-382.5 382.5"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.30383 2.30383" forcerange="-382.5 382.5"/>
    <general name="act3" joint="joint3" class="size2" ctrlrange="-4.2237 0.061087" forcerange="-243.525 243.525"/>
    <general name="act4" joint="joint4" class="size2" forcerange="-117.3 117.3"/>
    <general name="act5" joint="joint5" class="size3" ctrlrange="-2.1642 2.1642" forcerange="-38.25 38.25"/>
    <general name="act6" joint="joint6" class="size3" forcerange="-38.25 38.25"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
    <key name="ready" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "uf850.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "uf850", "uf850.xml", center="0.4 0 0.4", extent="1.0")


def build_lite6() -> None:
    print("Building Lite 6...")
    target_dir = MODELS_DIR / "lite6"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="lite6">
  <!-- UFACTORY Lite 6 desktop robot arm, high-fidelity multi-material visual OBJs
       with dedicated collision STLs. Actuator parameters follow Menagerie conventions. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="gray" rgba="0.78 0.78 0.78 1" specular="0.3" shininess="0.2"/>
    <material name="seal" rgba="0.8 0.82 0.88 1" specular="0.3" shininess="0.2"/>

    <!-- High-fidelity visual OBJs -->
    <mesh name="link_base_white_vis" file="lite6/visual/link_base_white.obj"/>
    <mesh name="link_base_gray_vis" file="lite6/visual/link_base_gray.obj"/>
    <mesh name="link1_white_vis" file="lite6/visual/link1_white.obj"/>
    <mesh name="link1_gray_vis" file="lite6/visual/link1_gray.obj"/>
    <mesh name="link2_white_vis" file="lite6/visual/link2_white.obj"/>
    <mesh name="link2_gray_vis" file="lite6/visual/link2_gray.obj"/>
    <mesh name="link3_vis" file="lite6/visual/link3.obj"/>
    <mesh name="link4_white_vis" file="lite6/visual/link4_white.obj"/>
    <mesh name="link4_gray_vis" file="lite6/visual/link4_gray.obj"/>
    <mesh name="link5_vis" file="lite6/visual/link5.obj"/>
    <mesh name="link6_silver_vis" file="lite6/visual/link6_silver.obj"/>
    <mesh name="link6_seal_vis" file="lite6/visual/link6_seal.obj"/>

    <!-- Collision STLs -->
    <mesh name="link_base_col" file="lite6/collision/link_base.stl"/>
    <mesh name="link1_col" file="lite6/collision/link1.stl"/>
    <mesh name="link2_col" file="lite6/collision/link2.stl"/>
    <mesh name="link3_col" file="lite6/collision/link3.stl"/>
    <mesh name="link4_col" file="lite6/collision/link4.stl"/>
    <mesh name="link5_col" file="lite6/collision/link5.stl"/>
    <mesh name="link6_col" file="lite6/collision/link6.stl"/>
  </asset>

  <default>
    <default class="lite6">
      <joint axis="0 0 1" armature="0.05" damping="1"/>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator;
           direct-drive wrist J4-J6 uses the retuned kp/kv and damping from the same spec -->
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" mass="0" density="0"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="lite6">
      <inertial pos="-0.00829545 3.26357e-05 0.0631195" mass="1.65394" diaginertia="0.003 0.003 0.002"/>
      <geom class="visual" mesh="link_base_white_vis" material="white"/>
      <geom class="visual" mesh="link_base_gray_vis" material="gray"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.2435">
        <inertial pos="-0.00036 0.04195 -0.0025" quat="0.608059 0.792349 -0.0438707 0.0228718" mass="1.411"
          diaginertia="0.00145276 0.00135275 0.000853355"/>
        <joint name="joint1" range="-6.28319 6.28319"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_gray_vis" material="gray"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="-0.5 0.5 0.5 0.5">
          <inertial pos="0.179 0 0.0584" quat="0.417561 0.571619 0.569585 0.417693" mass="1.34"
            diaginertia="0.00560971 0.0052152 0.00122018"/>
          <joint name="joint2" range="-2.61799 2.61799"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_gray_vis" material="gray"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.2002 0 0" quat="0 -0.707105 -0.707108 0">
            <inertial pos="0.072 -0.0357 -0.001" quat="0.128259 0.662963 -0.167256 0.71837" mass="0.953"
              diaginertia="0.0018521 0.00175546 0.000703807"/>
            <joint name="joint3" range="-0.061087 5.23599"/>
            <geom class="visual" mesh="link3_vis" material="white"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.087 -0.22761 0" quat="0.707105 0.707108 0 0">
              <inertial pos="-0.002 -0.0285 -0.0813" quat="0.975248 0.22109 0.00203498 -0.00262178" mass="1.284"
                diaginertia="0.00370503 0.00349091 0.00109586"/>
              <joint name="joint4" range="-6.28319 6.28319" damping="0"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_gray_vis" material="gray"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 0.707108 0 0">
                <inertial pos="0 0.01 0.0019" quat="0.71423 0.696388 -0.0531933 0.0456997" mass="0.804"
                  diaginertia="0.000567553 0.000529266 0.000507681"/>
                <joint name="joint5" range="-2.1642 2.1642" damping="0.1"/>
                <geom class="visual" mesh="link5_vis" material="white"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0 0.0625 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="0 -0.00194 -0.0102" quat="-0.0376023 0.704057 0.0446838 0.707738" mass="0.13"
                    diaginertia="0.000148148 8.57757e-05 7.71412e-05"/>
                  <joint name="joint6" range="-6.28319 6.28319" damping="0"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="visual" mesh="link6_seal_vis" material="seal"/>
                  <geom class="collision" mesh="link6_col"/>
                  <site name="attachment_site" pos="0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
  </contact>

  <actuator>
    <position joint="joint1" ctrlrange="-6.28319 6.28319" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint2" ctrlrange="-2.61799 2.61799" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint3" ctrlrange="-0.061087 5.23599" kp="2000" kv="200" forcerange="-30.6 30.6"/>
    <position joint="joint4" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
    <position joint="joint5" ctrlrange="-2.1642 2.1642" kp="80" kv="16" forcerange="-3.274812 3.274812"/>
    <position joint="joint6" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
    <!-- Studio factory ready pose (deg): 0 9.9 31.8 0 21.9 0 -->
    <key name="ready" qpos="0 0.172787595947 0.555014702134 0 0.382227106187 0" ctrl="0 0.172787595947 0.555014702134 0 0.382227106187 0"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "lite6.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "lite6", "lite6.xml", center="0.25 0 0.25", extent="0.6")


def build_gripper_g2() -> None:
    print("Building Gripper G2...")
    target_dir = MODELS_DIR / "gripper_g2"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="gripper_g2">
  <!-- UFACTORY Gripper G2, high-fidelity multi-material visual OBJs
       with dedicated collision STLs and 4-bar linkage mechanism. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="black" rgba="0.15 0.15 0.15 1" specular="0.2" shininess="0.1"/>

    <!-- Visual OBJs -->
    <mesh name="base_vis" file="gripper_g2/visual/base.obj"/>
    <mesh name="left_outer_knuckle_vis" file="gripper_g2/visual/left_outer_knuckle.obj"/>
    <mesh name="left_finger_vis" file="gripper_g2/visual/left_finger.obj"/>
    <mesh name="left_inner_knuckle_vis" file="gripper_g2/visual/left_inner_knuckle.obj"/>
    <mesh name="right_outer_knuckle_vis" file="gripper_g2/visual/right_outer_knuckle.obj"/>
    <mesh name="right_finger_vis" file="gripper_g2/visual/right_finger.obj"/>
    <mesh name="right_inner_knuckle_vis" file="gripper_g2/visual/right_inner_knuckle.obj"/>

    <!-- Collision STLs -->
    <mesh name="base_link_col" file="gripper_g2/collision/base_link.stl"/>
    <mesh name="left_outer_knuckle_col" file="gripper_g2/collision/left_outer_knuckle.stl"/>
    <mesh name="left_finger_col" file="gripper_g2/collision/left_finger.stl"/>
    <mesh name="left_inner_knuckle_col" file="gripper_g2/collision/left_inner_knuckle.stl"/>
    <mesh name="right_outer_knuckle_col" file="gripper_g2/collision/right_outer_knuckle.stl"/>
    <mesh name="right_finger_col" file="gripper_g2/collision/right_finger.stl"/>
    <mesh name="right_inner_knuckle_col" file="gripper_g2/collision/right_inner_knuckle.stl"/>
  </asset>

  <default>
    <default class="g2">
      <default class="g2_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="g2_collision">
        <geom type="mesh" group="3"/>
      </default>
      <default class="driver">
        <joint range="0 0.85" armature="0.005" damping="0.1" solreflimit="0.005 1"/>
      </default>
      <default class="follower">
        <joint range="0 0.85" solreflimit="0.005 1"/>
      </default>
      <default class="spring_link">
        <joint range="0 0.85" stiffness="0.05" springref="2.62" damping="0.00125"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="xarm_gripper_base_link" childclass="g2">
      <inertial pos="-0.000655 -0.00185 0.04803" quat="0.9974 -0.0718 -0.0062 0.0005"
        mass="0.54156" diaginertia="0.000471 0.000332 0.000255"/>
      <geom class="g2_visual" mesh="base_vis" material="white"/>
      <geom class="g2_collision" mesh="base_link_col"/>

      <!-- Left finger chain -->
      <body name="left_outer_knuckle" pos="0 0.035 0.059098">
        <inertial pos="0 0.02156 0.01518" quat="0.4779 0.8784 0 0" mass="0.03362"
          diaginertia="1.91e-05 1.79e-05 1.90e-06"/>
        <joint name="left_driver_joint" axis="1 0 0" class="driver"/>
        <geom class="g2_visual" mesh="left_outer_knuckle_vis" material="black"/>
        <geom class="g2_collision" mesh="left_outer_knuckle_col"/>
        <body name="left_finger" pos="0 0.035465 0.042039">
          <inertial pos="0 -0.0164 0.02926" quat="0.6976 0.1154 -0.1154 0.6976" mass="0.0483"
            diaginertia="1.88e-05 1.75e-05 3.57e-06"/>
          <joint name="left_finger_joint" axis="-1 0 0" class="follower"/>
          <geom class="g2_visual" mesh="left_finger_vis" material="black"/>
          <geom class="g2_collision" mesh="left_finger_col"/>
        </body>
      </body>
      <body name="left_inner_knuckle" pos="0 0.02 0.074098">
        <inertial pos="0 0.02205 0.02613" quat="0.6641 -0.2427 0.2427 0.6641" mass="0.02301"
          diaginertia="8.34e-06 6.09e-06 2.76e-06"/>
        <joint name="left_inner_knuckle_joint" axis="1 0 0" class="spring_link"/>
        <geom class="g2_visual" mesh="left_inner_knuckle_vis" material="black"/>
        <geom class="g2_collision" mesh="left_inner_knuckle_col"/>
      </body>

      <!-- Right finger chain -->
      <body name="right_outer_knuckle" pos="0 -0.035 0.059098">
        <inertial pos="0 -0.02156 0.01518" quat="0.8784 0.4779 0 0" mass="0.03362"
          diaginertia="1.91e-05 1.79e-05 1.90e-06"/>
        <joint name="right_driver_joint" axis="-1 0 0" class="driver"/>
        <geom class="g2_visual" mesh="right_outer_knuckle_vis" material="black"/>
        <geom class="g2_collision" mesh="right_outer_knuckle_col"/>
        <body name="right_finger" pos="0 -0.035465 0.042039">
          <inertial pos="0 0.0164 0.02926" quat="0.6976 -0.1154 0.1154 0.6976" mass="0.0483"
            diaginertia="1.88e-05 1.75e-05 3.57e-06"/>
          <joint name="right_finger_joint" axis="1 0 0" class="follower"/>
          <geom class="g2_visual" mesh="right_finger_vis" material="black"/>
          <geom class="g2_collision" mesh="right_finger_col"/>
        </body>
      </body>
      <body name="right_inner_knuckle" pos="0 -0.02 0.074098">
        <inertial pos="0 -0.02205 0.02613" quat="0.6641 0.2427 -0.2427 0.6641" mass="0.02301"
          diaginertia="8.34e-06 6.09e-06 2.76e-06"/>
        <joint name="right_inner_knuckle_joint" axis="-1 0 0" class="spring_link"/>
        <geom class="g2_visual" mesh="right_inner_knuckle_vis" material="black"/>
        <geom class="g2_collision" mesh="right_inner_knuckle_col"/>
      </body>

      <site name="link_tcp" pos="0 0 0.172"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="right_inner_knuckle" body2="right_outer_knuckle"/>
    <exclude body1="right_inner_knuckle" body2="right_finger"/>
    <exclude body1="left_inner_knuckle" body2="left_outer_knuckle"/>
    <exclude body1="left_inner_knuckle" body2="left_finger"/>
  </contact>

  <equality>
    <joint joint1="left_driver_joint" joint2="left_finger_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="left_inner_knuckle_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_driver_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_finger_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_inner_knuckle_joint" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <position name="gripper_motor" joint="left_driver_joint" kp="100" kv="10" forcerange="-50 50" ctrlrange="0 0.85"/>
  </actuator>

  <keyframe>
    <key name="open" qpos="0 0 0 0 0 0" ctrl="0"/>
    <key name="close" qpos="0.8 0.8 0.8 0.8 0.8 0.8" ctrl="0.8"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "gripper_g2.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "gripper_g2", "gripper_g2.xml", center="0 0 0.1", extent="0.3")


def build_bio_gripper_g2() -> None:
    print("Building Bio Gripper G2...")
    target_dir = MODELS_DIR / "bio_gripper_g2"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="bio_gripper_g2">
  <!-- UFACTORY Bio Gripper G2, high-fidelity visual OBJs and collision STLs. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>

    <mesh name="base_white_vis" file="bio_gripper_g2/visual/bio_gripper_g2_base_white.obj"/>
    <mesh name="base_silver_vis" file="bio_gripper_g2/visual/bio_gripper_g2_base_silver.obj"/>
    <mesh name="left_finger_vis" file="bio_gripper_g2/visual/bio_gripper_g2_left_finger.obj"/>
    <mesh name="right_finger_vis" file="bio_gripper_g2/visual/bio_gripper_g2_right_finger.obj"/>

    <mesh name="base_col" file="bio_gripper_g2/collision/bio_gripper_g2_link_base.stl"/>
    <mesh name="left_finger_col" file="bio_gripper_g2/collision/bio_gripper_g2_left_finger.stl"/>
    <mesh name="right_finger_col" file="bio_gripper_g2/collision/bio_gripper_g2_right_finger.stl"/>
  </asset>

  <default>
    <default class="bio_g2">
      <default class="bio_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="bio_collision">
        <geom type="mesh" group="3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="bio_gripper_g2_base_link" childclass="bio_g2">
      <inertial pos="0.016179 -2.3911E-05 0.023897" mass="0.38837" fullinertia="0.00088398 0.00018256 0.00094673 4.8665E-08 5.6276E-06 4.0828E-08"/>
      <geom class="bio_visual" mesh="base_white_vis" material="white"/>
      <geom class="bio_visual" mesh="base_silver_vis" material="silver"/>
      <geom class="bio_collision" mesh="base_col" pos="-0.022475 -0.000830 -0.004696"/>

      <body name="bio_gripper_g2_left_finger" pos="0.059 0 0.027">
        <inertial pos="0.021846 -0.0276 0.011911" mass="0.012105" fullinertia="3.7322E-06 1.744E-05 1.7775E-05 4.1705E-06 -4.0142E-06 1.2237E-06"/>
        <joint name="left_finger_joint" type="slide" axis="0 1 0" range="-0.0395 0" damping="0.5"/>
        <geom class="bio_visual" mesh="left_finger_vis" material="silver"/>
        <geom class="bio_collision" mesh="left_finger_col" pos="-0.012870 -0.011316 -0.002219"/>
      </body>

      <body name="bio_gripper_g2_right_finger" pos="0.059 0 0.027">
        <inertial pos="0.02037 0.027565 0.010274" mass="0.012068" fullinertia="4.1413E-06 1.6115E-05 1.6032E-05 -3.9682E-06 -4.5734E-06 -1.6711E-06"/>
        <joint name="right_finger_joint" type="slide" axis="0 1 0" range="0 0.0395" damping="0.5"/>
        <geom class="bio_visual" mesh="right_finger_vis" material="silver"/>
        <geom class="bio_collision" mesh="right_finger_col" pos="-0.013790 0.011745 -0.001039"/>
      </body>

      <site name="link_tcp" pos="0.135 0 0.055"/>
    </body>
  </worldbody>

  <contact>
    <!-- Adjacent finger/housing meshes overlap at the slide bearings. -->
    <exclude body1="bio_gripper_g2_base_link" body2="bio_gripper_g2_left_finger"/>
    <exclude body1="bio_gripper_g2_base_link" body2="bio_gripper_g2_right_finger"/>
  </contact>

  <equality>
    <joint joint1="right_finger_joint" joint2="left_finger_joint" polycoef="0 -1 0 0 0"/>
  </equality>

  <actuator>
    <position name="finger_actuator" joint="left_finger_joint" kp="500" kv="5" ctrlrange="-0.0395 0"/>
  </actuator>

  <keyframe>
    <key name="open" qpos="0 0" ctrl="0"/>
    <key name="close" qpos="-0.0395 0.0395" ctrl="-0.0395"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "bio_gripper_g2.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "bio_gripper_g2", "bio_gripper_g2.xml", center="0.05 0 0.05", extent="0.25")


def build_lite6_gripper() -> None:
    print("Building Lite6 Gripper...")
    target_dir = MODELS_DIR / "lite6_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="lite6_gripper">
  <!-- UFACTORY Lite6 Parallel Gripper, high-fidelity visual OBJs and collision STLs. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="black" rgba="0.15 0.15 0.15 1" specular="0.2" shininess="0.1"/>

    <mesh name="shell_vis" file="lite6_gripper/visual/shell.obj"/>
    <mesh name="finger1_white_vis" file="lite6_gripper/visual/finger1_white.obj"/>
    <mesh name="finger1_silver_vis" file="lite6_gripper/visual/finger1_silver.obj"/>
    <mesh name="finger1_black_vis" file="lite6_gripper/visual/finger1_black.obj"/>
    <mesh name="finger2_white_vis" file="lite6_gripper/visual/finger2_white.obj"/>
    <mesh name="finger2_silver_vis" file="lite6_gripper/visual/finger2_silver.obj"/>
    <mesh name="finger2_black_vis" file="lite6_gripper/visual/finger2_black.obj"/>

    <mesh name="shell_col" file="lite6_gripper/collision/shell.stl"/>
    <mesh name="finger1_col" file="lite6_gripper/collision/finger1.stl"/>
    <mesh name="finger2_col" file="lite6_gripper/collision/finger2.stl"/>
  </asset>

  <default>
    <default class="lg">
      <default class="lg_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="lg_collision">
        <geom type="mesh" group="3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="uflite_gripper_link" childclass="lg">
      <inertial pos="0 0 0.03" mass="0.25" fullinertia="0.00047106 0.00033072 0.00025642 0.00000039292 0.0000026537 -0.000010975"/>
      <geom class="lg_visual" mesh="shell_vis" material="white"/>
      <geom class="lg_collision" mesh="shell_col"/>

      <body name="uflite_finger1" pos="0 0 0.0543">
        <inertial pos="0 0.01 0.0086" mass="0.0163" fullinertia="1.425e-06 1.63e-06 8.63e-07 0 0 -3.1e-07"/>
        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.0089" damping="1"/>
        <geom class="lg_visual" mesh="finger1_white_vis" material="white"/>
        <geom class="lg_visual" mesh="finger1_silver_vis" material="silver"/>
        <geom class="lg_visual" mesh="finger1_black_vis" material="black"/>
        <geom class="lg_collision" mesh="finger1_col"/>
      </body>

      <body name="uflite_finger2" pos="0 0 0.0543">
        <inertial pos="0 -0.01 0.0086" mass="0.0163" fullinertia="1.425e-06 1.63e-06 8.63e-07 0 0 3.1e-07"/>
        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.0089" damping="1"/>
        <geom class="lg_visual" mesh="finger2_white_vis" material="white"/>
        <geom class="lg_visual" mesh="finger2_silver_vis" material="silver"/>
        <geom class="lg_visual" mesh="finger2_black_vis" material="black"/>
        <geom class="lg_collision" mesh="finger2_col"/>
      </body>

      <site name="link_tcp" pos="0 0 0.0836"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="finger_joint2" joint2="finger_joint1" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <position name="finger_actuator" joint="finger_joint1" kp="200" kv="10" ctrlrange="0 0.0089"/>
  </actuator>

  <keyframe>
    <key name="open" qpos="0 0" ctrl="0"/>
    <key name="close" qpos="0.0089 0.0089" ctrl="0.0089"/>
  </keyframe>
</mujoco>
"""
    (target_dir / "lite6_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "lite6_gripper", "lite6_gripper.xml", center="0 0 0.05", extent="0.2")


def build_lite6_vacuum_gripper() -> None:
    print("Building Lite6 Vacuum Gripper...")
    target_dir = MODELS_DIR / "lite6_vacuum_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="lite6_vacuum_gripper">
  <!-- UFACTORY Lite6 Vacuum Suction Gripper, visual OBJ and collision STL. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>

    <mesh name="vacuum_vis" file="lite6_vacuum_gripper/visual/vacuum_gripper.obj"/>
    <mesh name="vacuum_col" file="lite6_vacuum_gripper/collision/vacuum_gripper_lite.stl"/>
  </asset>

  <default>
    <default class="vg">
      <default class="vg_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="vg_collision">
        <geom type="mesh" group="3"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body name="uflite_vacuum_gripper_link" childclass="vg">
      <inertial pos="0 0 0.03" mass="0.14" fullinertia="0.00047106 0.00033072 0.00025642 0.00000039292 0.0000026537 -0.000010975"/>
      <geom class="vg_visual" mesh="vacuum_vis" material="white"/>
      <geom class="vg_collision" mesh="vacuum_col"/>
      <site name="suction_site" pos="0 0 0.061" size="0.01" rgba="0 1 0 0.5"/>
    </body>
  </worldbody>
</mujoco>
"""
    (target_dir / "lite6_vacuum_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "lite6_vacuum_gripper", "lite6_vacuum_gripper.xml", center="0 0 0.03", extent="0.15")


def build_xarm6_with_gripper() -> None:
    print("Building xArm 6 + Gripper G2 combo...")
    target_dir = MODELS_DIR / "xarm6_with_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="xarm6_with_gripper">
  <!-- UFACTORY xArm 6 equipped with Gripper G2 (Roadmap 0.0.1). -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="black" rgba="0.25 0.25 0.25 1" specular="0.2" shininess="0.1"/>
    <material name="seal" rgba="0.8 0.82 0.85 1" specular="0.3" shininess="0.2"/>

    <!-- xArm 6 Visual OBJs -->
    <mesh name="link_base_vis" file="xarm6/visual/link_base.obj"/>
    <mesh name="link1_seal_vis" file="xarm6/visual/link1_seal.obj"/>
    <mesh name="link1_white_vis" file="xarm6/visual/link1_white.obj"/>
    <mesh name="link1_silver_vis" file="xarm6/visual/link1_silver.obj"/>
    <mesh name="link2_white_vis" file="xarm6/visual/link2_white.obj"/>
    <mesh name="link2_silver_vis" file="xarm6/visual/link2_silver.obj"/>
    <mesh name="link3_white_vis" file="xarm6/visual/link3_white.obj"/>
    <mesh name="link3_silver_vis" file="xarm6/visual/link3_silver.obj"/>
    <mesh name="link4_white_vis" file="xarm6/visual/link4_white.obj"/>
    <mesh name="link4_silver_vis" file="xarm6/visual/link4_silver.obj"/>
    <mesh name="link5_white_vis" file="xarm6/visual/link5_white.obj"/>
    <mesh name="link5_silver_vis" file="xarm6/visual/link5_silver.obj"/>
    <mesh name="link6_white_vis" file="xarm6/visual/link6_white.obj"/>
    <mesh name="link6_black_vis" file="xarm6/visual/link6_black.obj"/>
    <mesh name="link6_silver_vis" file="xarm6/visual/link6_silver.obj"/>

    <!-- xArm 6 Collision STLs -->
    <mesh name="link_base_col" file="xarm6/collision/link_base.stl"/>
    <mesh name="link1_col" file="xarm6/collision/link1.stl"/>
    <mesh name="link2_col" file="xarm6/collision/link2.stl"/>
    <mesh name="link3_col" file="xarm6/collision/link3.stl"/>
    <mesh name="link4_col" file="xarm6/collision/link4.stl"/>
    <mesh name="link5_col" file="xarm6/collision/link5.stl"/>
    <mesh name="link6_col" file="xarm6/collision/link6.stl"/>

    <!-- Gripper G2 Visual OBJs -->
    <mesh name="g2_base_vis" file="gripper_g2/visual/base.obj"/>
    <mesh name="g2_left_outer_knuckle_vis" file="gripper_g2/visual/left_outer_knuckle.obj"/>
    <mesh name="g2_left_finger_vis" file="gripper_g2/visual/left_finger.obj"/>
    <mesh name="g2_left_inner_knuckle_vis" file="gripper_g2/visual/left_inner_knuckle.obj"/>
    <mesh name="g2_right_outer_knuckle_vis" file="gripper_g2/visual/right_outer_knuckle.obj"/>
    <mesh name="g2_right_finger_vis" file="gripper_g2/visual/right_finger.obj"/>
    <mesh name="g2_right_inner_knuckle_vis" file="gripper_g2/visual/right_inner_knuckle.obj"/>

    <!-- Gripper G2 Collision STLs -->
    <mesh name="g2_base_link_col" file="gripper_g2/collision/base_link.stl"/>
    <mesh name="g2_left_outer_knuckle_col" file="gripper_g2/collision/left_outer_knuckle.stl"/>
    <mesh name="g2_left_finger_col" file="gripper_g2/collision/left_finger.stl"/>
    <mesh name="g2_left_inner_knuckle_col" file="gripper_g2/collision/left_inner_knuckle.stl"/>
    <mesh name="g2_right_outer_knuckle_col" file="gripper_g2/collision/right_outer_knuckle.stl"/>
    <mesh name="g2_right_finger_col" file="gripper_g2/collision/right_finger.stl"/>
    <mesh name="g2_right_inner_knuckle_col" file="gripper_g2/collision/right_inner_knuckle.stl"/>
  </asset>

  <default>
    <default class="xarm6">
      <joint axis="0 0 1" armature="0.1" range="-6.28319 6.28319" frictionloss="1"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator -->
      <default class="size1">
        <joint damping="10"/>
        <general gainprm="1500" biasprm="0 -1500 -150"/>
      </default>
      <default class="size2">
        <joint damping="5"/>
        <general gainprm="1000" biasprm="0 -1000 -100"/>
      </default>
      <default class="size3">
        <joint damping="2"/>
        <general gainprm="800" biasprm="0 -800 -80"/>
      </default>
      <default class="driver">
        <joint range="0 0.85" armature="0.005" damping="0.1" solreflimit="0.005 1"/>
      </default>
      <default class="follower">
        <joint range="0 0.85" solreflimit="0.005 1"/>
      </default>
      <default class="spring_link">
        <joint range="0 0.85" stiffness="0.05" springref="2.62" damping="0.00125"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="xarm6">
      <geom class="visual" mesh="link_base_vis" material="silver"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.267">
        <inertial pos="0.00022 0.02951 -0.0124" quat="0.976139 -0.216079 -0.00387305 -0.0211673" mass="2.3814"
          diaginertia="0.00585697 0.00544434 0.00312315"/>
        <joint name="joint1" class="size1"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_silver_vis" material="silver"/>
        <geom class="visual" mesh="link1_seal_vis" material="seal"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="0.707105 -0.707108 0 0">
          <inertial pos="0.03881 -0.22783 0.03496" quat="0.635305 0.753023 0.170976 -0.010531" mass="2.2675"
            diaginertia="0.0294477 0.0281783 0.00383449"/>
          <joint name="joint2" range="-2.059 2.0944" class="size1"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_silver_vis" material="silver"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.0535 -0.2845 0">
            <inertial pos="0.07041 0.11631 0.0107" quat="0.845838 0.425539 -0.293179 0.132369" mass="1.875"
              diaginertia="0.00714132 0.00699471 0.00196753"/>
            <joint name="joint3" range="-3.927 0.19198" class="size2"/>
            <geom class="visual" mesh="link3_white_vis" material="white"/>
            <geom class="visual" mesh="link3_silver_vis" material="silver"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.0775 0.3425 0" quat="0.707105 -0.707108 0 0">
              <inertial pos="-0.00018 0.01798 -0.02291" quat="0.99929 -0.0369702 0.00179429 -0.00701173" mass="1.3192"
                diaginertia="0.00489614 0.00447403 0.00132609"/>
              <joint name="joint4" class="size2"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_silver_vis" material="silver"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 0.707108 0 0">
                <inertial pos="0.0651 0.03096 0.00315" quat="-0.223008 0.771309 0.16027 0.574163" mass="1.33854"
                  diaginertia="0.00241824 0.00213367 0.00108731"/>
                <joint name="joint5" range="-1.69297 3.14159" class="size2"/>
                <geom class="visual" mesh="link5_white_vis" material="white"/>
                <geom class="visual" mesh="link5_silver_vis" material="silver"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0.076 0.097 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="0 -0.00677 -0.01098" quat="0.487612 0.512088 -0.512088 0.487612" mass="0.17"
                    diaginertia="0.000132176 9.3e-05 5.85236e-05"/>
                  <joint name="joint6" class="size3"/>
                  <geom class="visual" mesh="link6_white_vis" material="white"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="visual" mesh="link6_black_vis" material="black"/>
                  <geom class="collision" mesh="link6_col"/>

                  <!-- Mounted Gripper G2 -->
                  <body name="xarm_gripper_base_link">
                    <inertial pos="-0.000655 -0.00185 0.04803" quat="0.9974 -0.0718 -0.0062 0.0005"
                      mass="0.54156" diaginertia="0.000471 0.000332 0.000255"/>
                    <geom class="visual" mesh="g2_base_vis" material="white"/>
                    <geom class="collision" mesh="g2_base_link_col"/>

                    <body name="left_outer_knuckle" pos="0 0.035 0.059098">
                      <inertial pos="0 0.02156 0.01518" quat="0.4779 0.8784 0 0" mass="0.03362"
                        diaginertia="1.91e-05 1.79e-05 1.90e-06"/>
                      <joint name="left_driver_joint" axis="1 0 0" class="driver"/>
                      <geom class="visual" mesh="g2_left_outer_knuckle_vis" material="black"/>
                      <geom class="collision" mesh="g2_left_outer_knuckle_col"/>
                      <body name="left_finger" pos="0 0.035465 0.042039">
                        <inertial pos="0 -0.0164 0.02926" quat="0.6976 0.1154 -0.1154 0.6976" mass="0.0483"
                          diaginertia="1.88e-05 1.75e-05 3.57e-06"/>
                        <joint name="left_finger_joint" axis="-1 0 0" class="follower"/>
                        <geom class="visual" mesh="g2_left_finger_vis" material="black"/>
                        <geom class="collision" mesh="g2_left_finger_col"/>
                      </body>
                    </body>
                    <body name="left_inner_knuckle" pos="0 0.02 0.074098">
                      <inertial pos="0 0.02205 0.02613" quat="0.6641 -0.2427 0.2427 0.6641" mass="0.02301"
                        diaginertia="8.34e-06 6.09e-06 2.76e-06"/>
                      <joint name="left_inner_knuckle_joint" axis="1 0 0" class="spring_link"/>
                      <geom class="visual" mesh="g2_left_inner_knuckle_vis" material="black"/>
                      <geom class="collision" mesh="g2_left_inner_knuckle_col"/>
                    </body>

                    <body name="right_outer_knuckle" pos="0 -0.035 0.059098">
                      <inertial pos="0 -0.02156 0.01518" quat="0.8784 0.4779 0 0" mass="0.03362"
                        diaginertia="1.91e-05 1.79e-05 1.90e-06"/>
                      <joint name="right_driver_joint" axis="-1 0 0" class="driver"/>
                      <geom class="visual" mesh="g2_right_outer_knuckle_vis" material="black"/>
                      <geom class="collision" mesh="g2_right_outer_knuckle_col"/>
                      <body name="right_finger" pos="0 -0.035465 0.042039">
                        <inertial pos="0 0.0164 0.02926" quat="0.6976 -0.1154 0.1154 0.6976" mass="0.0483"
                          diaginertia="1.88e-05 1.75e-05 3.57e-06"/>
                        <joint name="right_finger_joint" axis="1 0 0" class="follower"/>
                        <geom class="visual" mesh="g2_right_finger_vis" material="black"/>
                        <geom class="collision" mesh="g2_right_finger_col"/>
                      </body>
                    </body>
                    <body name="right_inner_knuckle" pos="0 -0.02 0.074098">
                      <inertial pos="0 -0.02205 0.02613" quat="0.6641 0.2427 -0.2427 0.6641" mass="0.02301"
                        diaginertia="8.34e-06 6.09e-06 2.76e-06"/>
                      <joint name="right_inner_knuckle_joint" axis="-1 0 0" class="spring_link"/>
                      <geom class="visual" mesh="g2_right_inner_knuckle_vis" material="black"/>
                      <geom class="collision" mesh="g2_right_inner_knuckle_col"/>
                    </body>

                    <site name="attachment_site" pos="0 0 0.172"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link6" body2="xarm_gripper_base_link"/>
    <exclude body1="right_inner_knuckle" body2="right_outer_knuckle"/>
    <exclude body1="right_inner_knuckle" body2="right_finger"/>
    <exclude body1="left_inner_knuckle" body2="left_outer_knuckle"/>
    <exclude body1="left_inner_knuckle" body2="left_finger"/>
  </contact>

  <equality>
    <joint joint1="left_driver_joint" joint2="left_finger_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="left_inner_knuckle_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_driver_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_finger_joint" polycoef="0 1 0 0 0"/>
    <joint joint1="left_driver_joint" joint2="right_inner_knuckle_joint" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <general name="act1" joint="joint1" class="size1" forcerange="-179.4445 179.4445"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.059 2.0944" forcerange="-179.4445 179.4445"/>
    <general name="act3" joint="joint3" class="size2" ctrlrange="-3.927 0.19198" forcerange="-92 92"/>
    <general name="act4" joint="joint4" class="size2" forcerange="-81.558 81.558"/>
    <general name="act5" joint="joint5" class="size2" ctrlrange="-1.69297 3.14159" forcerange="-30.6 30.6"/>
    <general name="act6" joint="joint6" class="size3" forcerange="-30.6 30.6"/>
    <position name="gripper" joint="left_driver_joint" kp="100" kv="10" forcerange="-50 50" ctrlrange="0 0.85"/>
  </actuator>

  <keyframe>
    <!-- J2=-0.3 lifts the mounted gripper clear of the floor (all-zero home leaves the tool floor-grazing) -->
    <!-- Gripper G2 TCP at (300, 0, 300) mm, rpy (180, 0, 0); joints (deg): 0 -36.184 -47.736 0 83.920 0 -->
    <key name="home" qpos="0 -0.631529917 -0.833150486 0 1.464680403 0 0 0 0 0 0 0" ctrl="0 -0.631529917 -0.833150486 0 1.464680403 0 0"/>
    <key name="ready" qpos="0 -0.631529917 -0.833150486 0 1.464680403 0 0 0 0 0 0 0" ctrl="0 -0.631529917 -0.833150486 0 1.464680403 0 0"/>
  </keyframe>
</mujoco>
"""
    # For meshes to be found from models/xarm6_with_gripper/, create assets dir with symlinks to both components
    assets_dir = target_dir / "assets"
    if assets_dir.is_symlink():
        assets_dir.unlink()
    assets_dir.mkdir(parents=True, exist_ok=True)

    xarm6_link = assets_dir / "xarm6"
    if not xarm6_link.exists():
        xarm6_link.symlink_to(Path("../../xarm6/assets/xarm6"))

    gripper_link = assets_dir / "gripper_g2"
    if not gripper_link.exists():
        gripper_link.symlink_to(Path("../../gripper_g2/assets/gripper_g2"))

    (target_dir / "xarm6_with_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "xarm6_with_gripper", "xarm6_with_gripper.xml", center="0.3 0 0.3", extent="0.9")


def build_xarm6_with_bio_gripper() -> None:
    print("Building xArm 6 + Bio Gripper G2 combo...")
    target_dir = MODELS_DIR / "xarm6_with_bio_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="xarm6_with_bio_gripper">
  <!-- UFACTORY xArm 6 equipped with Bio Gripper G2 (Roadmap 0.0.2). -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="black" rgba="0.25 0.25 0.25 1" specular="0.2" shininess="0.1"/>
    <material name="seal" rgba="0.8 0.82 0.85 1" specular="0.3" shininess="0.2"/>

    <!-- xArm 6 Visual OBJs -->
    <mesh name="link_base_vis" file="xarm6/visual/link_base.obj"/>
    <mesh name="link1_seal_vis" file="xarm6/visual/link1_seal.obj"/>
    <mesh name="link1_white_vis" file="xarm6/visual/link1_white.obj"/>
    <mesh name="link1_silver_vis" file="xarm6/visual/link1_silver.obj"/>
    <mesh name="link2_white_vis" file="xarm6/visual/link2_white.obj"/>
    <mesh name="link2_silver_vis" file="xarm6/visual/link2_silver.obj"/>
    <mesh name="link3_white_vis" file="xarm6/visual/link3_white.obj"/>
    <mesh name="link3_silver_vis" file="xarm6/visual/link3_silver.obj"/>
    <mesh name="link4_white_vis" file="xarm6/visual/link4_white.obj"/>
    <mesh name="link4_silver_vis" file="xarm6/visual/link4_silver.obj"/>
    <mesh name="link5_white_vis" file="xarm6/visual/link5_white.obj"/>
    <mesh name="link5_silver_vis" file="xarm6/visual/link5_silver.obj"/>
    <mesh name="link6_white_vis" file="xarm6/visual/link6_white.obj"/>
    <mesh name="link6_black_vis" file="xarm6/visual/link6_black.obj"/>
    <mesh name="link6_silver_vis" file="xarm6/visual/link6_silver.obj"/>

    <!-- xArm 6 Collision STLs -->
    <mesh name="link_base_col" file="xarm6/collision/link_base.stl"/>
    <mesh name="link1_col" file="xarm6/collision/link1.stl"/>
    <mesh name="link2_col" file="xarm6/collision/link2.stl"/>
    <mesh name="link3_col" file="xarm6/collision/link3.stl"/>
    <mesh name="link4_col" file="xarm6/collision/link4.stl"/>
    <mesh name="link5_col" file="xarm6/collision/link5.stl"/>
    <mesh name="link6_col" file="xarm6/collision/link6.stl"/>

    <!-- Bio Gripper G2 Visual OBJs -->
    <mesh name="bio_base_white_vis" file="bio_gripper_g2/visual/bio_gripper_g2_base_white.obj"/>
    <mesh name="bio_base_silver_vis" file="bio_gripper_g2/visual/bio_gripper_g2_base_silver.obj"/>
    <mesh name="bio_left_finger_vis" file="bio_gripper_g2/visual/bio_gripper_g2_left_finger.obj"/>
    <mesh name="bio_right_finger_vis" file="bio_gripper_g2/visual/bio_gripper_g2_right_finger.obj"/>

    <!-- Bio Gripper G2 Collision STLs -->
    <mesh name="bio_base_col" file="bio_gripper_g2/collision/bio_gripper_g2_link_base.stl"/>
    <mesh name="bio_left_finger_col" file="bio_gripper_g2/collision/bio_gripper_g2_left_finger.stl"/>
    <mesh name="bio_right_finger_col" file="bio_gripper_g2/collision/bio_gripper_g2_right_finger.stl"/>
  </asset>

  <default>
    <default class="xarm6">
      <joint axis="0 0 1" armature="0.1" range="-6.28319 6.28319" frictionloss="1"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3"/>
      </default>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator -->
      <default class="size1">
        <joint damping="10"/>
        <general gainprm="1500" biasprm="0 -1500 -150"/>
      </default>
      <default class="size2">
        <joint damping="5"/>
        <general gainprm="1000" biasprm="0 -1000 -100"/>
      </default>
      <default class="size3">
        <joint damping="2"/>
        <general gainprm="800" biasprm="0 -800 -80"/>
      </default>
      <default class="bio_slide">
        <joint type="slide" axis="0 1 0" damping="1"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="xarm6">
      <geom class="visual" mesh="link_base_vis" material="silver"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.267">
        <inertial pos="0.00022 0.02951 -0.0124" quat="0.976139 -0.216079 -0.00387305 -0.0211673" mass="2.3814"
          diaginertia="0.00585697 0.00544434 0.00312315"/>
        <joint name="joint1" class="size1"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_silver_vis" material="silver"/>
        <geom class="visual" mesh="link1_seal_vis" material="seal"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="0.707105 -0.707108 0 0">
          <inertial pos="0.03881 -0.22783 0.03496" quat="0.635305 0.753023 0.170976 -0.010531" mass="2.2675"
            diaginertia="0.0294477 0.0281783 0.00383449"/>
          <joint name="joint2" range="-2.059 2.0944" class="size1"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_silver_vis" material="silver"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.0535 -0.2845 0">
            <inertial pos="0.07041 0.11631 0.0107" quat="0.845838 0.425539 -0.293179 0.132369" mass="1.875"
              diaginertia="0.00714132 0.00699471 0.00196753"/>
            <joint name="joint3" range="-3.927 0.19198" class="size2"/>
            <geom class="visual" mesh="link3_white_vis" material="white"/>
            <geom class="visual" mesh="link3_silver_vis" material="silver"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.0775 0.3425 0" quat="0.707105 -0.707108 0 0">
              <inertial pos="-0.00018 0.01798 -0.02291" quat="0.99929 -0.0369702 0.00179429 -0.00701173" mass="1.3192"
                diaginertia="0.00489614 0.00447403 0.00132609"/>
              <joint name="joint4" class="size2"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_silver_vis" material="silver"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 0.707108 0 0">
                <inertial pos="0.0651 0.03096 0.00315" quat="-0.223008 0.771309 0.16027 0.574163" mass="1.33854"
                  diaginertia="0.00241824 0.00213367 0.00108731"/>
                <joint name="joint5" range="-1.69297 3.14159" class="size2"/>
                <geom class="visual" mesh="link5_white_vis" material="white"/>
                <geom class="visual" mesh="link5_silver_vis" material="silver"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0.076 0.097 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="0 -0.00677 -0.01098" quat="0.487612 0.512088 -0.512088 0.487612" mass="0.17"
                    diaginertia="0.000132176 9.3e-05 5.85236e-05"/>
                  <joint name="joint6" class="size3"/>
                  <geom class="visual" mesh="link6_white_vis" material="white"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="visual" mesh="link6_black_vis" material="black"/>
                  <geom class="collision" mesh="link6_col"/>

                  <!-- Mounted Bio Gripper G2 -->
                  <body name="bio_gripper_g2_base_link" pos="0.027018 -0.000715 -0.000692" quat="0.999994 0 0 -0.003531">
                    <inertial pos="0.016179 -2.3911E-05 0.023897" mass="0.38837" fullinertia="0.00088398 0.00018256 0.00094673 4.8665E-08 5.6276E-06 4.0828E-08"/>
                    <geom class="visual" mesh="bio_base_white_vis" material="white"/>
                    <geom class="visual" mesh="bio_base_silver_vis" material="silver"/>
                    <geom class="collision" mesh="bio_base_col" pos="-0.022475 -0.000830 -0.004696"/>

                    <body name="bio_gripper_g2_left_finger" pos="0.059 0 0.027">
                      <inertial pos="0.021846 -0.0276 0.011911" mass="0.012105" fullinertia="3.7322E-06 1.744E-05 1.7775E-05 4.1705E-06 -4.0142E-06 1.2237E-06"/>
                      <joint name="left_finger_joint" class="bio_slide" range="-0.0395 0"/>
                      <geom class="visual" mesh="bio_left_finger_vis" material="silver"/>
                      <geom class="collision" mesh="bio_left_finger_col" pos="-0.012870 -0.011316 -0.002219"/>
                    </body>

                    <body name="bio_gripper_g2_right_finger" pos="0.059 0 0.027">
                      <inertial pos="0.02037 0.027565 0.010274" mass="0.012068" fullinertia="4.1413E-06 1.6115E-05 1.6032E-05 -3.9682E-06 -4.5734E-06 -1.6711E-06"/>
                      <joint name="right_finger_joint" class="bio_slide" range="0 0.0395"/>
                      <geom class="visual" mesh="bio_right_finger_vis" material="silver"/>
                      <geom class="collision" mesh="bio_right_finger_col" pos="-0.013790 0.011745 -0.001039"/>
                    </body>

                    <site name="attachment_site" pos="0.135 0 0.055"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link6" body2="bio_gripper_g2_base_link"/>
  </contact>

  <equality>
    <joint joint1="right_finger_joint" joint2="left_finger_joint" polycoef="0 -1 0 0 0"/>
  </equality>

  <actuator>
    <general name="act1" joint="joint1" class="size1" forcerange="-179.4445 179.4445"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.059 2.0944" forcerange="-179.4445 179.4445"/>
    <general name="act3" joint="joint3" class="size2" ctrlrange="-3.927 0.19198" forcerange="-92 92"/>
    <general name="act4" joint="joint4" class="size2" forcerange="-81.558 81.558"/>
    <general name="act5" joint="joint5" class="size2" ctrlrange="-1.69297 3.14159" forcerange="-30.6 30.6"/>
    <general name="act6" joint="joint6" class="size3" forcerange="-30.6 30.6"/>
    <position name="finger_actuator" joint="left_finger_joint" kp="500" kv="5" ctrlrange="-0.0395 0"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
    <key name="ready" qpos="0 0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""
    # Create assets dir with symlinks to both components
    assets_dir = target_dir / "assets"
    if assets_dir.is_symlink():
        assets_dir.unlink()
    assets_dir.mkdir(parents=True, exist_ok=True)

    xarm6_link = assets_dir / "xarm6"
    if not xarm6_link.exists():
        xarm6_link.symlink_to(Path("../../xarm6/assets/xarm6"))

    bio_gripper_link = assets_dir / "bio_gripper_g2"
    if not bio_gripper_link.exists():
        bio_gripper_link.symlink_to(Path("../../bio_gripper_g2/assets/bio_gripper_g2"))

    (target_dir / "xarm6_with_bio_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "xarm6_with_bio_gripper", "xarm6_with_bio_gripper.xml", center="0.3 0 0.3", extent="0.9")


def build_lite6_with_gripper() -> None:
    print("Building Lite 6 + Gripper combo...")
    target_dir = MODELS_DIR / "lite6_with_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="lite6_with_gripper">
  <!-- UFACTORY Lite 6 equipped with Lite6 Parallel Gripper. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="gray" rgba="0.78 0.78 0.78 1" specular="0.3" shininess="0.2"/>
    <material name="seal" rgba="0.8 0.82 0.88 1" specular="0.3" shininess="0.2"/>
    <material name="black" rgba="0.15 0.15 0.15 1" specular="0.2" shininess="0.1"/>

    <!-- Lite 6 Visual OBJs -->
    <mesh name="link_base_white_vis" file="lite6/visual/link_base_white.obj"/>
    <mesh name="link_base_gray_vis" file="lite6/visual/link_base_gray.obj"/>
    <mesh name="link1_white_vis" file="lite6/visual/link1_white.obj"/>
    <mesh name="link1_gray_vis" file="lite6/visual/link1_gray.obj"/>
    <mesh name="link2_white_vis" file="lite6/visual/link2_white.obj"/>
    <mesh name="link2_gray_vis" file="lite6/visual/link2_gray.obj"/>
    <mesh name="link3_vis" file="lite6/visual/link3.obj"/>
    <mesh name="link4_white_vis" file="lite6/visual/link4_white.obj"/>
    <mesh name="link4_gray_vis" file="lite6/visual/link4_gray.obj"/>
    <mesh name="link5_vis" file="lite6/visual/link5.obj"/>
    <mesh name="link6_silver_vis" file="lite6/visual/link6_silver.obj"/>
    <mesh name="link6_seal_vis" file="lite6/visual/link6_seal.obj"/>

    <!-- Lite 6 Collision STLs -->
    <mesh name="link_base_col" file="lite6/collision/link_base.stl"/>
    <mesh name="link1_col" file="lite6/collision/link1.stl"/>
    <mesh name="link2_col" file="lite6/collision/link2.stl"/>
    <mesh name="link3_col" file="lite6/collision/link3.stl"/>
    <mesh name="link4_col" file="lite6/collision/link4.stl"/>
    <mesh name="link5_col" file="lite6/collision/link5.stl"/>
    <mesh name="link6_col" file="lite6/collision/link6.stl"/>

    <!-- Lite6 Gripper Visual OBJs -->
    <mesh name="shell_vis" file="lite6_gripper/visual/shell.obj"/>
    <mesh name="finger1_white_vis" file="lite6_gripper/visual/finger1_white.obj"/>
    <mesh name="finger1_silver_vis" file="lite6_gripper/visual/finger1_silver.obj"/>
    <mesh name="finger1_black_vis" file="lite6_gripper/visual/finger1_black.obj"/>
    <mesh name="finger2_white_vis" file="lite6_gripper/visual/finger2_white.obj"/>
    <mesh name="finger2_silver_vis" file="lite6_gripper/visual/finger2_silver.obj"/>
    <mesh name="finger2_black_vis" file="lite6_gripper/visual/finger2_black.obj"/>

    <!-- Lite6 Gripper Collision STLs -->
    <mesh name="shell_col" file="lite6_gripper/collision/shell.stl"/>
    <mesh name="finger1_col" file="lite6_gripper/collision/finger1.stl"/>
    <mesh name="finger2_col" file="lite6_gripper/collision/finger2.stl"/>
  </asset>

  <default>
    <default class="lite6">
      <joint axis="0 0 1" armature="0.05" damping="1"/>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator;
           direct-drive wrist J4-J6 uses the retuned kp/kv and damping from the same spec -->
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" mass="0" density="0"/>
      </default>
      <default class="lg_slide">
        <joint type="slide" axis="0 1 0" damping="1"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="lite6">
      <inertial pos="-0.00829545 3.26357e-05 0.0631195" mass="1.65394" diaginertia="0.003 0.003 0.002"/>
      <geom class="visual" mesh="link_base_white_vis" material="white"/>
      <geom class="visual" mesh="link_base_gray_vis" material="gray"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.2435">
        <inertial pos="-0.00036 0.04195 -0.0025" quat="0.608059 0.792349 -0.0438707 0.0228718" mass="1.411"
          diaginertia="0.00145276 0.00135275 0.000853355"/>
        <joint name="joint1" range="-6.28319 6.28319"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_gray_vis" material="gray"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="-0.5 0.5 0.5 0.5">
          <inertial pos="0.179 0 0.0584" quat="0.417561 0.571619 0.569585 0.417693" mass="1.34"
            diaginertia="0.00560971 0.0052152 0.00122018"/>
          <joint name="joint2" range="-2.61799 2.61799"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_gray_vis" material="gray"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.2002 0 0" quat="0 -0.707105 -0.707108 0">
            <inertial pos="0.072 -0.0357 -0.001" quat="0.128259 0.662963 -0.167256 0.71837" mass="0.953"
              diaginertia="0.0018521 0.00175546 0.000703807"/>
            <joint name="joint3" range="-0.061087 5.23599"/>
            <geom class="visual" mesh="link3_vis" material="white"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.087 -0.22761 0" quat="0.707105 0.707108 0 0">
              <inertial pos="-0.002 -0.0285 -0.0813" quat="0.975248 0.22109 0.00203498 -0.00262178" mass="1.284"
                diaginertia="0.00370503 0.00349091 0.00109586"/>
              <joint name="joint4" range="-6.28319 6.28319" damping="0"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_gray_vis" material="gray"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 0.707108 0 0">
                <inertial pos="0 0.01 0.0019" quat="0.71423 0.696388 -0.0531933 0.0456997" mass="0.804"
                  diaginertia="0.000567553 0.000529266 0.000507681"/>
                <joint name="joint5" range="-2.1642 2.1642" damping="0.1"/>
                <geom class="visual" mesh="link5_vis" material="white"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0 0.0625 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="0 -0.00194 -0.0102" quat="-0.0376023 0.704057 0.0446838 0.707738" mass="0.13"
                    diaginertia="0.000148148 8.57757e-05 7.71412e-05"/>
                  <joint name="joint6" range="-6.28319 6.28319" damping="0"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="visual" mesh="link6_seal_vis" material="seal"/>
                  <geom class="collision" mesh="link6_col"/>

                  <!-- Mounted Lite6 Gripper -->
                  <body name="uflite_gripper_link">
                    <inertial pos="0 0 0.03" mass="0.25" fullinertia="0.00047106 0.00033072 0.00025642 0.00000039292 0.0000026537 -0.000010975"/>
                    <geom class="visual" mesh="shell_vis" material="white"/>
                    <geom class="collision" mesh="shell_col"/>

                    <body name="uflite_finger1" pos="0 0 0.0543">
                      <inertial pos="0 0.01 0.0086" mass="0.0163" fullinertia="1.425e-06 1.63e-06 8.63e-07 0 0 -3.1e-07"/>
                      <joint name="finger_joint1" class="lg_slide" range="0 0.0089"/>
                      <geom class="visual" mesh="finger1_white_vis" material="white"/>
                      <geom class="visual" mesh="finger1_silver_vis" material="silver"/>
                      <geom class="visual" mesh="finger1_black_vis" material="black"/>
                      <geom class="collision" mesh="finger1_col"/>
                      <geom class="collision" type="box" size="0.006 0.001 0.005" pos="0 -0.011 0.006"
                        friction="2 0.01 0.001"/>
                    </body>

                    <body name="uflite_finger2" pos="0 0 0.0543">
                      <inertial pos="0 -0.01 0.0086" mass="0.0163" fullinertia="1.425e-06 1.63e-06 8.63e-07 0 0 3.1e-07"/>
                      <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.0089" damping="1"/>
                      <geom class="visual" mesh="finger2_white_vis" material="white"/>
                      <geom class="visual" mesh="finger2_silver_vis" material="silver"/>
                      <geom class="visual" mesh="finger2_black_vis" material="black"/>
                      <geom class="collision" mesh="finger2_col"/>
                      <geom class="collision" type="box" size="0.006 0.001 0.005" pos="0 0.011 0.006"
                        friction="2 0.01 0.001"/>
                    </body>

                    <site name="attachment_site" pos="0 0 0.0836"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link4" body2="link6"/>
    <exclude body1="link6" body2="uflite_gripper_link"/>
    <exclude body1="link5" body2="uflite_gripper_link"/>
    <exclude body1="link4" body2="uflite_gripper_link"/>
    <exclude body1="uflite_finger1" body2="uflite_finger2"/>
  </contact>

  <equality>
    <joint joint1="finger_joint2" joint2="finger_joint1" polycoef="0 1 0 0 0"/>
  </equality>

  <actuator>
    <position joint="joint1" ctrlrange="-6.28319 6.28319" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint2" ctrlrange="-2.61799 2.61799" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint3" ctrlrange="-0.061087 5.23599" kp="2000" kv="200" forcerange="-30.6 30.6"/>
    <position joint="joint4" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
    <position joint="joint5" ctrlrange="-2.1642 2.1642" kp="80" kv="16" forcerange="-3.274812 3.274812"/>
    <position joint="joint6" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
    <position name="finger_actuator" joint="finger_joint1" kp="200" kv="10" ctrlrange="0 0.0089"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0 0 0" ctrl="0 0 0 0 0 0 0"/>
    <!-- Studio factory ready pose (deg): 0 9.9 31.8 0 21.9 0 -->
    <key name="ready" qpos="0 0.172787595947 0.555014702134 0 0.382227106187 0 0 0" ctrl="0 0.172787595947 0.555014702134 0 0.382227106187 0 0"/>
  </keyframe>
</mujoco>
"""
    assets_dir = target_dir / "assets"
    if assets_dir.is_symlink():
        assets_dir.unlink()
    assets_dir.mkdir(parents=True, exist_ok=True)

    lite6_link = assets_dir / "lite6"
    if not lite6_link.exists():
        lite6_link.symlink_to(Path("../../lite6/assets/lite6"))

    gripper_link = assets_dir / "lite6_gripper"
    if not gripper_link.exists():
        gripper_link.symlink_to(Path("../../lite6_gripper/assets/lite6_gripper"))

    (target_dir / "lite6_with_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "lite6_with_gripper", "lite6_with_gripper.xml", center="0.25 0 0.25", extent="0.7")


def build_lite6_with_vacuum_gripper() -> None:
    print("Building Lite 6 + Vacuum Gripper combo...")
    target_dir = MODELS_DIR / "lite6_with_vacuum_gripper"
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_content = """<mujoco model="lite6_with_vacuum_gripper">
  <!-- UFACTORY Lite 6 equipped with Vacuum Suction Gripper. -->
  <compiler angle="radian" autolimits="true" meshdir="assets"/>

  <option integrator="implicitfast"/>

  <asset>
    <!-- Materials -->
    <material name="white" rgba="0.95 0.95 0.95 1" specular="0.4" shininess="0.3"/>
    <material name="silver" rgba="0.85 0.85 0.88 1" specular="0.9" shininess="0.8" reflectance="0.05"/>
    <material name="gray" rgba="0.78 0.78 0.78 1" specular="0.3" shininess="0.2"/>
    <material name="seal" rgba="0.8 0.82 0.88 1" specular="0.3" shininess="0.2"/>

    <!-- Lite 6 Visual OBJs -->
    <mesh name="link_base_white_vis" file="lite6/visual/link_base_white.obj"/>
    <mesh name="link_base_gray_vis" file="lite6/visual/link_base_gray.obj"/>
    <mesh name="link1_white_vis" file="lite6/visual/link1_white.obj"/>
    <mesh name="link1_gray_vis" file="lite6/visual/link1_gray.obj"/>
    <mesh name="link2_white_vis" file="lite6/visual/link2_white.obj"/>
    <mesh name="link2_gray_vis" file="lite6/visual/link2_gray.obj"/>
    <mesh name="link3_vis" file="lite6/visual/link3.obj"/>
    <mesh name="link4_white_vis" file="lite6/visual/link4_white.obj"/>
    <mesh name="link4_gray_vis" file="lite6/visual/link4_gray.obj"/>
    <mesh name="link5_vis" file="lite6/visual/link5.obj"/>
    <mesh name="link6_silver_vis" file="lite6/visual/link6_silver.obj"/>
    <mesh name="link6_seal_vis" file="lite6/visual/link6_seal.obj"/>

    <!-- Lite 6 Collision STLs -->
    <mesh name="link_base_col" file="lite6/collision/link_base.stl"/>
    <mesh name="link1_col" file="lite6/collision/link1.stl"/>
    <mesh name="link2_col" file="lite6/collision/link2.stl"/>
    <mesh name="link3_col" file="lite6/collision/link3.stl"/>
    <mesh name="link4_col" file="lite6/collision/link4.stl"/>
    <mesh name="link5_col" file="lite6/collision/link5.stl"/>
    <mesh name="link6_col" file="lite6/collision/link6.stl"/>

    <!-- Vacuum Gripper Visual OBJ & Collision STL -->
    <mesh name="vacuum_vis" file="lite6_vacuum_gripper/visual/vacuum_gripper.obj"/>
    <mesh name="vacuum_col" file="lite6_vacuum_gripper/collision/vacuum_gripper_lite.stl"/>
  </asset>

  <default>
    <default class="lite6">
      <joint axis="0 0 1" armature="0.05" damping="1"/>
      <!-- per-joint torque limits follow the official UFACTORY spec, set as forcerange on each actuator;
           direct-drive wrist J4-J6 uses the retuned kp/kv and damping from the same spec -->
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom type="mesh" group="3" mass="0" density="0"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
  </default>

  <worldbody>
    <body name="link_base" childclass="lite6">
      <inertial pos="-0.00829545 3.26357e-05 0.0631195" mass="1.65394" diaginertia="0.003 0.003 0.002"/>
      <geom class="visual" mesh="link_base_white_vis" material="white"/>
      <geom class="visual" mesh="link_base_gray_vis" material="gray"/>
      <geom class="collision" mesh="link_base_col"/>
      <body name="link1" pos="0 0 0.2435">
        <inertial pos="-0.00036 0.04195 -0.0025" quat="0.608059 0.792349 -0.0438707 0.0228718" mass="1.411"
          diaginertia="0.00145276 0.00135275 0.000853355"/>
        <joint name="joint1" range="-6.28319 6.28319"/>
        <geom class="visual" mesh="link1_white_vis" material="white"/>
        <geom class="visual" mesh="link1_gray_vis" material="gray"/>
        <geom class="collision" mesh="link1_col"/>
        <body name="link2" quat="-0.5 0.5 0.5 0.5">
          <inertial pos="0.179 0 0.0584" quat="0.417561 0.571619 0.569585 0.417693" mass="1.34"
            diaginertia="0.00560971 0.0052152 0.00122018"/>
          <joint name="joint2" range="-2.61799 2.61799"/>
          <geom class="visual" mesh="link2_white_vis" material="white"/>
          <geom class="visual" mesh="link2_gray_vis" material="gray"/>
          <geom class="collision" mesh="link2_col"/>
          <body name="link3" pos="0.2002 0 0" quat="0 -0.707105 -0.707108 0">
            <inertial pos="0.072 -0.0357 -0.001" quat="0.128259 0.662963 -0.167256 0.71837" mass="0.953"
              diaginertia="0.0018521 0.00175546 0.000703807"/>
            <joint name="joint3" range="-0.061087 5.23599"/>
            <geom class="visual" mesh="link3_vis" material="white"/>
            <geom class="collision" mesh="link3_col"/>
            <body name="link4" pos="0.087 -0.22761 0" quat="0.707105 0.707108 0 0">
              <inertial pos="-0.002 -0.0285 -0.0813" quat="0.975248 0.22109 0.00203498 -0.00262178" mass="1.284"
                diaginertia="0.00370503 0.00349091 0.00109586"/>
              <joint name="joint4" range="-6.28319 6.28319" damping="0"/>
              <geom class="visual" mesh="link4_white_vis" material="white"/>
              <geom class="visual" mesh="link4_gray_vis" material="gray"/>
              <geom class="collision" mesh="link4_col"/>
              <body name="link5" quat="0.707105 0.707108 0 0">
                <inertial pos="0 0.01 0.0019" quat="0.71423 0.696388 -0.0531933 0.0456997" mass="0.804"
                  diaginertia="0.000567553 0.000529266 0.000507681"/>
                <joint name="joint5" range="-2.1642 2.1642" damping="0.1"/>
                <geom class="visual" mesh="link5_vis" material="white"/>
                <geom class="collision" mesh="link5_col"/>
                <body name="link6" pos="0 0.0625 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="0 -0.00194 -0.0102" quat="-0.0376023 0.704057 0.0446838 0.707738" mass="0.13"
                    diaginertia="0.000148148 8.57757e-05 7.71412e-05"/>
                  <joint name="joint6" range="-6.28319 6.28319" damping="0"/>
                  <geom class="visual" mesh="link6_silver_vis" material="silver"/>
                  <geom class="visual" mesh="link6_seal_vis" material="seal"/>
                  <geom class="collision" mesh="link6_col"/>

                  <!-- Mounted Vacuum Gripper -->
                  <body name="uflite_vacuum_gripper_link">
                    <inertial pos="0 0 0.03" mass="0.14" fullinertia="0.00047106 0.00033072 0.00025642 0.00000039292 0.0000026537 -0.000010975"/>
                    <geom class="visual" mesh="vacuum_vis" material="white"/>
                    <geom class="collision" mesh="vacuum_col"/>
                    <site name="suction_site" pos="0 0 0.061" size="0.01" rgba="0 1 0 0.5"/>
                    <site name="attachment_site" pos="0 0 0.061"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link_base" body2="link1"/>
    <exclude body1="link1" body2="link2"/>
    <exclude body1="link2" body2="link3"/>
    <exclude body1="link3" body2="link4"/>
    <exclude body1="link4" body2="link5"/>
    <exclude body1="link5" body2="link6"/>
    <exclude body1="link4" body2="link6"/>
    <exclude body1="link6" body2="uflite_vacuum_gripper_link"/>
    <exclude body1="link5" body2="uflite_vacuum_gripper_link"/>
    <exclude body1="link4" body2="uflite_vacuum_gripper_link"/>
  </contact>

  <actuator>
    <position joint="joint1" ctrlrange="-6.28319 6.28319" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint2" ctrlrange="-2.61799 2.61799" kp="2000" kv="200" forcerange="-81.6 81.6"/>
    <position joint="joint3" ctrlrange="-0.061087 5.23599" kp="2000" kv="200" forcerange="-30.6 30.6"/>
    <position joint="joint4" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
    <position joint="joint5" ctrlrange="-2.1642 2.1642" kp="80" kv="16" forcerange="-3.274812 3.274812"/>
    <position joint="joint6" ctrlrange="-6.28319 6.28319" kp="40" kv="8" forcerange="-1.364505 1.364505"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0" ctrl="0 0 0 0 0 0"/>
    <!-- Studio factory ready pose (deg): 0 9.9 31.8 0 21.9 0 -->
    <key name="ready" qpos="0 0.172787595947 0.555014702134 0 0.382227106187 0" ctrl="0 0.172787595947 0.555014702134 0 0.382227106187 0"/>
  </keyframe>
</mujoco>
"""
    assets_dir = target_dir / "assets"
    if assets_dir.is_symlink():
        assets_dir.unlink()
    assets_dir.mkdir(parents=True, exist_ok=True)

    lite6_link = assets_dir / "lite6"
    if not lite6_link.exists():
        lite6_link.symlink_to(Path("../../lite6/assets/lite6"))

    vacuum_link = assets_dir / "lite6_vacuum_gripper"
    if not vacuum_link.exists():
        vacuum_link.symlink_to(Path("../../lite6_vacuum_gripper/assets/lite6_vacuum_gripper"))

    (target_dir / "lite6_with_vacuum_gripper.xml").write_text(xml_content, encoding="utf-8")
    write_scene_file(target_dir, "lite6_with_vacuum_gripper", "lite6_with_vacuum_gripper.xml", center="0.25 0 0.25", extent="0.7")



def verify_models() -> None:
    print("\nVerifying all models with MuJoCo...")
    models_to_test = [
        MODELS_DIR / "xarm6/scene.xml",
        MODELS_DIR / "xarm7/scene.xml",
        MODELS_DIR / "xarm5/scene.xml",
        MODELS_DIR / "uf850/scene.xml",
        MODELS_DIR / "lite6/scene.xml",
        MODELS_DIR / "gripper_g2/scene.xml",
        MODELS_DIR / "bio_gripper_g2/scene.xml",
        MODELS_DIR / "lite6_gripper/scene.xml",
        MODELS_DIR / "lite6_vacuum_gripper/scene.xml",
        MODELS_DIR / "xarm6_with_gripper/scene.xml",
        MODELS_DIR / "xarm5_with_gripper/scene.xml",
        MODELS_DIR / "xarm7_with_gripper/scene.xml",
        MODELS_DIR / "uf850_with_gripper/scene.xml",
        MODELS_DIR / "xarm6_with_bio_gripper/scene.xml",
        MODELS_DIR / "lite6_with_gripper/scene.xml",
        MODELS_DIR / "lite6_with_vacuum_gripper/scene.xml",
    ]

    for model_path in models_to_test:
        try:
            m = mujoco.MjModel.from_xml_path(str(model_path))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            print(f"  {model_path.parent.name:25s}: PASS (nq={m.nq}, nu={m.nu}, nbody={m.nbody}, ngeom={m.ngeom})")
        except Exception as e:
            print(f"  {model_path.parent.name:25s}: FAILED - {e}")
            raise


def main() -> None:
    build_xarm7()
    build_xarm5()
    build_uf850()
    build_lite6()
    build_gripper_g2()
    build_bio_gripper_g2()
    build_lite6_gripper()
    build_lite6_vacuum_gripper()
    build_xarm6_with_gripper()
    build_xarm6_with_bio_gripper()
    build_lite6_with_gripper()
    build_lite6_with_vacuum_gripper()
    verify_models()
    print("\nAll models built and verified successfully!")


if __name__ == "__main__":
    main()
