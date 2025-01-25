# UR5e with Cube Environment Documentation

This environment simulates a UR5e robot manipulating a cube in MuJoCo. The environment is designed for control and reinforcement learning tasks.


## Description

The UR5e with Cube environment is a simulation environment that models a UR5e robot manipulating a cube in a MuJoCo physics simulation. The environment provides an interface for controlling the robot and observing its state, allowing for the development and testing of control algorithms and reinforcement learning agents.

---

## Action Space (`action_space`)

The action space defines the control inputs to the robot. It includes:

| Num | Action                  | Control Min | Control Max | Name (in corresponding XML file) | Joint                  | Type (Unit)         |
|-----|-------------------------|-------------|-------------|-----------------------------------|------------------------|---------------------|
| 1   | Shoulder pan joint      | -6.2831     | 6.2831      | `shoulder_pan`                    | `shoulder_pan_joint`   | Torque (Nm)         |
| 2   | Shoulder lift joint     | -6.2831     | 6.2831      | `shoulder_lift`                   | `shoulder_lift_joint`  | Torque (Nm)         |
| 3   | Elbow joint             | -3.1415     | 3.1415      | `elbow`                           | `elbow_joint`          | Torque (Nm)         |
| 4   | Wrist 1 joint           | -6.2831     | 6.2831      | `wrist_1`                         | `wrist_1_joint`        | Torque (Nm)         |
| 5   | Wrist 2 joint           | -6.2831     | 6.2831      | `wrist_2`                         | `wrist_2_joint`        | Torque (Nm)         |
| 6   | Wrist 3 joint           | -6.2831     | 6.2831      | `wrist_3`                         | `wrist_3_joint`        | Torque (Nm)         |
| 7   | Gripper fingers         | 0           | 255         | `fingers_actuator`                | `left_driver_joint`    | Position (0-255)    |

---

## Observation Space (`observation_space`)

This action space is a `Box(19, float32)` where the first 12 values are the joint positions of the UR5e robot and the last 7 values are the position and orientation of the cube.

| Num | Observation               | Min       | Max       | Name (in corresponding XML file) | Joint                  | Type (Unit)         |
|-----|---------------------------|-----------|-----------|-----------------------------------|------------------------|---------------------|
| 1   | Shoulder pan joint        | -∞        | ∞         | `shoulder_pan_joint`              | Rotational             | Angle (radians)     |
| 2   | Shoulder lift joint       | -∞        | ∞         | `shoulder_lift_joint`             | Rotational             | Angle (radians)     |
| 3   | Elbow joint               | -∞        | ∞         | `elbow_joint`                     | Rotational             | Angle (radians)     |
| 4   | Wrist 1 joint             | -∞        | ∞         | `wrist_1_joint`                   | Rotational             | Angle (radians)     |
| 5   | Wrist 2 joint             | -∞        | ∞         | `wrist_2_joint`                   | Rotational             | Angle (radians)     |
| 6   | Wrist 3 joint             | -∞        | ∞         | `wrist_3_joint`                   | Rotational             | Angle (radians)     |
| 7   | Left driver joint         | -∞        | ∞         | `left_driver_joint`               | Rotational             | Angle (radians)     |
| 8   | Left spring joint         | -∞        | ∞         | `left_spring_joint`               | Rotational             | Angle (radians)     |
| 9   | Left follower joint       | -∞        | ∞         | `left_follower_joint`             | Rotational             | Angle (radians)     |
| 10  | Right driver joint        | -∞        | ∞         | `right_driver_joint`              | Rotational             | Angle (radians)     |
| 11  | Right spring joint        | -∞        | ∞         | `right_spring_link`               | Rotational             | Angle (radians)     |
| 12  | Right follower joint      | -∞        | ∞         | `right_follower_joint`            | Rotational             | Angle (radians)     |
| 13  | Cube position (x)         | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 14  | Cube position (y)         | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 15  | Cube position (z)         | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 16  | Cube orientation (qw)     | -∞        | ∞         | N/A (free body)                   | Rotational (quaternion)| Unitless            |
| 17  | Cube orientation (qx)     | -∞        | ∞         | N/A (free body)                   | Rotational (quaternion)| Unitless            |
| 18  | Cube orientation (qy)     | -∞        | ∞         | N/A (free body)                   | Rotational (quaternion)| Unitless            |
| 19  | Cube orientation (qz)     | -∞        | ∞         | N/A (free body)                   | Rotational (quaternion)| Unitless            |
| 20  | Target position (x)       | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 21  | Target position (y)       | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 22  | Target position (z)       | -∞        | ∞         | N/A (free body)                   | Translational          | Position (meters)   |
| 23  | Shoulder pan joint velocity| -∞        | ∞         | `shoulder_pan_joint`              | Rotational             | Velocity (rad/s)    |
| 24  | Shoulder lift joint velocity| -∞        | ∞         | `shoulder_lift_joint`             | Rotational             | Velocity (rad/s)    |
| 25  | Elbow joint velocity       | -∞        | ∞         | `elbow_joint`                     | Rotational             | Velocity (rad/s)    |
| 26  | Wrist 1 joint velocity     | -∞        | ∞         | `wrist_1_joint`                   | Rotational             | Velocity (rad/s)    |
| 27  | Wrist 2 joint velocity     | -∞        | ∞         | `wrist_2_joint`                   | Rotational             | Velocity (rad/s)    |
| 28  | Wrist 3 joint velocity     | -∞        | ∞         | `wrist_3_joint`                   | Rotational             | Velocity (rad/s)    |
| 29  | Left driver joint velocity | -∞        | ∞         | `left_driver_joint`               | Rotational             | Velocity (rad/s)    |
| 30  | Left spring joint velocity | -∞        | ∞         | `left_spring_joint`               | Rotational             | Velocity (rad/s)    |
| 31  | Left follower joint velocity| -∞        | ∞         | `left_follower_joint`             | Rotational             | Velocity (rad/s)    |
| 32  | Right driver joint velocity| -∞        | ∞         | `right_driver_joint`              | Rotational             | Velocity (rad/s)    |
| 33  | Right spring joint velocity| -∞        | ∞         | `right_spring_link`               | Rotational             | Velocity (rad/s)    |
| 34  | Right follower joint velocity| -∞        | ∞         | `right_follower_joint`            | Rotational             | Velocity (rad/s)    |
---
