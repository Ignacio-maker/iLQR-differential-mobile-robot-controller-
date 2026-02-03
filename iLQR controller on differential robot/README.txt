# iLQR Controller Implementation for Differential Drive Robot

This repository contains the source code for the implementation of an **Iterative Linear Quadratic Regulator (iLQR)** controller on a differential drive mobile robot. 

The project focuses on trajectory tracking and autonomous navigation. It includes simulations, hardware calibration tools, and real-world implementation scripts using ROS (Robot Operating System).

> **Note:** Each script in this repository is designed to run independently.

## 🛠️ Hardware Setup

The code was tested and prototyped using the following hardware configuration:

* **Computing Unit:** NVIDIA Jetson Nano
* **Motor Driver:** L298N H-Bridge
* **Actuators:** 2x DC Motors with Encoders
* **Sensors:** LiDAR (for the ROS implementation)

## 📂 Project Structure & Description

### 1. Simulations
This folder contains the simulated behavior of the iLQR controller prior to hardware implementation. These scripts allow for validation of the control logic in a virtual environment.
* `ilqr_circle_path_sim`: Simulation of the robot tracking a circular path.
* `ilqr_lemniscat_path_sim`: Simulation of the robot tracking a lemniscate (figure-eight) path.
* `ilqr_line_path_sim`: Simulation of the robot tracking a straight line.

### 2. PWM_motor_tester
Contains utilities for hardware calibration and system identification.
* `pwm_percent_controller`: Code used to calibrate and measure the output torque generated at different PWM duty cycles. This ensures accurate mapping between control signals and motor response.

### 3. Trajectory Control
This directory contains the standalone controller implementations for specific trajectories on the physical robot.
* `ilqr_circle_path`: Implementation of the iLQR controller for a circular trajectory.
* `ilqr_line_path`: Implementation of the iLQR controller for a straight-line trajectory.
* **Trajectory_Ploting/**:
    * Contains versions of the scripts above (`ilqr_circle_path_plot`, `ilqr_line_path_plot`) that include real-time plotting functions to visualize the robot's performance and error correction during operation.

### 4. ROS_subscriber_and_controller
This folder contains the integration of the iLQR controller within a ROS environment for autonomous navigation.
* `ilqr_controller_ros_sub`: This node implements the controller in conjunction with a LiDAR sensor. It receives signals to apply circular or straight trajectories for obstacle avoidance and simultaneous mapping of the environment.

**Dependency:**
This ROS code is designed to work in tandem with the navigation stack developed by my project partner. You can find the corresponding navigation and mapping repository here:
* https://github.com/zdenc0de/autonomous-cart

## 🚀 Usage

Since each code is independent:
1. Ensure your hardware is connected according to the pin definitions in the specific script you wish to run.
2. If running the **ROS** implementation, ensure the `roscore` and the partner's navigation node are active before launching the subscriber.
3. For **Simulations** or **Trajectory Control**, simply execute the Python script directly on the Jetson Nano.