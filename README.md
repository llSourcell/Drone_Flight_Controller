# Drone Flight Controller

## Overview 

This is the code for [this](https://youtu.be/PngA5YLFuvU) video on Youtube by Siraj Raval on Deep Deterministic Policy Gradients. Its apart of week 9 of 10 of the Move 37 Course at School of AI. 

# Table of Contents

- [Install](#install)
- [Download](#download)
- [Develop](#develop)
- [Submit](#submit)


# Install

This project uses ROS (Robot Operating System) as the primary communication mechanism between your agent and the simulation. You can either install it on your own machine ("native install").

## ROS Virtual Machine

Download the compressed VM disk image and unzip it:

- Compressed VM Disk Image: [RoboVM_V2.1.0.zip](https://s3-us-west-1.amazonaws.com/udacity-robotics/Virtual+Machines/Lubuntu_071917/RoboVM_V2.1.0.zip)
- MD5 checksum: `MD5(Ubuntu 64-bit Robo V2.1.0.ova)= 95bfba89fbdac5f2c0a2be2ae186ddbb`

You will need a Virtual Machine player to run the VM, such as VMWare or VirtualBox:

- [VMWare](http://www.vmware.com/): If you use a Windows/Linux system, you can get [Workstation Player](https://www.vmware.com/products/workstation-player.html) for free, or if you're on a Mac, you can get a trial of [Fusion](https://www.vmware.com/products/fusion.html).
- [VirtualBox](https://www.virtualbox.org/): Download and install the appropriate version for your system.

Open your VM player, and then "Open" / "Import" the VM disk image that you just unzipped (the `.ova` file).

Configure the settings for your VM to allocate at least 2 processors and 4GB of RAM (more the merrier!). Now launch the VM, and follow the on-screen instructions for one-time setup steps.

- Username: `robond`
- Password: `robo-nd`

To open a terminal in your VM, press `Ctrl+Alt+T`. If prompted "Do you want to source ROS?", answer `y` (yes). This is where you will execute your project code.

## ROS Native Install

If you choose to install ROS (Robot Operating System) on your own machine, it is recommended that you use Ubuntu 16.04 LTS as your operating system. To install ROS, please follow the instructions here: [ROS Installation](http://wiki.ros.org/kinetic/Installation)

# Download

## Project Code

On the machine where you have installed ROS (a VM, or your local machine), create a directory named `catkin_ws`, and inside it create a subdirectory named `src`. If you're using a VM, you can also share a folder on your file-system between the host and VM. That might make it easier for you to prepare your report and submit your project for review.

Now clone this repository or download it inside the `src` directory. This is where you will develop your project code. Your folder structure should look like the following (ROS has a fairly complicated build system, as you will see!):

```
- ~/catkin_ws/
  - src/
    - RL-Quadcopter/
      - quad_controller_rl/
        - ...
```

The root of this structure (`catkin_ws`) is a [catkin workspace](http://wiki.ros.org/catkin/workspaces), which you can use to organize and work on all your ROS-based projects (the name `catkin_ws` is not mandatory - you can change it to anything you want).

## Simulator

Download the Udacity Quadcopter Simulator, nicknamed **DroneSim**, for your host computer OS [here](https://github.com/udacity/RoboND-Controls-Lab/releases). 

To start the simulator, simply run the downloaded executable file. You may need to run the simulator _after_ the `roslaunch` step mentioned below in the Run section, so that it can connect to a running ROS master.

_Note: If you are using a Virtual Machine (VM), you cannot run the simulator inside the VM. You have to download and run the simulator for your **host operating system** and connect it to your VM (see below)._

### Connecting the Simulator to a VM

If you are running ROS in a VM, there are a couple of steps necessary to make sure it can communicate with the simulator running on your host system. If not using a VM, these steps are not needed.

#### Enable Networking on VM

- **VMWare**: The default setting should work. To verify, with the VM running, go to the Virtual Machine menu > Network Adapter. NAT should be selected.
- **VirtualBox**:
  1. In the VirtualBox Manager, go to Global Tools (top-right corner) > Host Network Manager.
  2. Create a new Host-only Network. You can leave the default settings, e.g. Name = "vboxnet0", Ipv4 Address/Mask = "192.168.56.1/24", and DHCP Server enabled.
  3. Switch back to Machine Tools, and with your VM selected, open its Settings.
  4. Go to the Network tab, change "Attached to" (network type) to "Host-only Adapter", and pick "vboxnet0" from the "Name" dropdown.
  5. Hit Ok to save, and (re)start the VM.

#### Obtain IP Addresses for Host and VM

In a terminal on your host computer, run `ifconfig`. It will list all the network interfaces available, both physical and virtual. There should be one named something like `vmnet` or `vboxnet`. Note the IP address (`inet` or `inet addr`) mentioned for that interface, e.g. `192.168.56.1`. This is your **Host IP address**.

Do the same inside the VM. Here the interface may have a different name, but the IP address should have a common prefix. Note down the complete IP address, e.g. `192.168.56.101` - this your **VM IP address**.

#### Edit Simulator Settings

Inside the simulator's `_Data` or `/Contents` folder (on Mac, right-click the app > Show Package Contents), edit `ros_settings.txt`:

- Set `vm-ip` to the **VM IP address** and set `vm-override` to `true`.
- Set `host-ip` to the **Host IP address** and set `host-override` to `true`.

The host and/or VM's IP address can change when it is restarted. If you are experiencing connectivity problems, be sure to check that the actual IP addresses match what you have in `ros_settings.txt`.


# Develop

Starter code is provided in `quad_controller_rl/` with all the Python modules (`.py` files) under the `src/quad_controller_rl/` package, and the main project notebook under `notebooks/`. Take a look at the files there, but you do not have to make any changes to the code at this point. Complete the following two steps first (**Build** and **Run**), to ensure your ROS installation is working correctly.

## Build

To prepare your code to run with ROS, you will first need to build it. This compiles and links different modules ("ROS nodes") needed for the project. Fortunately, you should only need to do this once, since changes to Python scripts don't need recompilation.

- Go to your catkin workspace (`catkin_ws/`):

```bash
$ cd ~/catkin_ws/
```

- Build ROS nodes:

```bash
$ catkin_make
```

- Enable command-line tab-completion and some other useful ROS utilities:

```bash
$ source devel/setup.bash
```

## Run

To run your project, start ROS with the `rl_controller.launch` file:

```bash
$ roslaunch quad_controller_rl rl_controller.launch
```

You should see a few messages on the terminal as different nodes get spun up. Now you can run the simulator, which is a separate Unity application (note that you must start ROS first, and then run the simulator). Once the simulator initializes itself, you should start seeing additional messages in your ROS terminal, indicating a new episode starting every few seconds. The quadcopter in the simulation should show its blades running as it gets control inputs from the agent, and it should reset at the beginning of each episode.

Tip: If you get tired of this two-step startup process, edit the `quad_controller_rl/scripts/drone_sim` script and enter a command that runs the simulator application. It will then be launched automatically with ROS!

## Implement

Once you have made sure ROS and the simulator are running without any errors, and that they can communicate with each other, try modifying the code in `agents/policy_search.py` - this is a sample agent that runs by default (e.g. add a `print` statement). Every time you make a change, you will need to stop the simulator (press `Esc` with the simulator window active), and shutdown ROS (press `Ctrl+C` in the terminal). Save your change, and `roslaunch` again.

Now you should be ready to start coding! Open the project notebook for further instructions (assuming you are in your catkin workspace):

```bash
$ jupyter notebook src/RL-Quadcopter/quad_controller_rl/notebooks/RL-Quadcopter.ipynb
```

## Credits 

Credits for this code go to [sksq96](https://github.com/sksq96/deep-rl-quadcopter). I've merely created a wrapper to get people started. 
