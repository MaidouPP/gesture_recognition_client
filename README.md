# ROS wrapper of the gesture recognizer

This repo is supposed to work along with the gesture recognizer in CogRob workspace.

## Preparations

1. Clone this repo to a local catkin workspace.
2. Clone the `workspace` anywhere.
3. Clone `use_cogrob_workspace` from `cogrob_ros`, which is [here](https://github.com/CogRob/cogrob_ros/tree/master/use_cogrob_workspace).
4. Follow the instruction in the `use_cogrob_workspace`. Remember to sync files anytime you modified the workspace codes (gesture part).
5. Get trained HMM model ready.

## Run
`rosrun gesture_recognition_client gesture_recognition_client_node.py`
