cmake_minimum_required(VERSION 2.8.3)
project(gesture_recognition_client)

find_package(catkin REQUIRED COMPONENTS
grpc
std_msgs
message_generation)

generate_proto(openpose_rpc GRPC
               proto/openpose.proto proto/openpose_rpc.proto)

# Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
  FILES
    Gesture.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
CATKIN_DEPENDS message_runtime
)
