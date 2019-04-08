cmake_minimum_required(VERSION 2.8.3)
project(gesture_recognition_client)
find_package(catkin REQUIRED COMPONENTS grpc std_msgs)
catkin_package()
generate_proto(openpose_rpc GRPC
               proto/openpose.proto proto/openpose_rpc.proto)
