#!/usr/bin/env python

from absl import flags
from hmmlearn import hmm
from sklearn.cluster import KMeans
from time import gmtime, strftime

import cv2
import grpc
import hmmlearn
import pickle
import numpy as np
import os
import signal
import sys
import rospy
import threading
import time
import use_cogrob_workspace

from cogrob.perception.gesture.hmm_gesture_recognizer import GestureRecognizer
from gesture_recognition_client.proto import openpose_rpc_pb2
from gesture_recognition_client.proto import openpose_rpc_pb2_grpc
from gesture_recognition_client.msg import Gesture

FLAGS = flags.FLAGS

flags.DEFINE_string("server_address", "192.168.111.102", "Server's IP address.")
flags.DEFINE_string("server_port", "7018", "Server's port")
flags.DEFINE_bool("dump_data", False,
                  "Whether write down keypoints and images")
flags.DEFINE_string("output_dir", "/home/users/lshixin/data/hmm_train_data",
                    "Output directory")
flags.DEFINE_float("fps", 10.0, "FPS to send to the server", lower_bound=1.001)
flags.DEFINE_string("format", "jpg", "Format to send, either jpg or png")
flags.DEFINE_string("left_label_path", "/home/users/lshixin/data/hmm_train_data/left_labels.txt",
                    "Path to left hand label file, what gestures users want to classify.")
flags.DEFINE_string("right_label_path", "/home/users/lshixin/data/hmm_train_data/right_labels.txt",
                    "Path to right hand label file, what gestures users want to classify.")
flags.DEFINE_string("left_gesture_model_path",
                    "/home/users/lshixin/data/hmm_train_data/hmm_model/left_2019-03-13_21_39_30",
                    "Path to left gesture trained HMM model.")
flags.DEFINE_string("right_gesture_model_path",
                    "/home/users/lshixin/data/hmm_train_data/hmm_model/right_2019-03-13_21_39_23",
                    "Path to right gesture trained HMM model.")

kLenOfGestureSeries = 5
kStaticGestures = ["cross", "equal"]
kSmallDis = 0.1


class WaitToTerminate:
  def __init__(self):
    self._kill_now = False
    signal.signal(signal.SIGINT, self.ExitGracefully)
    signal.signal(signal.SIGTERM, self.ExitGracefully)
    while not self._kill_now:
      signal.pause()

  def ExitGracefully(self, signum, frame):
    self._kill_now = True

class RequestIterator(object):
  def __init__(self):
    rospy.loginfo("Opening Video Capture Device")
    self._cap = cv2.VideoCapture(2)
    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # It turns out the first read will take some extra time.
    self._cap.read()
    self._cap.read()
    self._next_send_time = time.time()
    self._send_interval = 1.0 / FLAGS.fps
    rospy.loginfo("Opened Video Capture Device")


  def __iter__(self):
    return self


  def _next(self):
    # Prevents the client send too fast.
    if time.time() < self._next_send_time:
      time.sleep(self._next_send_time - time.time())
      self._next_send_time = time.time() + self._send_interval

    ret, img = self._cap.read()
    if not ret:
      rospy.logerr("Returned data is None.")
      return None
    else:
      request = openpose_rpc_pb2.Get2DKeyPointsOnImageRequest()
      img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
      if FLAGS.format == "PNG":
        img_str = cv2.imencode(".png", img)[1].tostring()
        request.image_format = openpose_rpc_pb2.PNG_PICTURE_FORMAT
      elif FLAGS.format == "JPG":
        img_str = cv2.imencode(".jpg", img)[1].tostring()
      else:
        raise NotImplementedError("Unsupport format: " + FLAGS.format)

      request.image_data = img_str
      return request


  def __next__(self):
    return self._next()


  def next(self):
    return self._next()


def AngleBetweenVectors(v1, v2):
  cos_ang = np.dot(v1, v2)
  sin_ang = np.linalg.norm(np.cross(v1, v2))
  return np.arctan2(sin_ang, cos_ang)


def ReadBodyPts(body_pts):
  # Extract the needed joints
  assert(len(body_pts) == 25)
  body_pts = [np.array([pt.x, pt.y]) for pt in body_pts[1:9]]

  # Use the middle of joint 1 and joint 8 as the center
  center = (body_pts[0] + body_pts[7]) / 2

  # Normalize according to the vector
  norm = np.linalg.norm(body_pts[0] - body_pts[7])

  # Start "shrinking" the joints
  normalized_body_pts = []
  for i in range(8):
    normalized_body_pts.append((body_pts[i] - center) / norm)
  return normalized_body_pts

def CheckStaticGesture(body_pts):
  assert(len(body_pts) == 8)
  left_elbow = [body_pts[5][0], body_pts[5][1]]
  left_wrist = [body_pts[6][0], body_pts[6][1]]
  right_elbow = [body_pts[2][0], body_pts[2][1]]
  right_wrist = [body_pts[3][0], body_pts[3][1]]
  left_arm = np.array(left_wrist) - np.array(left_elbow)
  right_arm = np.array(right_wrist) - np.array(right_elbow)

  # Check if it's a cross or an euqal body sign
  if np.abs(left_elbow[1] - right_elbow[1]) < kSmallDis and \
     np.abs(right_wrist[1] - left_wrist[1]) < kSmallDis and \
     left_elbow[0] * left_wrist[0] < 0 and \
     right_elbow[0] * right_wrist[0] < 0 and \
     AngleBetweenVectors(left_arm, right_arm) < np.pi * 0.75 and \
     AngleBetweenVectors(left_arm, right_arm) > np.pi * 0.25:
    return 0
  elif np.abs(left_elbow[0] - right_wrist[0]) < kSmallDis and \
       np.abs(left_wrist[0] - right_elbow[0]) < kSmallDis and \
       np.abs(left_elbow[1] - left_wrist[1]) < kSmallDis and \
       np.abs(right_elbow[1] - right_wrist[1]) < kSmallDis and \
       (AngleBetweenVectors(left_arm, right_arm) < np.pi * 0.15 or \
       AngleBetweenVectors(left_arm, right_arm) > np.pi * 0.85):
    return 1
  else:
    return -1


class GestureClient(object):
    def __init__(self):
      self._channel = grpc.insecure_channel(
          FLAGS.server_address + ':' + FLAGS.server_port)
      self._request_iterable = RequestIterator()
      self._stub = openpose_rpc_pb2_grpc.OpenPoseRpcStub(self._channel)
      self._responses = self._stub.Get2DKeyPointsOnStreamingImages(
        self._request_iterable)

      # Build two classifiers for left and right hand gesture recognition
      self._left_gesture_recognier = GestureRecognizer(FLAGS.left_gesture_model_path,
                                                 FLAGS.left_label_path)
      self._right_gesture_recognier = GestureRecognizer(FLAGS.right_gesture_model_path,
                                                  FLAGS.right_label_path)
      # Left and right buffer for saving time series
      self._left_x = []
      self._right_x = []
      self._client_thread = threading.Thread(target=self._Run)
      self._client_thread.daemon = True
      self._client_thread.start()

      # ROS publisher
      self._publisher = rospy.Publisher("gesture_recognition/gesture", Gesture, queue_size=10)

    def _Run(self):
      r = rospy.Rate(10) # 10hz
      rospy.loginfo("Started receiving responses from server.")

      # Start receiving streaming responses
      for response in self._responses:
        if response is None:
          break

        body_pts = response.person_2d.body_pts
        if len(body_pts) == 0:
          rospy.logerr("No body points detected yet.")
          continue
        # Get left and right hand coordinate
        normalized_body_pts = ReadBodyPts(body_pts)

        left_feature = np.array([normalized_body_pts[6][0], normalized_body_pts[6][1]], dtype=np.float32)
        right_feature = np.array([normalized_body_pts[3][0], normalized_body_pts[3][1]], dtype=np.float32)
        if np.isnan(left_feature).any() or np.isnan(left_feature).any():
          rospy.logerr("Joints of interest not detected correctly.")
          continue

        # Check whether the static gesture applied
        static_ges = CheckStaticGesture(normalized_body_pts)
        if static_ges != -1:
          rospy.loginfo("User is doing {}".format(kStaticGestures[static_ges]))
          ges_msg = Gesture()
          ges_msg.header.stamp = rospy.get_rostime()
          ges_msg.gesture_name = kStaticGestures[static_ges]
          self._publisher.publish(ges_msg)
          self._left_x = []
          self._right_x = []
          buf = np.asarray(
              bytearray(response.rendered_image_data), dtype=np.uint8)
          img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
          print("showing frame")
          cv2.imshow("frame", img)
          kMsecToWait = 10
          kEscCode = 27
          if cv2.waitKey(kMsecToWait) & 0xFF == kEscCode:
              break
          r.sleep()
          continue

        # Classify left and right hand separately
        left_ges = -1
        right_ges = -1
        left_score = -1000
        right_score = -1000
        if len(self._left_x) >= kLenOfGestureSeries:
          self._left_x = self._left_x[1:]
          self._left_x.append(left_feature)
          left_ges, left_score = self._left_gesture_recognier.Test(self._left_x)
          if left_ges > -1:
            rospy.loginfo("User left hand is doing {} with score {}".format
                         (self._left_gesture_recognier.gestures[left_ges], left_score))
            self._left_x = [] # left_x[len(left_x)/2:]
        else:
          self._left_x.append(left_feature)

        if len(self._right_x) >= kLenOfGestureSeries:
          self._right_x = self._right_x[1:]
          self._right_x.append(right_feature)
          right_ges, right_score = self._right_gesture_recognier.Test(self._right_x)
          if right_ges > -1:
            rospy.loginfo("User right hand is doing {} with score {}".format
                         (self._right_gesture_recognier.gestures[right_ges], right_score))
            self._right_x = []  # right_x[len(right_x)/2:]
        else:
          self._right_x.append(right_feature)

        # Publish gesture topic (combine the gesture from left and right hand)
        ges_msg = Gesture()
        ges_msg.header.stamp = rospy.get_rostime()
        if left_ges == -1 and right_ges == -1:
          continue
        elif left_ges >= 0 and right_ges >= 0:
          ges_msg.gesture_name = self._left_gesture_recognier.gestures[left_ges] + '_' + \
                                 self._right_gesture_recognier.gestures[right_ges]
          ges_msg.score = (left_score + right_score) / 2
        elif left_ges >= 0:
          ges_msg.gesture_name = self._left_gesture_recognier.gestures[left_ges]
          ges_msg.score = left_score
        elif right_ges >= 0:
          ges_msg.gesture_name = self._right_gesture_recognier.gestures[right_ges]
          ges_msg.score = right_score
        self._publisher.publish(ges_msg)

        # Display rendered image.
        buf = np.asarray(
          bytearray(response.rendered_image_data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        print("showing frame")
        cv2.imshow("frame", img)
        kMsecToWait = 10
        kEscCode = 27
        if cv2.waitKey(kMsecToWait) & 0xFF == kEscCode:
          break

        r.sleep()


def Run(argv):
  FLAGS(argv)

  rospy.loginfo("gesture_recognition_client_node started.")
  rospy.init_node('gesture_recognition_client_node')

  FLAGS.format = FLAGS.format.upper()
  assert FLAGS.format in ("PNG", "JPG")
  client = GestureClient()

  rospy_spin_thread = threading.Thread(target=rospy.spin)
  rospy_spin_thread.daemon = True
  rospy_spin_thread.start()

  WaitToTerminate()

if __name__ == "__main__":
  Run(sys.argv)
