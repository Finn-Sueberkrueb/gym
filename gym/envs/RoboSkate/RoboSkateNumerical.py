"""
RoboSkate Gym Environment (RoboSkate Game needs to be launched separately)

This environment uses as observations only the joint angles and an angle that describes how the current "forward"
position should be corrected relative to the current position.
"""

import numpy as np
import gym
from gym import spaces
import threading
import os
from sys import platform
import math
import grpc
import time
from gym.envs.RoboSkate.grpcClient import service_pb2_grpc
from gym.envs.RoboSkate.grpcClient.service_pb2 import InitializeRequest, NoParams, RunGameRequest, SetInfoRequest
import io
from PIL import Image
import socket
import imageio
import torch
from torch import nn
from torchvision.io import read_image


# Value Range for observations abs(-Min) = Max
max_Joint_force = 300.0

max_Joint_pos_1 = 180
max_Joint_pos_2 = 90.0
max_Joint_pos_3 = 125.0

max_Joint_vel = 170.0

max_board_pos_XY = 220.0
max_board_pos_Z = 50.0
max_board_vel_XY = 10.0
max_board_vel_Z = 10.0


# --------------------------------------------------------------------------------
# ------------------ gRPC functions ----------------------------------------------
# --------------------------------------------------------------------------------
def initialize(stub, string):
    reply = stub.initialize(InitializeRequest(json=string))
    if reply.success != bytes('0', encoding='utf8'):
        print("Initialize failure")

# This function should tell if the connection to RoboSkate is possible
def isRunning(stub):
    try:
        reply = stub.initialize(InitializeRequest(json="0,10,10"))
    except:
        # No grpc channel yet.
        return False
    else:
        if reply.success != bytes('0', encoding='utf8'):
            # Connection possible but negativ reply
            print("Something went wrong with RoboSkate.")
            return False
        else:
            # Connection possible and positive response
            return True


def set_info(stub, joint1, joint2, joint3):
    # passing value to the RoboSkate Game
    reply = stub.set_info(SetInfoRequest(boardCraneJointAngles=[joint1 * max_Joint_force,
                                                                joint2 * max_Joint_force,
                                                                joint3 * max_Joint_force]))
    if reply.success != bytes('0', encoding='utf8'):
        print("SetInfo gRPC failure")


def get_info(stub):
    # get current observations from RoboSkate Game
    reply = stub.get_info(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        # normalisation
        reply.boardCraneJointAngles[0] /= max_Joint_pos_1
        reply.boardCraneJointAngles[1] /= max_Joint_pos_2
        reply.boardCraneJointAngles[2] /= max_Joint_pos_3
        reply.boardCraneJointAngles[3] /= max_Joint_vel
        reply.boardCraneJointAngles[4] /= max_Joint_vel
        reply.boardCraneJointAngles[5] /= max_Joint_vel

        reply.boardPosition[0] /= max_board_pos_XY
        reply.boardPosition[1] /= max_board_pos_Z
        reply.boardPosition[2] /= max_board_pos_XY
        reply.boardPosition[3] /= max_board_vel_XY
        reply.boardPosition[4] /= max_board_vel_Z
        reply.boardPosition[5] /= max_board_vel_XY

        reply.boardRotation[7] /= 1  # If the board is pointing straight forward, this entry is 1.
        reply.boardRotation[8] /= 1
        reply.boardRotation[9] /= 1  # If the board points to the left, this entry is 1.
        reply.boardRotation[10] /= 1
        reply.boardRotation[11] /= 1  # In the Boll is flat on the ground this is 1 (yaw dose not change this value)
        reply.boardRotation[12] /= 1

        return reply
    else:
        print("GetInfo gRPC failure")

def get_camera(stub, i):
    reply = stub.get_camera(NoParams())

    image = reply.imageData
    stream = io.BytesIO(image)
    img = Image.open(stream)

    return np.asarray(img)

def run_game(stub, simTime):
    # Run the game for one time step (duration of simTime)
    reply = stub.run_game(RunGameRequest(time=simTime))
    if reply.success != bytes('0', encoding='utf8'):
        print("RunGame gRPC failure")


def shutdown(stub):
    reply = stub.shutdown(NoParams())
    if reply.success == bytes('0', encoding='utf8'):
        print("Shutdown gRPC success")
    else:
        print("Shutdown gRPC failure")



# --------------------------------------------------------------------------------
# ------------------ Start RoboSkate Game ----------------------------------------
# --------------------------------------------------------------------------------
def startRoboSkate(port, graphics_environment):
    if graphics_environment:
        # choose Platform and run with graphics
        if platform == "darwin":
            var = os.system("nohup ../games/RoboSkate.app/Contents/MacOS/RoboSkate -screen-height 200 -screen-width 300 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "linux" or platform == "linux2":
            var = os.system("nohup ../games/RoboSkate/roboskate.x86_64 -screen-height 200 -screen-width 300 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "win32":
            print("Running RoboSkate on windows in the background has not been tested yet!")
            var = os.system("nohup ../games/RoboSkate/RoboSkate.exe -screen-height 200 -screen-width 300 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")

    else:
        # choose Platform and run in batchmode
        if platform == "darwin":
            var = os.system("nohup ../games/RoboSkate.app/Contents/MacOS/RoboSkate -nographics -batchmode -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "linux" or platform == "linux2":
            var = os.system("nohup ../games/RoboSkate/roboskate.x86_64 -nographics -batchmode  -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "win32":
            print("Running RoboSkate on windows in the background has not been tested yet!")
            var = os.system("nohup ../games/RoboSkate/RoboSkate.exe -nographics -batchmode  -p " + str(port) + " > RoboSkate" + str(port) + ".log &")



# --------------------------------------------------------------------------------
# ------------------ RoboSkate Environment ---------------------------------------
# --------------------------------------------------------------------------------
class RoboSkateNumerical(gym.Env):

    def is_port_open(self, host, port):
        # determine whether `host` has the `port` open
        # creates a new socket
        s = socket.socket()
        try:
            # tries to connect to host using that port
            s.connect((host, port))
        except:
            # cannot connect, port is closed
            return False
        else:
            # the connection was established, port is open!
            return True

    def __init__(self,
                 max_episode_length=1000,
                 startport=50051,
                 rank=-1,
                 small_checkpoint_radius=True,
                 headlessMode=True,
                 AutostartRoboSkate=True,
                 startLevel=0,
                 random_start_level=False,
                 cameraWidth=200,
                 cameraHeight=60):

        super(RoboSkateNumerical, self).__init__()


        print("RoboSkate Env start with rank: " + str(rank))
        self.max_episode_length = max_episode_length
        self.Port = startport + rank
        self.headlessMode = headlessMode
        self.startLevel = startLevel
        self.random_start_level = random_start_level
        self.cameraWidth = cameraWidth
        self.cameraHeight = cameraHeight
        self.old_steering_angle = 0
        self.old_distance_to_next_checkpoint = 0


        # x position, y position, checkpoint radius
        self.checkpoints = np.array([[  30,   0, 5], # 0 - Level 0
                                     [  55,   0, 5],
                                     [  72,   0, 5],
                                     [  97, -10, 5],
                                     [ 108, -35, 5],
                                     [ 108, -77, 4],  # 5 - Level 1
                                     [80.5, -76, 3],
                                     [80.5, -65, 3],
                                     [  80, -48, 3],  # 8 - Level 2
                                     [  72, -38, 3],
                                     [  64, -45, 3],
                                     [  60, -55, 3],
                                     [  49, -53, 3],
                                     [47.5, -40, 3],
                                     [47.5, -30, 3]])

        if small_checkpoint_radius:
            # set all radius to 1
            self.checkpoints[:,2] = 1


        self.start_checkpoint_for_level = {0: 0,
                                           1: 5,
                                           2: 8}

        # gRPC channel
        address = 'localhost:' + str(self.Port)
        channel = grpc.insecure_channel(address)
        self.stub = service_pb2_grpc.CommunicationServiceStub(channel)

        # Check if the port ist alreay open
        if not(self.is_port_open('localhost', self.Port)):
            if AutostartRoboSkate:
                startRoboSkate(self.Port, not(headlessMode))

                print("Wait until RoboSkate is started with port: " + str(self.Port))
                while(not isRunning(self.stub)):
                    time.sleep(2)

                print("RoboSkate started with port: " + str(self.Port))
            else:
                print("RoboSkate needs to be started manual before.")
        else:
            print("RoboSkate with port " + str(self.Port) + " already running or port is used from different app.")



        # state from the game: position, velocity, angle
        self.state = 0
        self.reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # discrete actions: joint1, joint2, joint3
        # The first array are the lowest accepted values, the second are the highest accepted values.
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(3,),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(15,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculation of Steering Angle based on checkpoints

    def checkpoint_follower(self, x_pos, y_pos, x_orientation, y_oriantation):
        checkpoint_reached = False

        # get current position and orientation
        position = np.array([x_pos, y_pos])
        orientation = np.array([x_orientation, y_oriantation])
        # Normalize orientation
        orientation = orientation / np.linalg.norm(orientation)

        # calculate distance to next checkpoint
        distance_to_next_checkpoint = self.checkpoints[self.next_checkpoint][:2] - position

        if (np.linalg.norm([distance_to_next_checkpoint]) <= self.checkpoints[self.next_checkpoint][2]):
            # closer to checkpoint than checkpoint radius
            self.next_checkpoint += 1
            checkpoint_reached = True
            # re calculate distance to next checkpoint since new checkpoint
            distance_to_next_checkpoint = self.checkpoints[self.next_checkpoint][:2] - position

        # calculate angle towards next checkpoint
        direction_to_next_checkpoint = np.arctan2(distance_to_next_checkpoint[1],
                                                  distance_to_next_checkpoint[0]) * 180 / math.pi

        # Calculate rotation error
        current_orientation = np.arctan2(orientation[1], orientation[0]) * 180 / math.pi

        if abs(current_orientation - direction_to_next_checkpoint) > 180:
            # case where we go over the +-180°
            rotation_error = -(360 - abs(current_orientation - direction_to_next_checkpoint)) * np.sign(
                current_orientation - direction_to_next_checkpoint)
        else:
            rotation_error = (current_orientation - direction_to_next_checkpoint)

        return np.linalg.norm([distance_to_next_checkpoint]), -rotation_error, checkpoint_reached


    # ------------------------------------------------------------------------------------------------------------------
    # Set the Robot Arm to a low starting possition to get an easyer start
    def setstartposition(self):
        for i in range(5):

            self.state = get_info(self.stub)
            joint2 = (55 - self.state.boardCraneJointAngles[1]*max_Joint_pos_2)
            joint3 = (110 - self.state.boardCraneJointAngles[2]*max_Joint_pos_3)
            set_info(self.stub, 0, joint2/20, joint3/10)
            run_game(self.stub, 0.2)

        set_info(self.stub, 0,0,0)



    def reset(self):

        self.rewardsum = 0

        # set start level
        if self.random_start_level:
            self.startLevel = np.random.randint(3)

        # set corresponding checkpoint for startLevel
        self.next_checkpoint = self.start_checkpoint_for_level[self.startLevel]

        # Reset environoment
        initialize(self.stub, str(self.startLevel) + "," + str(self.cameraWidth) + "," + str(self.cameraHeight))

        # set a predefined starting position to assist learning
        self.setstartposition()

        self.start = time.time()
        self.stepcount = 0

        # get the current state
        self.state = get_info(self.stub)

        distance_to_next_checkpoint, self.steering_angle, _ = self.checkpoint_follower(self.state.boardPosition[0] * max_board_pos_XY,
                                                             self.state.boardPosition[2] * max_board_pos_XY,
                                                             self.state.boardRotation[7],
                                                             self.state.boardRotation[9])

        self.old_steering_angle = self.steering_angle
        self.old_distance_to_next_checkpoint = distance_to_next_checkpoint

        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardCraneJointAngles[3],
                         self.state.boardCraneJointAngles[4],
                         self.state.boardCraneJointAngles[5],
                         self.state.boardPosition[3],
                         self.state.boardPosition[5],
                         self.state.boardRotation[7],
                         self.state.boardRotation[8],
                         self.state.boardRotation[9],
                         self.state.boardRotation[10],
                         self.state.boardRotation[11],
                         self.state.boardRotation[12],
                         self.steering_angle]).astype(np.float32)



    def step(self, action):
        # set the actions
        # The observation will be the board state information like position, velocity and angle
        set_info(self.stub, action[0], action[1], action[2])

        # Run RoboSkate Game for time 0.2s
        run_game(self.stub, 0.2)

        # get the current observations
        self.state = get_info(self.stub)

        if not(self.headlessMode):
            # render image in Unity
            image = get_camera(self.stub, self.stepcount).transpose([2, 0, 1])
        else:
            image = 0



        distance_to_next_checkpoint, \
        self.steering_angle, \
        checkpoint_reached = self.checkpoint_follower(self.state.boardPosition[0] * max_board_pos_XY,
                                                      self.state.boardPosition[2] * max_board_pos_XY,
                                                      self.state.boardRotation[7],
                                                      self.state.boardRotation[9])

        if checkpoint_reached:
            # Do not use distance to next checkpoint at checkpoint since it jumps to next checkpoints distance
            self.reward = 3
        else:
            driving_reward = self.old_distance_to_next_checkpoint - distance_to_next_checkpoint
            steering_reward = abs(self.old_steering_angle) - abs(self.steering_angle)
            self.reward = driving_reward*5 + steering_reward*1


        self.old_steering_angle = self.steering_angle
        self.old_distance_to_next_checkpoint = distance_to_next_checkpoint

        done = False
        # Termination conditions
        if self.next_checkpoint >= 5: # use only level 1 # (self.checkpoints.shape[0]-1):
            # final end reached, last checkpoint is outside the path
            done = True
            print("final end reached")
        elif self.stepcount >= self.max_episode_length:
            # Stop if max episode is reached
            done = True
            print("episode end at checkpoint: " + str(self.next_checkpoint))
        elif self.state.boardPosition[1] * max_board_pos_Z <= -7:
            # Stop if fallen from path
            self.reward -= 15
            print("fallen from path")
            done = True
        elif abs(self.state.boardRotation[11]) < 0.30:
            # Stop if board is tipped
            self.reward -= 10
            print("board tipped")
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 150:
            # Stop if turning the first joint to fast "Helicopter"
            self.reward -= 10
            print("Helicopter")
            done = True


        # additional information that will be shared
        info = {"step": self.stepcount,
                "xPos": (self.state.boardPosition[0] * max_board_pos_XY),
                "yPos": (self.state.boardPosition[2] * max_board_pos_XY),
                "image": image}

        self.stepcount += 1

        # Output reward in Excel copy and paste appropriate format.
        # self.rewardsum += self.reward
        # print(("%3.2f\t %3.2f" % (self.rewardsum, self.reward)).replace(".",","))

        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardCraneJointAngles[3],
                         self.state.boardCraneJointAngles[4],
                         self.state.boardCraneJointAngles[5],
                         self.state.boardPosition[3],
                         self.state.boardPosition[5],
                         self.state.boardRotation[7],
                         self.state.boardRotation[8],
                         self.state.boardRotation[9],
                         self.state.boardRotation[10],
                         self.state.boardRotation[11],
                         self.state.boardRotation[12],
                         self.steering_angle]).astype(np.float32), self.reward, done, info


    def render(self, mode='human'):
        # render is not in use since Unity game.
        pass

    def close(self):
        print('close gRPC client' + str(self.Port))
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))

