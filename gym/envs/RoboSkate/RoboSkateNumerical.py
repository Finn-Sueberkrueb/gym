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

# This function should tell if the connection to roboskate is possible
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
            # Connection possible and positiv awnser
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
# ------------------ RoboSkate Game ----------------------------------------------
# --------------------------------------------------------------------------------
def startRoboSkate(port, graphics_environment):
    if graphics_environment:
        # choose Platform and run with graphics
        if platform == "darwin":
            var = os.system("nohup ../games/RoboSkate.app/Contents/MacOS/RoboSkate -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "linux" or platform == "linux2":
            var = os.system("nohup ../games/RoboSkate/roboskate.x86_64 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "win32":
            print("Running RoboSkate on windows in the background has not been tested yet!")
            var = os.system("nohup ../games/RoboSkate/RoboSkate.exe -p " + str(port) + " > RoboSkate" + str(port) + ".log &")

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
                 headlessMode=True,
                 AutostartRoboSkate=True,
                 startLevel=0,
                 cameraWidth=200,
                 cameraHeight=60):

        super(RoboSkateNumerical, self).__init__()


        print("RoboSkate Env start with rank: " + str(rank))
        self.startLevel = startLevel
        self.cameraWidth = cameraWidth
        self.cameraHeight = cameraHeight
        self.headlessMode = headlessMode
        self.Port = startport + rank
        self.max_episode_length = max_episode_length


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
    # Calculation of Steering Angle in Level 1 splitted in straight part and kurv.
    # After CNN implementation still used for Reward

    def calculateSteeringAngleLevel1(self, x_pos, y_pos, x_orientation, y_oriantation):

        if x_pos <= 72.5:
            # first part drive straight
            correct_orientation = 0
            correct_yPosition = 0
            position_error = y_pos

        elif (x_pos > 72.5) and (y_pos > -35.21):
            # Calculate the orientation in the curve by the tangent of a circle
            correct_orientation = -math.sin((x_pos - 72.5) * (math.pi / 2) / 35.21) * 90
            correct_radius = 35.21
            current_radius = math.sqrt( ((x_pos-72.5)**2) + ((y_pos + 35.21)**2))
            position_error = current_radius - correct_radius
        else:
            correct_orientation = -90
            position_error = x_pos - 72.5


        # Calculate rotation error
        # Normalization
        x_ori = x_orientation / abs(x_orientation + y_oriantation)
        y_ori = y_oriantation / abs(x_orientation + y_oriantation)

        if x_ori >= 0:
            current_orientation = math.atan(y_ori / x_ori) * 90
        else:
            if y_ori >= 0:
                current_orientation = 180 - math.atan(y_ori / abs(x_ori)) * 90
            else:
                current_orientation = -180 - math.atan(y_ori / abs(x_ori)) * 90

        rotation_error = (current_orientation - correct_orientation)

        direction_error = (-1) * np.clip(rotation_error + position_error * 5, -25, 25)

        return direction_error


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

        initialize(self.stub, str(self.startLevel) + "," + str(self.cameraWidth) + "," + str(self.cameraHeight))

        # Set a predefined startposition to make learning easyer
        self.setstartposition()

        self.start = time.time()
        self.stepcount = 0

        # get the current state
        self.state = get_info(self.stub)

        if not(self.headlessMode):
            # render image in Unity
            image = get_camera(self.stub, self.stepcount).transpose([2, 0, 1])
            #imageio.imwrite("./RoboSkate.png", image[0])
        else:
            image = 0

        self.directionError = self.calculateSteeringAngleLevel1(self.state.boardPosition[0] * max_board_pos_XY,
                                                               self.state.boardPosition[2] * max_board_pos_XY,
                                                               self.state.boardRotation[7],
                                                               self.state.boardRotation[9])


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
                         self.directionError]).astype(np.float32)



    def step(self, action):
        # set the actions
        # The observation will be the board state information like position, velocity and angle

        set_info(self.stub, action[0], action[1], action[2])

        # Run RoboSkate Game for time 0.2s
        run_game(self.stub, 0.2)

        self.oldstate = self.state
        self.oldirectionError = self.directionError

        # get the current observations
        self.state = get_info(self.stub)

        if not(self.headlessMode):
            # render image in Unity
            image = get_camera(self.stub, self.stepcount).transpose([2, 0, 1])
        else:
            image = 0

        self.directionError = self.calculateSteeringAngleLevel1(self.state.boardPosition[0] * max_board_pos_XY,
                                                                self.state.boardPosition[2] * max_board_pos_XY,
                                                                self.state.boardRotation[7],
                                                                self.state.boardRotation[9])


        directionCorrection = abs(self.oldirectionError) - abs(self.directionError)
        if (self.state.boardPosition[0]*max_board_pos_XY) < 72.5:
            forward_reward = (self.state.boardPosition[0] - self.oldstate.boardPosition[0]) * max_board_pos_XY
        else:
            forward_reward = (self.state.boardPosition[0] - self.oldstate.boardPosition[0]) * max_board_pos_XY + (self.oldstate.boardPosition[2] - self.state.boardPosition[2]) * max_board_pos_XY

        self.reward = forward_reward * 50 + directionCorrection * 10


        # Termination conditions
        if self.stepcount >= self.max_episode_length:
            # Stop if max episode is reached
            #print("time is up")
            done = True
        elif self.state.boardPosition[2]* max_board_pos_XY <= -45:
            # Stop if checkpoint is reached
            #print("checkpoint1 reached")
            self.reward += 200
            done = False
        elif self.state.boardPosition[1]* max_board_pos_Z <= -2:
            # Stop if checkpoint is reached
            #print("checkpoint1 reached")
            done = True
        elif abs(self.state.boardRotation[11]) < 0.40:
            # Stop if board is tipped
            #print("board is tipped")
            self.reward -= 200
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 150:
            # Stop turning the first joint
            #print("Helicopter")
            self.reward -= 200
            done = True
        else:
            done = False

        # additional information that will be shared
        info = {"step": self.stepcount,
                "xPos": (self.state.boardPosition[0] * max_board_pos_XY),
                "yPos": (self.state.boardPosition[2] * max_board_pos_XY),
                "direction error": self.directionError,
                "image": image}

        self.stepcount += 1

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
                         self.directionError]).astype(np.float32), self.reward/10.0, done, info


    def render(self, mode='human'):
        # render is not in use since Unity game.
        pass

    def close(self):
        print('close gRPC client' + str(self.Port))
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))

