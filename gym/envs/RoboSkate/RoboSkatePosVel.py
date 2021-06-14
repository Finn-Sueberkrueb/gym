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
        reply.boardRotation[9] /= 1  # If the board points to the left, this entry is 1.
        reply.boardRotation[11] /= 1  # In the Boll is flat on the ground this is 1 (yaw dose not change this value)

        return reply
    else:
        print("GetInfo gRPC failure")


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
def RoboSkate_thread(port=50051, graphics_environment=False):
    if graphics_environment:
        # choose Platform and run with graphics
        if platform == "darwin":
            var = os.system("../games/RoboSkate.app/Contents/MacOS/RoboSkate -p " + str(port))
        elif platform == "linux" or platform == "linux2":
            var = os.system("../games/RoboSkate/roboskate.x86_64 -p " + str(port))
        elif platform == "win32":
            var = os.system("../games/RoboSkate/RoboSkate.exe -p " + str(port))

    else:
        # choose Platform and run in batchmode
        if platform == "darwin":
            var = os.system("../games/RoboSkate.app/Contents/MacOS/RoboSkate -nographics -batchmode -p " + str(port))
        elif platform == "linux" or platform == "linux2":
            var = os.system("../games/RoboSkate/roboskate.x86_64 -nographics -batchmode  -p " + str(port))
        elif platform == "win32":
            var = os.system("../games/RoboSkate/RoboSkate.exe -nographics -batchmode  -p " + str(port))

# --------------------------------------------------------------------------------
# ------------------ RoboSkate Environment ---------------------------------------
# --------------------------------------------------------------------------------
class RoboSkatePosVel(gym.Env):

    def __init__(self, port=50051, render=False):
        super(RoboSkatePosVel, self).__init__()

        self.Port = port

        self.max_episode_length = 1000

        threading.Thread(target=RoboSkate_thread, args=(self.Port,render)).start()

        time.sleep(15)

        print("RoboSkate started with port: " + str(self.Port))

        # gRPC channel
        address = 'localhost:' + str(self.Port)

        channel = grpc.insecure_channel(address)
        self.stub = service_pb2_grpc.CommunicationServiceStub(channel)

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
                                            shape=(8,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculation of the direction error This function shall be replaced later by image data
    def calculateDirectionError(self, x_pos, y_pos, x_orientation, y_oriantation):

        # everything in degree
        if x_pos <= 72.5:
            # first part drive straight
            correct_orientation = 0
        elif x_pos > 72.5:
            # Calculate the orientation in the curve by the tangent of a circle
            correct_orientation = -math.sin((x_pos - 72.5) * (math.pi / 2) / 35.19) * 90

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

        direction_error = current_orientation - correct_orientation

        return -direction_error, current_orientation, correct_orientation


    # ------------------------------------------------------------------------------------------------------------------
    def setstartposition(self):
        for i in range(50):

            self.state = get_info(self.stub)
            joint2 = (55 - self.state.boardCraneJointAngles[1]*max_Joint_pos_2)
            joint3 = (110 - self.state.boardCraneJointAngles[2]*max_Joint_pos_3)
            set_info(self.stub, 0, joint2/20, joint3/10)
            run_game(self.stub, 0.2)

        set_info(self.stub, 0,0,0)



    def reset(self):

        initialize(self.stub, "0,10,10")

        self.setstartposition()

        self.start = time.time()
        self.stepcount = 0

        # get the current state
        self.state = get_info(self.stub)

        self.directionError, \
        self.currentOrientation, \
        self.correctOrientation = self.calculateDirectionError(self.state.boardPosition[0] * max_board_pos_XY,
                                                               self.state.boardPosition[2] * max_board_pos_XY,
                                                               self.state.boardRotation[7],
                                                               self.state.boardRotation[9])


        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardCraneJointAngles[3],
                         self.state.boardCraneJointAngles[4],
                         self.state.boardCraneJointAngles[5],
                         self.state.boardRotation[8],
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

        self.directionError, \
        self.currentOrientation, \
        self.correctOrientation = self.calculateDirectionError(self.state.boardPosition[0] * max_board_pos_XY,
                                                               self.state.boardPosition[2] * max_board_pos_XY,
                                                               self.state.boardRotation[7],
                                                               self.state.boardRotation[9])


        directionCorrection = abs(self.oldirectionError) - abs(self.directionError)
        forward_reward = (self.state.boardPosition[0] - self.oldstate.boardPosition[0]) * max_board_pos_XY + (self.state.boardPosition[2] - self.oldstate.boardPosition[2]) * max_board_pos_XY
        ctrl_cost = abs(action[0]) + abs(action[1]) + abs(action[2])
        survive_reward = 1

        self.reward = forward_reward*50 - ctrl_cost*1 + directionCorrection*20 + survive_reward


        # Termination conditions
        if self.stepcount >= self.max_episode_length:
            # Stop if max episode is reached
            #print("time is up")
            done = True
        elif (self.state.boardPosition[0]* max_board_pos_XY > 105) and (self.state.boardPosition[2]* max_board_pos_XY < 33):
            # Stop if checkpoint is reached
            #print("checkpoint1 reached")
            done = True
        elif abs(self.state.boardRotation[11]) < 0.30:
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
        info = {"time": self.stepcount,
                "clocktime": (time.time() - self.start),
                "xPos": (self.state.boardPosition[0] * max_board_pos_XY),
                "yPos": (self.state.boardPosition[2] * max_board_pos_XY),
                "direction error": self.directionError,
                "current Orientation": self.currentOrientation,
                "correct Orientation": self.correctOrientation}

        self.stepcount += 1

        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardCraneJointAngles[3],
                         self.state.boardCraneJointAngles[4],
                         self.state.boardCraneJointAngles[5],
                         self.state.boardRotation[8],
                         self.directionError]).astype(np.float32), self.reward, done, info


    def render(self, mode='human'):
        print('render is not in use!')

    def close(self):
        print('close gRPC client' + str(self.Port))
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))

