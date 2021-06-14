"""
RoboSkate Gym Environment (RoboSkate Game needs to be launched separately)

This environment uses as observations only the joint angles and an angle that describes how the current "forward"
position should be corrected relative to the current position.
"""

import numpy as np
import gym
from gym import spaces
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
    if reply.success == bytes('0', encoding='utf8'):
        print("Initialize gRPC success")
        pass
    else:
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
# ------------------ RoboSkate Environment ---------------------------------------
# --------------------------------------------------------------------------------
class RoboSkateEnv(gym.Env):

    def __init__(self):
        super(RoboSkateEnv, self).__init__()

        self.ID = 50051

        print("Port of RoboSkate environment: " + str(self.ID))
        self.use_camera = False
        self.imageHeight = 10
        self.imageWidth = 10

        self.max_episode_length = 1000

        # gRPC channel
        Port = str(self.ID)
        address = 'localhost:' + Port

        channel = grpc.insecure_channel(address)
        self.stub = service_pb2_grpc.CommunicationServiceStub(channel)

        # state from the game: position, velocity, angle
        self.state = 0
        self.reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        # discrete actions: joint1, joint2, joint3
        # The first array are the lowest accepted values, the second are the highest accepted values.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float)

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
    def reset(self):
        print('reset RoboSkate environment ' + str(self.ID))

        initialize(self.stub, "0,10,10")
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

        self.reward = self.get_reward(self.state, self.directionError)

        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardRotation[8],
                         self.directionError]).astype(np.float)

    # Reward Funktion
    def get_reward(self, state, directionError):
        # variables used for reward
        basehight = 0.3
        arm1lengt = 1.1
        arm2lengt = 0.9

        checkpoint1_X = 107.69
        checkpoint1_Y = -35.21

        reward = 0

        # Reward the correct oriantation to trevel
        reward -= abs(directionError)*10

        # calculate the hight off the ball over the board
        ballhight = (math.cos(((state.boardCraneJointAngles[2] * max_Joint_pos_3 + state.boardCraneJointAngles[1] * max_Joint_pos_2) / 180.0) * math.pi) * arm2lengt + math.cos((state.boardCraneJointAngles[1] * max_Joint_pos_2 / 180) * math.pi) * arm1lengt)

        # use the hight off the ball and calculate reward with curve that uses a flat ground and is slightly shiftes ubwards
        reward -= ((ballhight+basehight) ** 2)

        # reward the speed
        reward += state.boardPosition[3] * max_board_vel_XY * 10  # X Velocity

        # reward the position towards the checkpoint
        reward += checkpoint1_X - state.boardPosition[0]*max_board_pos_XY  # X Deviation from checkpoint
        reward += checkpoint1_Y - state.boardPosition[2]*max_board_pos_XY  # Y Deviation from checkpoint
        return reward

    def step(self, action):
        # set the actions
        # The observation will be the board state information like position, velocity and angle

        set_info(self.stub, action[0], action[1], action[2])

        # Run RoboSkate Game for time 0.2s
        run_game(self.stub, 0.2)

        # get the current observations
        self.state = get_info(self.stub)

        self.directionError, \
        self.currentOrientation, \
        self.correctOrientation = self.calculateDirectionError(self.state.boardPosition[0] * max_board_pos_XY,
                                                               self.state.boardPosition[2] * max_board_pos_XY,
                                                               self.state.boardRotation[7],
                                                               self.state.boardRotation[9])

        # calculate current Reward
        self.reward = self.get_reward(self.state, self.directionError)


        # Termination conditions
        if self.stepcount >= self.max_episode_length:
            # Stop if max episode is reached
            print("time is up")
            done = True
        elif (self.state.boardPosition[0]* max_board_pos_XY > 105) and (self.state.boardPosition[2]* max_board_pos_XY < 33):
            # Stop if checkpoint is reached
            print("checkpoint1 reached")
            self.reward += 2000
            done = True
        elif abs(self.state.boardRotation[11]) < 0.30:
            # Stop if board is tipped
            print("board is tipped")
            self.reward -= 2000
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 150:
            # Stop turning the first joint
            print("Helicopter")
            self.reward -= 1000
            done = True
        else:
            done = False
        '''
        elif abs(self.directionError) > 50:
            # Stop if directionError is to high
            print("directionError to high")
            self.reward -= 2000
            done = True
        '''

        # additional information that will be shared
        info = {"time": (time.time() - self.start),
                "xPos": (self.state.boardPosition[0] * max_board_pos_XY),
                "yPos": (self.state.boardPosition[2] * max_board_pos_XY),
                "direction error": self.directionError,
                "current Orientation": self.currentOrientation,
                "correct Orientation": self.correctOrientation}

        self.stepcount += 1

        return np.array([self.state.boardCraneJointAngles[0],
                         self.state.boardCraneJointAngles[1],
                         self.state.boardCraneJointAngles[2],
                         self.state.boardRotation[8],
                         self.directionError]).astype(np.float), self.reward, done, info


    def render(self, mode='human'):
        print('render is not in use!')

    def close(self):
        print('close gRPC client' + str(self.ID))
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))

