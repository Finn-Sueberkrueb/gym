"""
RoboSkate Gym Environment (RoboSkate Game needs to be launched separately)

This environment uses as observations only the joint angles and an angle that describes how the current "forward"
position should be corrected relative to the current position.
"""

import numpy as np
import gym
from gym import spaces
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
import torch
from torch import nn
import cv2


# Value Range for observations abs(-Min) = Max
max_Joint_force = 250.0

max_Joint_pos_1 = 185
max_Joint_pos_2 = 95.0
max_Joint_pos_3 = 130.0

max_Joint_vel = 170.0

max_board_pos_XY = 220.0
max_board_pos_Z = 50.0
max_board_vel_XY = 8.0
max_board_vel_Z = 8.0


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


        # use coordinate frame as defined in RoboSkate_API.md
        # Sorry for all the different coordinate systems system
        # Calculate ortogonal vector to the side
        forward_vec = np.array([reply.boardRotation[7],reply.boardRotation[8],reply.boardRotation[9]])
        up_vec = np.array([reply.boardRotation[10],reply.boardRotation[11],reply.boardRotation[12]])
        left_vec = np.cross(forward_vec,up_vec)

        # euler angles
        board_yaw = np.arctan2(reply.boardRotation[9], reply.boardRotation[7]) / math.pi
        board_pitch = np.arctan2(np.linalg.norm(np.array([reply.boardRotation[7],reply.boardRotation[9]])), reply.boardRotation[8]) / math.pi
        board_roll = np.arctan2(np.linalg.norm(np.array([left_vec[0], left_vec[2]])), left_vec[1]) / math.pi


        # forward Velocity. Not really accurate since only the direction +- is used for the velocity. Sidways would also be a valid velocity here.
        velocity_vec = np.array([reply.boardPosition[3], reply.boardPosition[5]])
        yaw_vec = np.array([reply.boardRotation[7], reply.boardRotation[9]])
        board_forward_velocity = np.linalg.norm(np.array([reply.boardPosition[3], reply.boardPosition[5]])) * np.sign(np.dot(yaw_vec,velocity_vec))

        return reply, board_yaw, board_roll, board_pitch, board_forward_velocity
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
            os.system("nohup ../games/RoboSkate3.app/Contents/MacOS/RoboSkate -screen-height 200 -screen-width 300 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "linux" or platform == "linux2":
            os.system("nohup ../games/RoboSkate3/roboskate.x86_64 -screen-height 200 -screen-width 300 -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "win32":
            # TODO: Running RoboSkate on windows in the background has not been tested yet!
            pass

    else:
        # choose Platform and run in batchmode
        if platform == "darwin":
            os.system("nohup ../games/RoboSkate3.app/Contents/MacOS/RoboSkate -nographics -batchmode -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "linux" or platform == "linux2":
            os.system("nohup ../games/RoboSkate3/roboskate.x86_64 -nographics -batchmode  -p " + str(port) + " > RoboSkate" + str(port) + ".log &")
        elif platform == "win32":
            # TODO: Running RoboSkate on windows in the background has not been tested yet!
            pass


# --------------------------------------------------------------------------------
# ------------------ Model -------------------------------------------------------
# --------------------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 10)

# Define model
class SegmentationNetwork(nn.Module):
    def __init__(self, image_channels=3, h_dim=2560, z_dim=8):
        super(SegmentationNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(8,6), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def forward(self, x):
        #print(x.shape) #(14x 3x 200x60)
        h = self.encoder(x)
        #print(h.shape) (14x 2560)
        output, _, _ = self.bottleneck(h)
        return output

    def decode_from_z(self, z):
        #print(z)
        z = self.fc3(z)
        #print(z.shape) #(14x 2560)
        output = self.decoder(z)
        #print(output.shape)  # (14x 3x 200x 60)
        return output



# --------------------------------------------------------------------------------
# ------------------ RoboSkate Environment ---------------------------------------
# --------------------------------------------------------------------------------
class RoboSkateSegmentation(gym.Env):

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
                 max_episode_length=2000,
                 startport=50051,
                 rank=-1,
                 small_checkpoint_radius=False,
                 headlessMode=False,
                 AutostartRoboSkate=True,
                 startLevel=0,
                 random_start_level=False,
                 cameraWidth=200,
                 cameraHeight=60,
                 show_image_reconstruction=False):

        super(RoboSkateSegmentation, self).__init__()


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
        self.show_image_reconstruction = show_image_reconstruction


        # x position, y position, checkpoint radius
        self.checkpoints = np.array([[    30,      0, 5.0], # U      # 0 Start Level 0
                                     [    55,      0, 5.0], # V
                                     [  72.5,      0, 5.0], # C_0
                                     [    87,     -3, 5.0], # S
                                     [  98.5,    -12, 5.0], # W
                                     [   105,  -22.4, 5.0], # T
                                     [107.69, -35.21, 5.0], # C_2
                                     [107.69,    -50, 2.0], # A     # 7 Start Level 1
                                     [107.69,    -57, 1.5], # C
                                     [107.69,    -65, 1.5], # D
                                     [107.69,    -73, 2.0], # E
                                     [   101,  -77.4, 1.5], # G
                                     [    95,  -77.4, 1.5], # F
                                     [    89,  -77.4, 1.5], # I
                                     [  82.2,  -77.4, 1.5], # C_4
                                     [    80,    -75, 1.5], # J
                                     [    80,    -70, 1.5], # K
                                     [    80,    -65, 1.5], # C_5
                                     [    80,    -58, 1.5], # L
                                     [    80,    -50, 3.0], # C_6     # 19 Start Level 2
                                     [    79,    -44, 3.0], # M
                                     [  76.5,    -40, 3.0], # N
                                     [    71,    -39, 3.0], # C_7
                                     [    67,  -40.6, 3.0], # O
                                     [    64,    -45, 3.0], # C_8
                                     [  62.8,  -50.4, 3.0], # P
                                     [  59.8,  -54.7, 3.0], # C_9
                                     [    54,    -56, 3.0], # Q
                                     [  49.5,    -53, 3.0], # C_10
                                     [  47.9,    -47, 3.0], # R
                                     [  47.5,    -40, 2], # C_11
                                     [  47.5,    -30, 1]]) # C_12

        if small_checkpoint_radius:
            # set all radius to 1
            self.checkpoints[:,2] = 1


        self.start_checkpoint_for_level = {0: 0,
                                           1: 7,
                                           2: 19}

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

        # Load CNN Model
        self.model = SegmentationNetwork()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.load_state_dict(
            torch.load("../trained_models/Segmentation/model/model_z_dim_8_early_stop.pth", map_location=torch.device(device)))
        self.model.eval()

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
                                            shape=(9+8,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # Calculation of Steering Angle based on checkpoints

    def checkpoint_follower(self, x_pos, y_pos, board_yaw):
        checkpoint_reached = False

        # get current position and orientation
        position = np.array([x_pos, y_pos])

        # calculate distance to next checkpoint
        distance_to_next_checkpoint = self.checkpoints[self.next_checkpoint][:2] - position

        if np.linalg.norm([distance_to_next_checkpoint]) <= self.checkpoints[self.next_checkpoint][2]:
            # closer to checkpoint than checkpoint radius
            self.next_checkpoint += 1
            checkpoint_reached = True
            # re calculate distance to next checkpoint since new checkpoint
            distance_to_next_checkpoint = self.checkpoints[self.next_checkpoint][:2] - position

        # calculate angle towards next checkpoint
        direction_to_next_checkpoint = np.arctan2(distance_to_next_checkpoint[1],
                                                  distance_to_next_checkpoint[0]) * 180.0 / math.pi

        # Board orientation / YAW in [-1,1]
        current_orientation = board_yaw * 180.0

        # Calculate rotation error
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

            self.state, _, _, _, _ = get_info(self.stub)
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
        self.state, board_yaw, board_roll, board_pitch, board_forward_velocity = get_info(self.stub)

        if not(self.headlessMode):
            # render image in Unity
            image = get_camera(self.stub, self.stepcount).transpose([2, 0, 1])
            image = image / 255.0

            with torch.no_grad():
                cnn_latent_space = self.model(torch.from_numpy(np.array([image])).float())
        else:
            image = 0

        distance_to_next_checkpoint, self.steering_angle, _ = self.checkpoint_follower(self.state.boardPosition[0] * max_board_pos_XY,
                                                                                       self.state.boardPosition[2] * max_board_pos_XY,
                                                                                       board_yaw)

        self.old_steering_angle = self.steering_angle
        self.old_distance_to_next_checkpoint = distance_to_next_checkpoint

        numerical_observations = np.array([ self.state.boardCraneJointAngles[0],
                                            self.state.boardCraneJointAngles[1],
                                            self.state.boardCraneJointAngles[2],
                                            self.state.boardCraneJointAngles[3],
                                            self.state.boardCraneJointAngles[4],
                                            self.state.boardCraneJointAngles[5],
                                            board_forward_velocity,
                                            board_pitch,
                                            board_roll]).astype(np.float32)

        # TODO: how to avoid the tensor transformations?
        np_cnn_latent_space = cnn_latent_space.cpu().detach().numpy()[0]
        return np.concatenate((numerical_observations, np_cnn_latent_space), axis=0)



    def step(self, action):
        # set the actions
        # The observation will be the board state information like position, velocity and angle
        set_info(self.stub, action[0], action[1], action[2])

        # Run RoboSkate Game for time 0.2s
        run_game(self.stub, 0.2)

        # get the current observations
        self.state, board_yaw, board_roll, board_pitch, board_forward_velocity = get_info(self.stub)

        if not(self.headlessMode):
            # render image in Unity
            image = get_camera(self.stub, self.stepcount).transpose([2, 0, 1])
            # imageio.imwrite("./RoboSkateR.png", image[0])
            image = image / 255.0

            with torch.no_grad():
                cnn_latent_space = self.model(torch.from_numpy(np.array([image])).float())
        else:
            image = 0

        if self.show_image_reconstruction:
            state = cnn_latent_space.detach().numpy()[0]
            reconstructed_image = self.model.decode_from_z(torch.tensor([state])).detach().numpy()[0].transpose([1, 2, 0])
            cv2.imshow("reconstructed image", reconstructed_image)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                pass



        distance_to_next_checkpoint, \
        self.steering_angle, \
        checkpoint_reached = self.checkpoint_follower(self.state.boardPosition[0] * max_board_pos_XY,
                                                      self.state.boardPosition[2] * max_board_pos_XY,
                                                      board_yaw)

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
        if self.next_checkpoint >= (self.checkpoints.shape[0]-1):
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
        elif abs(board_roll-0.5) > 0.35:
            # Stop if board is tipped
            self.reward -= 10
            print("board tipped")
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 200:
            # Stop if turning the first joint to fast "Helicopter"
            self.reward -= 10
            print("Helicopter")
            done = True

        # additional information that will be shared
        info = {"step": self.stepcount,
                "board_pitch": board_pitch,
                "steering_angle": self.steering_angle}

        self.stepcount += 1

        numerical_observations = np.array([self.state.boardCraneJointAngles[0],
                                           self.state.boardCraneJointAngles[1],
                                           self.state.boardCraneJointAngles[2],
                                           self.state.boardCraneJointAngles[3],
                                           self.state.boardCraneJointAngles[4],
                                           self.state.boardCraneJointAngles[5],
                                           board_forward_velocity,
                                           board_pitch,
                                           board_roll]).astype(np.float32)

        # TODO: how to avoid the tensor transformations?
        np_cnn_latent_space = cnn_latent_space.cpu().detach().numpy()[0]
        return np.concatenate((numerical_observations, np_cnn_latent_space), axis=0), self.reward, done, info

    def render(self, mode='human'):
        # render is not in use since Unity game.
        pass

    def close(self):
        print('close gRPC client' + str(self.Port))
        shutdown(self.stub)
        self.end = time.time()
        print("Time for the epoch: " + str(self.end - self.start))

