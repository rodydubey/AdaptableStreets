#!/usr/bin/python
import os

#******************************
#******** Enviroment **********
#******************************

ENV_NAME = 'Adaptable_street_rl'


PATH_SAVE_MODEL = "model/{}/".format(ENV_NAME)
# PATH_LOAD_FOLDER = "model/Adaptable_street_rl/save_agent_202303201456_5448/"
PATH_LOAD_FOLDER = None

BUFFER_CAPACITY = 100000
BATCH_SIZE = 128
MIN_SIZE_BUFFER = 256

CRITIC_HIDDEN_0 = 64
CRITIC_HIDDEN_1 = 64
ACTOR_HIDDEN_0 = 64 
ACTOR_HIDDEN_1 = 64

ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
GAMMA = 0.95
TAU = 0.01

MAX_GAMES = 300
TRAINING_STEP = 5
MAX_STEPS = 2
EVALUATION_FREQUENCY = MAX_GAMES/10
SAVE_FREQUENCY = MAX_GAMES/10