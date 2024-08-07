import neat
import os
import retro
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# Configuration variables
game = 'SuperMarioBros-Nes' # Game name
game_state = 'Level1-1' # game level (this will vary depending on game you are running)
number_of_generations = 500 # Total generations simulated by NEAT
number_of_skipped_frames = 4 # Amount of frames skipped until given to neural network
max_frames_for_training = 250 # Max amount of allowed for AI to not improve fitness score
show_game_on_screen = True # Toggle on or off the viewing
record = True # Records runs where a new high fitness value was achieved


# NEAT Configuration--------------------------------------------------------

# Use the current working directory
local_dir = os.getcwd()
config_path = os.path.join(local_dir, 'config-feedforward-normalized.txt')
config_path = os.path.normpath(config_path)

# Debug prints to ensure correct file path
print("Using configuration file:", config_path)

# Load the configuration file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


# Functions-----------------------------------------------------------------

# Saves frames as images so they can be turned into videos if you want to record the best runs
def save_obs_as_images(obs_frames, frame_dir='frames'):
    # Ensure the directory exists
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    # Save each frame as an image
    for idx, obs in enumerate(obs_frames):
        frame_path = os.path.join(frame_dir, f'frame_{idx:05d}.png')
        cv2.imwrite(frame_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    
    print(f'{len(obs_frames)} frames saved in {frame_dir}')

# Creates videos from saved images 
def create_videos_from_images(frame_dir='frames', video_dir='videos', fps=30):
    # Ensure the directories exist
    if not os.path.exists(frame_dir):
        print(f'Frame directory {frame_dir} does not exist.')
        return
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])

    if not frames:
        print(f'No frames found in {frame_dir}')
        return

    # Read the first frame to get the frame dimensions
    first_frame = cv2.imread(frames[0])
    frame_height, frame_width, _ = first_frame.shape

    # Generate a unique video file name using the timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_file = f'output_video_{timestamp}.avi'
    video_out_path = os.path.join(video_dir, video_file)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f'Failed to create video writer for {video_out_path}')
        return

    for frame in frames:
        img = cv2.imread(frame)
        if img is None:
            print(f'Failed to read frame {frame}')
            continue
        out.write(img)

    out.release()
    print(f'Video saved as {video_out_path}')

    # Delete frames after creating the video
    for frame in frames:
        os.remove(frame)
    print(f'All frames deleted from {frame_dir}')


# Evaluation of Genomes and main loop-----------------------------------------

# Example fitness function and main function where Neural Network is trained
# Genenomes are produced and evaluatied
def eval_genomes(genomes, config):
    # Ensures variables can be accessed inside the evaluation genomes function
    global best_fitness
    global env
    global game 
    global number_of_skipped_frames
    global show_game_on_screen

    # Allows for a fresh game after all genomes have run so memory does not leak 
    env.close()
    env = retro.make(game=game, state=game_state)

    # Repeats for number of genomes specified in config file
    for genome_id, genome in genomes:

        # First initialization of loop ----------------------------------------
        
        ob = env.reset()  # First image
        obs_frames = [] # List of frames to be turned into images
        obs_frames.append(ob) # Added to pending images for possible video 
        # First action
        random_action = env.action_space.sample()

        # Creating Neural Network based on config file
        net = neat.nn.RecurrentNetwork.create(genome, config)

        # Setting Conunters and fitness trackers
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        done = False

         # Processing environment space for neural network

        inx, iny, inc = env.observation_space.shape  # inc = color
        # Image reduction for faster processing
        inx = int(inx / 8)
        iny = int(iny / 8)
        ob = cv2.resize(ob, (inx, iny))  # Resize the current frame
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ob = np.reshape(ob, (inx, iny))       
        oned_image = np.ndarray.flatten(ob)
        neuralnet_output = net.activate(oned_image)  # Get output from neural network

        # Interacting with game

        # Retrieves observation, reward, done flag, and game information
        ob, rew, done, info = env.step(neuralnet_output)  # Apply the network output to the game
        obs_frames.append(ob)
        # Updates fitness
        fitness_current += rew

        # Loops until flag 'done' is activated to True ---------------------------------
        while not done: 
            frame += 1 # Updates frame
            # This section runs after the amount of set number_of_skipped_frames
            if frame % number_of_skipped_frames == 0: 
                # Processing environment space for neural network
                ob = cv2.resize(ob, (inx, iny))  # Resize the current frame
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                ob = np.reshape(ob, (inx, iny))
                oned_image = np.ndarray.flatten(ob)

                # Interacting with game environment
                neuralnet_output = net.activate(oned_image)  # Get output from neural network
                ob, rew, done, info = env.step(neuralnet_output)  # Apply the network output to the game
                obs_frames.append(ob) # adding to pending images
                fitness_current += rew #updating fitness

                # Updates max fitness if new record has been achieved
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1  # Count the frames until successful
                # Checks if game screen needs to render
                if show_game_on_screen:
                    # Render using Matplotlib
                    plt.imshow(env.render(mode='rgb_array'))
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.01)
                    plt.clf()
            # Runs every other frame that is not skipped
            else:
                # Uses last known nueral network output while frames are skipped
                ob, rew, done, info = env.step(neuralnet_output)  # Apply the network output to the game
                obs_frames.append(ob) # adding to pending images
                fitness_current += rew # updates fitness

                # Updates max fitness if new record has been achieved
                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1  # Count the frames until successful

                if show_game_on_screen:
                    # Render using Matplotlib
                    plt.imshow(env.render(mode='rgb_array'))
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(0.01)
                    plt.clf()

            # Final checks ------------------------------------------------------

            # Checks to see if game is done or max frames has occured
            if done or counter == max_frames_for_training:
                # If only max frames has occured we need to make sure flag 'done' is set to True
                done = True
                # Shows scores of genomes
                print(genome_id, fitness_current)
                # If a new fitness max has been achieved that value is updated and video is made if
                # 'record' is set to True
                if fitness_current > best_fitness:
                    best_fitness = fitness_current
                    if record:
                        save_obs_as_images(obs_frames)
                        create_videos_from_images()
                # Updates fitness value for the current genome
                genome.fitness = fitness_current

# Setting up NEAT populations and Game Environment --------------------------------------------


# Create the population, which is the top-level object for a NEAT run
p = neat.Population(config)
# Add a reporter to show progress in the terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Creates initial environment
env = retro.make(game=game, state=game_state)
best_fitness = 0

# Run for up to number_of_generations
winner = p.run(eval_genomes, number_of_generations)

# Display the winning genome
print('\nBest genome:\n{!s}'.format(winner))

# Close the environment
env.close()