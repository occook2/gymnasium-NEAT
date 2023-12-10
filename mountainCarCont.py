import gymnasium as gym
import pandas as pd
import numpy as np
import neat, os, pickle

def eval_genomes(genomes, config):
    # Takes in set of genomes. For each genome, run the entire test then return its fitness
    
    # First set the environment
    env = gym.make('MountainCar-v0')

    # Test setup and execution for each genome
    for genome_id, genome in genomes:
        # Test setup
        fitness = 0 # Set/reset fitness
        net = neat.nn.FeedForwardNetwork.create(genome, config) # Create NN from genome
        observation, _ = env.reset()
        max_right = observation[0]*50

        # Test Execution
        while 1:
            output = net.activate(observation) # Let NN determine next action
            if output[0] < -0.2: # Convert NN output to environment action
                action = 0
            elif output[0] > 0.2:
                action = 2
            else:
                action = 1
             
            # Move environment forward one step based on NN action
            observation, reward, terminated, truncated, _ = env.step(action)
            if observation[0] > max_right:
                max_right = observation[0]*50

            fitness += reward # Add reward since one step was taken

            if truncated or terminated: # Cart to   - Reset environment for next genome and end test
                env.reset() # Fail-safe
                break
        
        genome.fitness = fitness + max_right # Set fitness to genome fitness 


def run(config_file):  
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, \
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, \
                                    config_file)
    # Create a Population, Top Level object for NEAT
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for number of generations.
    winner = p.run(eval_genomes, 100)

if __name__ == '__main__':
    # Find the Config File for Neat, will be in the same folder
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'mountainCarConfig.txt')
    run(config_path)
