import gymnasium as gym
import pandas as pd
import numpy as np
import neat, os, pickle

def eval_genomes(genomes, config):
    # This will run each simulation and evaluate the genomes of 1 population
    

    nets = []
    ge = []
    envs = []
    obvs = []

    # Create all NNs from genomes and assign them to new Cart Pole (random inital state)
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        env = gym.make('CartPole-v1')
        ob, _ = env.reset()
        envs.append(env)
        obvs.append(ob)

    while len(envs) > 0:
        for g, nn, env, observation  in zip(ge, nets, envs, obvs):
            # First feed observation into neural net to get an output
            output = nn.activate(observation)
            # Set action based on nn output
            if output[0] > 0: 
                action = 1
            else:
                action = 0
            
            # Move environment forward one step
            observation, reward, terminated, info, _ = env.step(action)

            if terminated: # Cart failed to balance poll - remove from all lists
                ge.pop(envs.index(env))
                nets.pop(envs.index(env))
                obvs.pop(envs.index(env))
                env.close()
                envs.pop(envs.index(env)) 
                
            else:
                g.fitness += reward


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
    winner = p.run(eval_genomes, 5)

if __name__ == '__main__':
    # Find the Config File for Neat, will be in the same folder
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)

