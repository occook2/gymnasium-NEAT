import gymnasium as gym
import pandas as pd
import numpy as np
import neat, os, pickle

global gen_count
gen_count = 0
global df_generation 
df_generation = pd.DataFrame(columns=['Generation #', 'Species #', 'Input Move #', 'Input Move', 'Output 1', 'Output 2', 'Output 3', 'Output 4', 'New Fitness', 'Dead?'])

def eval_genomes(genomes, config):
    # This will run each simulation and evaluate the genomes of 1 population
    global gen_count
    global df_generation
    gen_count += 1
    
    nets = []
    ge = []
    envs = []
    obvs = []
    species_num = []
    species_count = 0

    # Create all NNs from genomes and assign them to new Cart Pole (random inital state)
    for genome_id, genome in genomes:
        species_count += 1
        species_num.append(species_count)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        env = gym.make('CartPole-v1')
        ob, _ = env.reset()
        envs.append(env)
        obvs.append(ob)
    
    move_num = 0

    while len(envs) > 0:
        move_num += 1
        for g, nn, env, observation, s  in zip(ge, nets, envs, obvs, species_num):
            # First feed observation into neural net to get an output
            output = nn.activate(observation)
            # Set action based on nn output
            if output[0] > 0: 
                action = 1
            else:
                action = 0
            
            # Move environment forward one step
            observation, reward, terminated, info, _ = env.step(action)
            inf = {'Generation #' : [gen_count], 'Input Move #' : [move_num], 'Input Move': [action], \
                   'Output 1':[observation[0]], 'Output 2':[observation[1]], 'Output 3':[observation[2]], \
                    'Output 4':[observation[3]], 'Species #': s}

            if terminated: # Cart failed to balance poll - remove from all lists
                inf['Dead?'] = [True]
                inf['New Fitness'] = [g.fitness]

                ge.pop(envs.index(env))
                nets.pop(envs.index(env))
                obvs.pop(envs.index(env))
                species_num.pop(envs.index(env))
                env.close()
                envs.pop(envs.index(env)) 
                
            else:
                g.fitness += reward
                inf['Dead?'] = [False]
                inf['New Fitness'] = [g.fitness]
            df_move = pd.DataFrame.from_dict(inf)
            df_generation = pd.concat([df_generation, df_move], ignore_index=True)


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

    df_generation.to_excel("output.xlsx")

