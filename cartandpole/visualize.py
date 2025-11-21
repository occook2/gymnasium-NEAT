import gymnasium as gym
import neat
import pickle
import os

def visualize_winner(config_file, model_file):
    """Load and visualize the trained model"""
    
    # Load the config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    
    # Load the winner genome
    with open(model_file, 'rb') as f:
        winner = pickle.load(f)
    
    print(f"Loaded model with fitness: {winner.fitness}")
    
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Create environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Run multiple episodes
    num_episodes = 5
    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Get action from neural network
            output = net.activate(observation)
            if output[0] >= 0:
                action = 1
            else:
                action = 0
            
            # Take action
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")
                break
    
    env.close()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    model_path = os.path.join(local_dir, 'winner.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: No trained model found at {model_path}")
        print("Please run train.py first to train a model.")
    else:
        visualize_winner(config_path, model_path)
