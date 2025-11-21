# NEAT + Gymnasium Environments

This project implements the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm on various Gymnasium (formerly OpenAI Gym) environments. Watch neural networks evolve in real-time as they learn to solve classic reinforcement learning challenges!

## 🎯 Overview

NEAT is a genetic algorithm for evolving artificial neural networks. This project applies NEAT to four different Gymnasium environments, demonstrating how evolutionary algorithms can train agents to perform complex tasks without traditional gradient-based learning.

## 🎮 Environments

### 1. **CartPole-v1** (`cartandpole.py`)
Balance a pole on a moving cart by applying left/right forces.
- **Inputs**: 4 (cart position, cart velocity, pole angle, pole angular velocity)
- **Outputs**: 1 (left or right action)
- **Goal**: Keep the pole balanced as long as possible

### 2. **MountainCar-v0** (`mountainCar.py`)
Drive an underpowered car up a steep hill using momentum.
- **Inputs**: 2 (position, velocity)
- **Outputs**: 1 (push left, no push, or push right)
- **Challenge**: Build momentum by rocking back and forth

### 3. **MountainCarContinuous-v0** (`mountainCarCont.py`)
Continuous action space version of the MountainCar problem.
- **Inputs**: 2 (position, velocity)
- **Outputs**: 1 (continuous force value)
- **Fitness**: Combines reward with maximum rightward position achieved

### 4. **Pendulum-v1** (`pendulum.py`)
Swing up and balance an inverted pendulum.
- **Inputs**: 3 (cosine of angle, sine of angle, angular velocity)
- **Outputs**: 1 (continuous torque)
- **Goal**: Keep the pendulum upright with minimal effort

## 🛠️ Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

This will install all required dependencies:
- `gymnasium` - OpenAI Gym environments
- `neat-python` - NEAT algorithm implementation
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `pygame` - Environment rendering (optional)

## 🚀 Usage

Simply run any of the Python files to watch NEAT evolve solutions in real-time:

```bash
# Train on CartPole
python cartandpole/cartandpole.py

# Train on MountainCar (discrete)
python mountainCar/mountainCar.py

# Train on MountainCar (continuous)
python mountainCar/mountainCarCont.py

# Train on Pendulum
python pendulum/pendulum.py
```

Each script will:
1. Load its corresponding configuration file
2. Initialize a population of random neural networks
3. Evolve the networks over multiple generations
4. Display progress statistics in the terminal
5. Return the best-performing genome (winner)

## 📁 Project Structure

```
gymnasiumTest/
├── cartandpole/
│   ├── cartandpole.py          # CartPole implementation
│   └── cartandpoleConfig.txt   # NEAT config for CartPole
├── mountainCar/
│   ├── mountainCar.py          # MountainCar (discrete) implementation
│   ├── mountainCarConfig.txt   # NEAT config for MountainCar
│   ├── mountainCarCont.py      # MountainCar (continuous) implementation
│   └── mountainCarContConfig.txt # NEAT config for MountainCar Continuous
├── pendulum/
│   ├── pendulum.py             # Pendulum implementation
│   └── pendulum.txt            # NEAT config for Pendulum
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚙️ Configuration Files

Each `.txt` file is a NEAT configuration that defines:
- **Population parameters**: Population size, fitness thresholds
- **Genome structure**: Number of inputs/outputs, activation functions
- **Mutation rates**: Connection/node addition, weight mutations
- **Species parameters**: Compatibility thresholds, stagnation limits
- **Reproduction settings**: Elitism, survival thresholds

Key configuration settings (example from `cartandpoleConfig.txt`):
- **Population size**: 100
- **Fitness threshold**: 500
- **Activation function**: tanh
- **Network type**: Feed-forward
- **Initial connections**: Full connectivity

## 🧬 How It Works

1. **Initialization**: Create a population of random neural networks
2. **Evaluation**: Each genome is tested on the environment
3. **Fitness Assignment**: Genomes receive fitness scores based on performance
4. **Selection**: Best-performing genomes are selected for reproduction
5. **Reproduction**: Create new generation through crossover and mutation
6. **Repeat**: Process continues until fitness threshold is met or max generations reached

### Fitness Functions

- **CartPole**: Cumulative reward (timesteps balanced)
- **MountainCar**: Cumulative reward (negative time penalty)
- **MountainCarCont**: Reward + bonus for maximum rightward position
- **Pendulum**: Cumulative reward (negative cost of angle and velocity)

## 📊 Monitoring Progress

The terminal displays real-time statistics:
- Current generation
- Population size
- Species count
- Best fitness achieved
- Average fitness
- Generation time

## 🎓 Learning Resources

- **NEAT Paper**: [Stanley & Miikkulainen, 2002](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **Gymnasium Docs**: [gymnasium.farama.org](https://gymnasium.farama.org/)
- **NEAT-Python**: [neat-python.readthedocs.io](https://neat-python.readthedocs.io/)

## 🔬 Experimentation Ideas

- Adjust mutation rates in config files
- Change population size and generation counts
- Modify fitness functions for different behaviors
- Add visualization to watch the best genome perform
- Save and load trained genomes using pickle
- Compare performance across different environments

## 📝 Notes

- Training times vary significantly between environments
- Pendulum requires the most generations (500) due to complexity
- CartPole typically solves fastest (5 generations often sufficient)
- MountainCar environments are challenging due to sparse rewards

## 🤝 Contributing

Feel free to experiment with different:
- Hyperparameters
- Fitness functions
- Additional Gymnasium environments
- Visualization methods

## 📄 License

This project is for educational purposes, demonstrating NEAT algorithm implementation on Gymnasium environments.
