import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64,
                 update_every=4, min_experiences=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size, hidden_size=64).to(self.device)
        self.target_model = DQN(state_size, action_size, hidden_size=64).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.generation = 0
        self.update_every = update_every
        self.min_experiences = min_experiences
        self.learn_step_counter = 0
        
        # For normalization
        self.state_mean = np.zeros(state_size)
        self.state_std = np.ones(state_size)
        self.norm_samples = 0

    def normalize_state(self, state):
        # Update running statistics for normalization
        if self.norm_samples < 1000:  # Only update during initial phase
            if self.norm_samples == 0:
                self.state_mean = state
                self.state_std = np.ones_like(state)
            else:
                self.state_mean = (self.state_mean * self.norm_samples + state) / (self.norm_samples + 1)
                self.state_std = np.sqrt(
                    (self.state_std**2 * self.norm_samples + (state - self.state_mean)**2) / (self.norm_samples + 1)
                )
            self.norm_samples += 1
            
        # Prevent division by zero
        self.state_std = np.maximum(self.state_std, 1e-5)
        
        # Normalize
        return (state - self.state_mean) / self.state_std

    def remember(self, state, action, reward, next_state, done):
        # Store normalized states
        norm_state = self.normalize_state(state)
        norm_next_state = self.normalize_state(next_state) 
        self.memory.append((norm_state, action, reward, norm_next_state, done))

    def act(self, state):
        # Normalize state
        norm_state = self.normalize_state(state)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(norm_state).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        self.model.train()
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        # Only train if we have enough samples
        if len(self.memory) < self.min_experiences:
            return
            
        # Only update every few steps
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_every != 0:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([transition[1] for transition in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([1 if transition[4] else 0 for transition in minibatch])).to(self.device)

        # Get current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update target network every 100 learning steps
        if self.learn_step_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.generation += 1

    def save(self, filename):
        save_dict = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'epsilon': self.epsilon,
            'generation': self.generation
        }
        torch.save(save_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.epsilon = checkpoint['epsilon']
        self.generation = checkpoint['generation']
        self.norm_samples = 1000  # Assume we've collected enough normalization samples