import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_height, state_width, action_size):
        super(DQNNetwork, self).__init__()
        # Two input channels (stacked frames)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the output size after conv layers
        conv_output_size = self._get_conv_output_size(state_height, state_width)
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def _get_conv_output_size(self, height, width):
        dummy_input = torch.zeros(1, 2, height, width)
        x = nn.functional.relu(self.conv1(dummy_input))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return int(np.prod(x.size()))
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        # Ensure contiguous memory before flattening
        x = x.contiguous().view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_height, state_width, action_size):
        self.state_height = state_height
        self.state_width = state_width
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = DQNNetwork(state_height, state_width, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.model_path = "pong_dqn_model.pth"
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Model loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        else:
            print("No saved model found. Starting with a new model.")
            return False
            
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).to(self.device)
        # Rearrange dimensions from [batch, height, width, channel] to [batch, channel, height, width]
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract and process batch components
        states = np.array([i[0][0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        states_tensor = states_tensor.permute(0, 3, 1, 2)
        
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        next_states_tensor = next_states_tensor.permute(0, 3, 1, 2)
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        self.optimizer.zero_grad()
        current_q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.model(next_states_tensor).max(1)[0]
        
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess_frame(frame):
    """Convert an RGB frame to grayscale, downsample, and normalize."""
    grayscale = np.mean(frame, axis=2).astype(np.uint8)
    downsampled = grayscale[::4, ::4]
    normalized = downsampled / 255.0
    return normalized
