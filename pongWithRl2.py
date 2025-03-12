import pygame
import random
import numpy as np
import time
import os
from config import WIDTH, HEIGHT, FPS, WIN_SCORE, BLACK, RED
from ball import Ball
from paddle import Paddle
from dqn_agent import DQNAgent, preprocess_frame

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong RL with Vision (PyTorch)")

# Dimensions after preprocessing
FRAME_WIDTH = WIDTH // 4
FRAME_HEIGHT = HEIGHT // 4

def draw_score(surface, score1, score2):
    font = pygame.font.SysFont('Calibri', 32)
    score_text = font.render(f"Player 1: {score1}  Player 2: {score2}", True, RED)
    surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))

def draw_win(surface, winner):
    font = pygame.font.SysFont('Calibri', 64)
    win_text = font.render(f"Player {winner} Wins", True, RED)
    surface.fill(BLACK)
    surface.blit(win_text, (WIDTH // 2 - win_text.get_width() // 2,
                            HEIGHT // 2 - win_text.get_height() // 2))
    pygame.display.update()
    pygame.time.delay(2000)

def get_frame():
    """Capture the current game frame and preprocess it"""
    # Get pixel data from the pygame surface
    frame = pygame.surfarray.array3d(win)
    
    # Convert from (width, height, 3) to (height, width, 3)
    frame = frame.transpose((1, 0, 2))
    
    # Preprocess the frame (resize, grayscale, normalize)
    processed_frame = preprocess_frame(frame)
    
    return processed_frame

def main():
    clock = pygame.time.Clock()
    run = True
    score1 = 0
    score2 = 0
    
    # Initialize game objects
    ball = Ball()
    left_paddle = Paddle(x=100 - 10, y=HEIGHT // 2 - 60, speed=5)
    right_paddle = Paddle(x=WIDTH - 100 - 10, y=HEIGHT // 2 - 60, speed=5)
    
    # RL agent setup
    action_size = 3  # Up, Down, Stay
    agent = DQNAgent(FRAME_HEIGHT, FRAME_WIDTH, action_size)
    
    # RL parameters
    batch_size = 32
    done = False
    reward = 0
    frame_interval = 4  # Process every 4th frame (skip frames for speed)
    frame_count = 0
    train_interval = 5  # Train every 5 steps
    train_count = 0
    save_interval = 1000  # Save model every 1000 frames
    
    # Render the initial frame
    win.fill(BLACK)
    ball.draw(win)
    left_paddle.draw(win)
    right_paddle.draw(win)
    draw_score(win, score1, score2)
    pygame.display.update()
    
    # Get initial frames (we need two consecutive frames to capture motion)
    current_frame = get_frame()
    # Wait a bit to get a different frame
    time.sleep(0.01)
    win.fill(BLACK)
    ball.draw(win)
    left_paddle.draw(win)
    right_paddle.draw(win)
    draw_score(win, score1, score2)
    pygame.display.update()
    next_frame = get_frame()
    
    # Stack frames to capture motion (ball velocity)
    state = np.stack((current_frame, next_frame), axis=2)
    state = np.expand_dims(state, axis=0)  # Add batch dimension
    
    while run:
        frame_count += 1
        clock.tick(FPS)
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save model before quitting
                agent.save_model()
                run = False
                
            # Manual control for left paddle (optional, since we'll use automatic control)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    left_paddle.vel = -left_paddle.speed
                elif event.key == pygame.K_s:
                    left_paddle.vel = left_paddle.speed
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_w, pygame.K_s):
                    left_paddle.vel = 0
        
        # AUTOMATIC CONTROL for left paddle (as requested)
        if ball.y < left_paddle.y + left_paddle.height / 2:
            left_paddle.vel = -left_paddle.speed
        elif ball.y > left_paddle.y + left_paddle.height / 2:
            left_paddle.vel = left_paddle.speed
        else:
            left_paddle.vel = 0
        
        # Only process every frame_interval frames for the agent
        if frame_count % frame_interval == 0:
            # RL agent chooses an action
            action = agent.act(state)
            
            # Execute the action
            if action == 0:  # Move up
                right_paddle.vel = -right_paddle.speed
            elif action == 1:  # Move down
                right_paddle.vel = right_paddle.speed
            else:  # Stay still
                right_paddle.vel = 0
        
        # Save the prev_state before updating
        prev_state = state
        
        # Update game objects
        ball.update()
        left_paddle.update()
        right_paddle.update()
        
        # Check for scoring
        if ball.x - ball.radius <= 0:
            score2 += 1
            ball.reset()
            done = True
            reward = -10  # Opponent scored, big penalty
        elif ball.x + ball.radius >= WIDTH:
            score1 += 1
            ball.reset()
            done = True
            reward = -10  # Agent missed, big penalty
        
        # Collision with left paddle
        if left_paddle.x <= ball.x - ball.radius <= left_paddle.x + left_paddle.width:
            if left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
                ball.x = left_paddle.x + left_paddle.width + ball.radius
                ball.vel_x *= -1
                ball.vel_y += random.uniform(-1, 1)
                corner_threshold = 10
                if abs(ball.y - left_paddle.y) < corner_threshold or abs(ball.y - (left_paddle.y + left_paddle.height)) < corner_threshold:
                    ball.vel_x *= 1.2
                    ball.vel_y *= 1.2
                reward = 1  # Small reward for opponent hit (ball coming back to agent)
        
        # Collision with right paddle (RL agent)
        if right_paddle.x <= ball.x + ball.radius <= right_paddle.x + right_paddle.width:
            if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
                ball.x = right_paddle.x - ball.radius
                ball.vel_x *= -1
                ball.vel_y += random.uniform(-1, 1)
                corner_threshold = 10
                if abs(ball.y - right_paddle.y) < corner_threshold or abs(ball.y - (right_paddle.y + right_paddle.height)) < corner_threshold:
                    ball.vel_x *= 1.2
                    ball.vel_y *= 1.2
                reward = 5  # Big reward for successful hit
        
        # Calculate additional rewards based on paddle position relative to ball
        # Encourage the paddle to stay aligned with the ball
        if ball.vel_x > 0:  # Ball is moving towards the agent
            paddle_center = right_paddle.y + right_paddle.height / 2
            distance_to_ball = abs(paddle_center - ball.y)
            alignment_reward = -distance_to_ball / HEIGHT  # Negative reward based on distance
            reward += alignment_reward
        
        # Render the game
        win.fill(BLACK)
        ball.draw(win)
        left_paddle.draw(win)
        right_paddle.draw(win)
        draw_score(win, score1, score2)
        
        # Draw RL information
        font = pygame.font.SysFont('Calibri', 18)
        epsilon_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, RED)
        win.blit(epsilon_text, (WIDTH - 150, HEIGHT - 30))
        
        pygame.display.update()
        
        # Get new frame
        current_frame = next_frame
        next_frame = get_frame()
        
        # Stack frames to capture motion
        state = np.stack((current_frame, next_frame), axis=2)
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        # Remember the transition (only every frame_interval frames)
        if frame_count % frame_interval == 0:
            agent.remember(prev_state, action, reward, state, done)
            
            # Train the model periodically
            train_count += 1
            if train_count % train_interval == 0:
                agent.replay(min(len(agent.memory), batch_size))
            
            if done:
                done = False
            
            # Reset reward for next step
            reward = 0
            
        # Save model periodically
        if frame_count % save_interval == 0:
            agent.save_model()
        
        # Check win condition
        if score1 == WIN_SCORE:
            draw_win(win, 1)
            # Save model before ending the game
            agent.save_model()
            run = False
        elif score2 == WIN_SCORE:
            draw_win(win, 2)
            # Save model before ending the game
            agent.save_model()
            run = False
    
    pygame.quit()

if __name__ == "__main__":
    main()