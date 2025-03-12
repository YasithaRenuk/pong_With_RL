import pygame
import random
import numpy as np
import os
from config import WIDTH, HEIGHT, FPS, BLACK, RED
from ball import Ball
from paddle import Paddle
from rl_agent import RLAgent

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong with RL")

def get_state(ball, left_paddle, right_paddle):
    # State: [ball.x, ball.y, ball.vel_x, ball.vel_y,
    #         left_paddle.y, right_paddle.y, left_paddle.vel, right_paddle.vel]
    return np.array([ball.x, ball.y, ball.vel_x, ball.vel_y,
                     left_paddle.y, right_paddle.y, left_paddle.vel, right_paddle.vel], dtype=np.float32)

def draw_score(surface, score1, score2):
    font = pygame.font.SysFont('Calibri', 32)
    score_text = font.render(f"Player 1: {score1}  Player 2: {score2}", True, RED)
    surface.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))

def draw_metadata(surface, agent, x=10, y=HEIGHT - 120):
    font = pygame.font.SysFont('Calibri', 20)
    metadata_lines = [
        f"Generation: {agent.generation}",
        f"Memory: {len(agent.memory)} / {agent.memory.maxlen}",
        f"Batch Size: {agent.batch_size}",
        f"Gamma: {agent.gamma}",
        f"Epsilon: {agent.epsilon:.4f} (min: {agent.epsilon_min}, decay: {agent.epsilon_decay})"
    ]
    for i, line in enumerate(metadata_lines):
        text_surface = font.render(line, True, RED)
        surface.blit(text_surface, (x, y + i * (font.get_height() + 5)))

def main():
    clock = pygame.time.Clock()
    score1 = 0
    score2 = 0

    ball = Ball()
    left_paddle = Paddle(x=100 - 10, y=HEIGHT // 2 - 60, speed=5)
    right_paddle = Paddle(x=WIDTH - 100 - 10, y=HEIGHT // 2 - 60, speed=5)

    # Create RL agent with larger network and better parameters
    state_size = 8
    action_size = 3
    agent = RLAgent(state_size, action_size, 
                   min_experiences=1000,  # Collect 1000 experiences before training
                   update_every=4)        # Only train every 4 frames

    # Load saved model if it exists
    model_path = "dqn_model.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
        print("Loaded saved model.")

    frame_count = 0
    train_mode = True  # Set to False to just watch the trained agent

    run = True
    while run:
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    train_mode = not train_mode
                    print(f"Training mode: {train_mode}")

        # Current state
        state = get_state(ball, left_paddle, right_paddle)
        
        # Let the agent choose an action
        action = agent.act(state)
        
        # Apply the action
        if action == 0:
            right_paddle.vel = -right_paddle.speed
        elif action == 1:
            right_paddle.vel = right_paddle.speed
        else:
            right_paddle.vel = 0

        # Left paddle simple AI: follows the ball
        if ball.y < left_paddle.y + left_paddle.height / 2:
            left_paddle.vel = -left_paddle.speed
        elif ball.y > left_paddle.y + left_paddle.height / 2:
            left_paddle.vel = left_paddle.speed
        else:
            left_paddle.vel = 0

        # Update game objects
        ball.update()
        left_paddle.update()
        right_paddle.update()

        # Calculate reward
        reward = 0
        done = False
        
        # Calculate distance
        # current_ball_distance = abs(right_paddle.x - ball.x)
        if ball.vel_x > 0:  # Ball moving toward right paddle
            # Reward for moving toward ball
            paddle_ball_vertical_distance = abs(right_paddle.y + right_paddle.height/2 - ball.y)
            reward += max(0, 0.01 * (1 - paddle_ball_vertical_distance/HEIGHT))
        
        # Collision with right paddle: reward for hitting the ball
        if right_paddle.x <= ball.x + ball.radius <= right_paddle.x + right_paddle.width:
            if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
                ball.x = right_paddle.x - ball.radius  # reposition ball
                ball.vel_x *= -1
                ball.vel_y += random.uniform(-1, 1)
                reward += 2

                corner_threshold = 10  # pixels threshold for corner collision
                if abs(ball.y - right_paddle.y) < corner_threshold or abs(ball.y - (right_paddle.y + right_paddle.height)) < corner_threshold:
                    ball.vel_x *= 1.2
                    ball.vel_y *= 1.2

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

        # Scoring events)
        if ball.x - ball.radius <= 0:
            reward = 1  # Agent's opponent missed
            score2 += 1
            done = True
            ball.reset()
        elif ball.x + ball.radius >= WIDTH:
            reward = -2  # Agent missed
            score1 += 1
            done = True
            ball.reset()

        # Get next state
        next_state = get_state(ball, left_paddle, right_paddle)
        
        # Train the agent (only in training mode)
        if train_mode:
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
        
        # Save model occasionally
        if frame_count % 1000 == 0 and train_mode:
            agent.save(model_path)
            print(f"Model saved at frame {frame_count}")

        # Drawing
        win.fill(BLACK)
        ball.draw(win)
        left_paddle.draw(win)
        right_paddle.draw(win)
        draw_score(win, score1, score2)
        draw_metadata(win, agent)
        
        # Indicate training mode
        if not train_mode:
            font = pygame.font.SysFont('Calibri', 24)
            text = font.render("Evaluation Mode (Press T to toggle)", True, RED)
            win.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT - 30))

        pygame.display.update()
        clock.tick(FPS)

    # Save model when exiting
    if train_mode:
        agent.save(model_path)
        print("Model saved.")
    pygame.quit()

if __name__ == "__main__":
    main()
