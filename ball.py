import pygame
import random
from config import WIDTH, HEIGHT, BLUE, BLACK

class Ball:
    def __init__(self, radius=15):
        self.radius = radius
        self.reset()
    
    def reset(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        ball_vel = 3
        direction = random.choice([0, 1])
        angle = random.choice([0, 1, 2])
        
        if direction == 0:
            if angle == 0:
                self.vel_y = -2 * ball_vel
                self.vel_x = ball_vel
            elif angle == 1:
                self.vel_y = -ball_vel
                self.vel_x = ball_vel
            else:
                self.vel_y = -ball_vel
                self.vel_x = 2 * ball_vel
        else:
            if angle == 0:
                self.vel_y = 2 * ball_vel
                self.vel_x = ball_vel
            elif angle == 1:
                self.vel_y = ball_vel
                self.vel_x = ball_vel
            else:
                self.vel_y = ball_vel
                self.vel_x = 2 * ball_vel
        
        if random.choice([True, False]):
            self.vel_x *= -1

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        # Bounce off the top and bottom
        if self.y - self.radius <= 0 or self.y + self.radius >= HEIGHT:
            self.vel_y *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), self.radius)
