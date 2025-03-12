import pygame
from config import HEIGHT, RED

class Paddle:
    def __init__(self, x, y, width=5, height=120, speed=1):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.vel = 0

    def update(self):
        self.y += self.vel
        # Keep paddle on screen
        if self.y < 0:
            self.y = 0
        if self.y + self.height > HEIGHT:
            self.y = HEIGHT - self.height

    def draw(self, surface):
        pygame.draw.rect(surface, RED, pygame.Rect(self.x, int(self.y), self.width, self.height))
