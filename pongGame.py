# main.py
import pygame
import random
from config import WIDTH, HEIGHT, FPS, WIN_SCORE, BLACK, RED
from ball import Ball
from paddle import Paddle

pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

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

def main():
    clock = pygame.time.Clock()
    run = True
    score1 = 0
    score2 = 0

    ball = Ball()
    left_paddle = Paddle(x=100 - 10, y=HEIGHT // 2 - 60,speed=2)
    right_paddle = Paddle(x=WIDTH - 100 - 10, y=HEIGHT // 2 - 60,speed=2)
    
    while run:
        # clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    right_paddle.vel = -right_paddle.speed
                elif event.key == pygame.K_DOWN:
                    right_paddle.vel = right_paddle.speed
                if event.key == pygame.K_w:
                    left_paddle.vel = -left_paddle.speed
                elif event.key == pygame.K_s:
                    left_paddle.vel = left_paddle.speed
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    right_paddle.vel = 0
                if event.key in (pygame.K_w, pygame.K_s):
                    left_paddle.vel = 0
            
        ball.update()
        
        # Uncomment this block to enable simple AI for the left paddle
        # if ball.y < left_paddle.y + left_paddle.height / 2:
        #     left_paddle.vel = -left_paddle.speed
        # elif ball.y > left_paddle.y + left_paddle.height / 2:
        #     left_paddle.vel = left_paddle.speed
        # else:
        #     left_paddle.vel = 0

        left_paddle.update()
        right_paddle.update()

        # Check for scoring
        if ball.x - ball.radius <= 0:
            score2 += 1
            ball.reset()
        elif ball.x + ball.radius >= WIDTH:
            score1 += 1
            ball.reset()

        # Collision with left paddle
        if left_paddle.x <= ball.x - ball.radius <= left_paddle.x + left_paddle.width:
            if left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
                ball.x = left_paddle.x + left_paddle.width + ball.radius
                ball.vel_x *= -1
                # Add a random vertical offset
                ball.vel_y += random.uniform(-1, 1)
                # Speed up ball if collision is near paddle's top or bottom (corner)
                corner_threshold = 10  # pixels threshold for corner collision
                if abs(ball.y - left_paddle.y) < corner_threshold or abs(ball.y - (left_paddle.y + left_paddle.height)) < corner_threshold:
                    ball.vel_x *= 1.2
                    ball.vel_y *= 1.2

        # Collision with right paddle
        if right_paddle.x <= ball.x + ball.radius <= right_paddle.x + right_paddle.width:
            if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
                ball.x = right_paddle.x - ball.radius
                ball.vel_x *= -1
                # Add a random vertical offset
                ball.vel_y += random.uniform(-1, 1)
                # Speed up ball if collision is near paddle's top or bottom (corner)
                corner_threshold = 10  # pixels threshold for corner collision
                if abs(ball.y - right_paddle.y) < corner_threshold or abs(ball.y - (right_paddle.y + right_paddle.height)) < corner_threshold:
                    ball.vel_x *= 1.2
                    ball.vel_y *= 1.2

        win.fill(BLACK)
        ball.draw(win)
        left_paddle.draw(win)
        right_paddle.draw(win)
        draw_score(win, score1, score2)
        pygame.display.update()

        # Check win condition
        if score1 == WIN_SCORE:
            draw_win(win, 1)
            run = False
        elif score2 == WIN_SCORE:
            draw_win(win, 2)
            run = False

    pygame.quit()

if __name__ == "__main__":
    main()
