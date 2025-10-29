import numpy as np
import random

class SnakeGame:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.snake = [(self.width//2, self.height//2)]
        self.direction = (0, 1)  # 右向き
        self.spawn_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        empty = [(x, y) for x in range(self.width)
                       for y in range(self.height)
                       if (x, y) not in self.snake]
        self.food = random.choice(empty)

    def step(self, action):
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        idx = dirs.index(self.direction)
        if action == 0:
            self.direction = dirs[(idx - 1) % 4]
        elif action == 2:
            self.direction = dirs[(idx + 1) % 4]

        new_head = (self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1])

        reward = -0.1
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.done = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 10
                self.score += 1
                self.spawn_food()
            else:
                self.snake.pop()
        return self.get_state(), reward, self.done, self.score

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] + self.direction[1], head[1] - self.direction[0])
        point_r = (head[0] - self.direction[1], head[1] + self.direction[0])
        point_f = (head[0] + self.direction[0], head[1] + self.direction[1])
        danger_l = int(point_l in self.snake or not (0 <= point_l[0] < self.width and 0 <= point_l[1] < self.height))
        danger_r = int(point_r in self.snake or not (0 <= point_r[0] < self.width and 0 <= point_r[1] < self.height))
        danger_f = int(point_f in self.snake or not (0 <= point_f[0] < self.width and 0 <= point_f[1] < self.height))
        food_left = int(self.food[0] < head[0])
        food_right = int(self.food[0] > head[0])
        food_up = int(self.food[1] < head[1])
        food_down = int(self.food[1] > head[1])
        return np.array([danger_l, danger_r, danger_f, food_left,]()
