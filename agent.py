import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI,Direction,Point
from model import Linear_QNet, QTrainer
from helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE= 1000
LR = 0.001
BLOCK=20
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(14,512,3)
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        #TODO: model, trainer

    def get_state(self,game):
        head = game.snake[0]
        point_l = Point(head.x-20,head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x,head.y-20)
        point_d = Point(head.x,head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN


        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x<game.head.x,
            game.food.x>game.head.x,
            game.food.y<game.head.y,
            game.food.y>game.head.y
        ]
        state = state+self.free_space_features(game)
        return np.array(state,dtype=float)


    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

        

    def train_long_memory(self):

        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory

        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)



    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_games)   # keep some exploration late
        final_move = [0, 0, 0]

        space = state[11:14]
        safe = [i for i in range(3) if space[i] >= 1.0] or [0, 1, 2]  # if none safe, allow all

        if random.randint(0, 200) < self.epsilon:
            move = random.choice(safe)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                pred = self.model(state0)
            mask = torch.full((3,), float('-inf'))
            mask[safe] = 0
            move = torch.argmax(pred + mask).item()

        final_move[move] = 1
        return final_move 


    def reachable_area(self, game, start):
        """Free cells reachable from `start`, treating the body (minus tail) as walls."""
        body = {(p.x, p.y) for p in game.snake[:-1]}
        sx, sy = start.x, start.y
        if not (0 <= sx < game.w and 0 <= sy < game.h) or (sx, sy) in body:
            return 0
        q, seen = deque([(sx, sy)]), {(sx, sy)}
        while q:
            x, y = q.popleft()
            for dx, dy in ((BLOCK, 0), (-BLOCK, 0), (0, BLOCK), (0, -BLOCK)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < game.w and 0 <= ny < game.h
                        and (nx, ny) not in body and (nx, ny) not in seen):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return len(seen)

    def free_space_features(self, game):
        head = game.snake[0]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        dirs = [clock_wise[idx],              # straight
                clock_wise[(idx + 1) % 4],    # right turn
                clock_wise[(idx - 1) % 4]]    # left turn
        delta = {Direction.RIGHT: (BLOCK, 0), Direction.LEFT: (-BLOCK, 0),
                Direction.UP: (0, -BLOCK), Direction.DOWN: (0, BLOCK)}
        feats = []
        for d in dirs:
            dx, dy = delta[d]
            area = self.reachable_area(game, Point(head.x + dx, head.y + dy))
            feats.append(min(area / len(game.snake), 1.0))  # 1.0 = enough room
        return feats

def train():
    plot_scores = []

    plot_mean_scores = []

    total_score = 0

    record = 0

    agent = Agent()

    game =  SnakeGameAI()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward,done,score = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game,',agent.n_games,'Score',score,"Record:",record)

            plot_scores.append(score)
            total_score+=score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__ == '__main__':
    train()



