
from matplotlib import pyplot as plt
import gymnasium as gym
from typing import Optional
from collections import defaultdict
import numpy as np
import random
import pygame
agent_cool=[]
target_cool=[]
is_training=True
rng = np.random.default_rng()
matrix = rng.integers(low=0, high=2, size=(10,10))
x=random.randint(0,matrix.shape[0]-1)
import random
y=random.randint(0,matrix.shape[1]-1)
coordinates1=[random.randint(0,matrix.shape[0]-1),random.randint(0,matrix.shape[1]-1)]
coordinates2=[random.randint(0,matrix.shape[0]-1),random.randint(0,matrix.shape[1]-1)]
matrix[x,y]=-3
matrix[coordinates1[0],coordinates1[1]]=-2
print(coordinates1)
episode_over=False
total_reward=0

class GridWorldEnv(gym.Env):
    global matrix
    global coordinates1
    global x
    global is_training
    global y
    metadata={"render_modes":["human","rgb_array"],"render_fps":4}
    def __init__(self, render_mode="human",size: int = 10):
        learning_rate = 0.02        # How fast to learn (higher = faster but less stable)      # Number of hands to practice
        start_epsilon = 1.0         # Start with 100% random actions
        epsilon_decay = 0.01  # Reduce exploration over time
        final_epsilon = 0.1 
        # The size of the square grid (5x5 by default)
        self.size = size
        self.window_size=512
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([x,y], dtype=np.int32)
        self._target_location = np.array([coordinates1[0], coordinates1[1]], dtype=np.int32)
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )
        
        
        print('observe:',self.observation_space)
        

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }
        self.list=[]
        self.q_values = np.zeros((size**2,size**2,4))
        for i in range(0,15):
            num=random.randint(0,99)
            self.list.append(num)
            self.q_values[num]=1
                    
        learning_rate = 0.05
        epsilon_decay = 0.01
        final_epsilon = 0.1
        discount_factor = 0.95,
        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards
        # Exploration parameters
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
        self.window = None
        self.clock = None
    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    def update(
        self,
        obs: tuple[int, int,bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int,bool],
    ):
        """Update Q-value based on experience.
        #q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (reward + discount_factor_g * np.max(q[q_new_state_idx]) - q[q_state_action_idx])

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)

        future_q_value = (not terminated) * np.max(self.q_values[next_obs['agent'],next_obs['target']])

        target = reward + self.discount_factor[0] * (future_q_value - self.q_values[obs['agent'],obs['target']][action])
        # How wrong was our current estimate?
        temporal_difference = self.lr * target

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs['agent'],obs['target']][action] = (
            self.q_values[next_obs['agent'],obs['target']][action] + temporal_difference
        )
        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def _get_obs(self):
        global agent_cool
        global target_cool
        agent_cool=self._agent_location[0]*10+self._agent_location[1]
        target_cool=self._target_location[0]*10+self._target_location[1]
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        
        agent_target=self._agent_location[0]*10+self._agent_location[1]
        target_target=self._target_location[0]*10+self._target_location[1]
        return {"agent": agent_target, "target": target_target}
        """
        return{"agent": agent_cool,"target":target_cool}
    def get_action(self, obs: tuple[int, int,bool]) -> int:
        """Choose an action using epsilon-greedy strategy.
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        # With probability (1-epsilon): exploit (best known action)
        else:
            return(int(np.argmax(self.q_values[obs['agent'],obs['target']])))
            
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.
        
        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        render_mode="human"
        if not is_training:
            assert render_mode is None or render_mode in self.metadata["render_modes"]
            self.render_mode = render_mode
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode=="human":
            self._render_frame()
        return observation, info
    def step(self, action,steps):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]
        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        info = self._get_info()
        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        distance = np.linalg.norm(self._agent_location - self._target_location)
        reward = 2000 if terminated else -1*distance+(-0.1*steps)
        observation = self._get_obs()
        truncated=True if steps>=200 else False
        if self.render_mode == "human" and not is_training:
            self._render_frame()
        if(observation['agent'] in self.list):
            reward=-200
            print("CANCER")
            terminated=True
        return observation,reward,terminated,truncated,info
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        def get_action(obs: tuple[int, int, bool]):
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

            # With probability (1-epsilon): exploit (best known action)
            else:
                return int(np.argmax(self.q_values[obs['agent'],obs['target']]))
    """
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    """

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)

env=GridWorldEnv()
observation, info = env.reset()
print(observation, info)
print(matrix)
print('resert',env.reset())
observation, info = env.reset()

episodes=50000
total_steps=0

for i in range(1,episodes):
    observation,info=env.reset()
    terminated = False
    truncated=False
    steps=0
    while not terminated and not truncated:
        steps+=1
        action = env.get_action(observation)
        next_observation,reward,terminated,truncated,info = env.step(action,steps)
        env.update(observation, action, reward, terminated, next_observation)
        observation=next_observation
    env.decay_epsilon()
    print('episode',i,'steps',steps)
    total_steps+=steps
    print('avg_steps:',total_steps/i)
is_training=False
episodes=100
for i in range(1,episodes):
    observation,info=env.reset()
    terminated = False
    truncated=False
    steps=0
    while not terminated and not truncated:
        steps+=1
        action = env.get_action(observation)
        next_observation,reward,terminated,truncated,info = env.step(action,steps)
        env.update(observation, action, reward, terminated, next_observation)
        observation=next_observation
    env.decay_epsilon()
    print('episode',i,'steps',steps)
    total_steps+=steps
    print('avg_steps:',total_steps/i)