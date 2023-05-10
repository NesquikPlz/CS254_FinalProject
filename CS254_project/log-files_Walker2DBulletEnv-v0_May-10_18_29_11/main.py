import gymnasium as gym
from sb3_contrib import TRPO

env = gym.make("Walker2d-v4", render_mode='human')
observation, info = env.reset()

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("trpo_pendulum")

for _ in range(1000):
	env.render()
	action = env.action_space.sample()
	observation, reward, terminated, truncated, info = env.step(action)

	if terminated or truncated:
		observation, info = env.reset()

env.close()