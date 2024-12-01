import argparse
import cv2
import gymnasium as gym
import FigFollowerEnv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import traceback

from RAgent import RAgent

torch.set_grad_enabled(False)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_reward_performance(info):
    performance = info['accum_improv'] / info['max_reward']
    return performance


def get_all_models(path):
    DIR = Path(path)
    files = list(DIR.glob('*.pt'))
    new_files = [(int(file.name.split('.')[0].split('_')[1]), str(file)) for file in files]
    new_files.sort(key = lambda f: f[0])
    return new_files


def to_tensor(obs):
    tensor = torch.from_numpy(obs)
    tensor = tensor.float() / 255
    return tensor.permute(2, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,
        help="Path to the folder where the trained models are stored. It will read all .pt files.")
    parser.add_argument("--repeats", type=int, default=5,
        help="Make the current agent interact with the env for n episodes.")
    parser.add_argument("--show", action='store_true',
        help="Show in a window what is the agent seeing.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parameters
    args = parse_args()
    repeats = args.repeats
    show = args.show
    # Creating parts of the algorithm
    env = gym.make('FigFollowerEnv-v1', render_mode="rgb_array", fps=10, nodes = 6, max_time = 60)
    agent = RAgent(env.action_space.shape[0], 0.1, device=DEVICE).to(DEVICE).eval()
    accum_reward = 0.0
    models = get_all_models(args.folder)
    models.insert(0, (0, 'Base'))
    all_rewards = np.zeros((len(models), repeats))
    all_preform = np.zeros((len(models)))
    try:
        for i, model in enumerate(models):
            if model[0] > 0:
                agent.load_state_dict(
                    torch.load(
                        model[1],
                        weights_only=True, 
                        map_location=torch.device(DEVICE)
                    )
                )
            for rep in range(repeats):
                print(f'{i+1}/{len(models)} {model[1]}) repeat: {rep+1}/{repeats}')
                ended = False
                accum_reward = 0.0
                next_obs, info = env.reset()
                hiddens = [
                    torch.zeros((1, 1, 128)).to(DEVICE),
                    torch.zeros((1, 1, 4)).to(DEVICE)
                ]
                while not ended:
                    obs = next_obs
                    input = to_tensor(obs).to(DEVICE).unsqueeze(dim=0)
                    actions, logprobs, hiddens = agent.actor(hiddens, input)
                    next_obs, reward, terminated, truncated, info = env.step(actions.squeeze().cpu().numpy())
                    ended = terminated or truncated
                    accum_reward += reward
                    if show:
                        cv2.imshow('Evaluate', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                        key = cv2.waitKey(1)
                        if key != -1:
                            break
                all_rewards[i, rep] = accum_reward
                all_preform[i] = get_reward_performance(info)
        x_data = np.array([model[0] for model in models])
        reward_data = all_rewards.mean(axis=1)
        fig, (ax0, ax1) = plt.subplots(2)
        ax0.plot(x_data, reward_data)
        ax1.plot(x_data, all_preform)

        #ax0.set_xlabel('Pasos de tiempo')
        ax0.set_title('Recompensa acumulada promedio')
        ax1.set_xlabel('Pasos de tiempo')
        ax1.set_title('Porcentaje de recompensa positiva obtenida del posible')
        ax1.set_ylim([0.0, 1.0])
        plt.show()
    except Exception as e:
        del(agent)
        env.close()
        print(traceback.format_exc())
    env.close()