#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In[1]:
import gc


def calculate_tensors():
    num_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj), obj.size())
                num_tensors += 1
        except:
            pass

    print("num_tensors: {}".format(num_tensors))


# In[2]:


import torch
import numpy as np

from env import make_pytorch_env
from decision_transformer.models.decision_transformer import DecisionTransformer


# In[3]:


class MyClass:
    # vars to class
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# In[4]:


loaded_model = torch.load("./exp/2023.03.16/170104-default/model.pt")
loaded_pretrain_model = torch.load("./exp/2023.03.16/170104-default/pretrain_model.pt")

variant = loaded_model['args']
args = MyClass(**variant)

# In[5]:


loaded_model.keys()


# In[6]:


def _get_env_spec(variant):
    #####env = gym.make(variant["env"])
    env = make_pytorch_env(args)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]

    env.close()
    return state_dim, act_dim, action_range


# In[7]:


state_dim, act_dim, action_range = _get_env_spec(vars(args))
target_entropy = -act_dim

MAX_EPISODE_LEN = 4000

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    action_range=action_range,
    max_length=variant["K"],
    eval_context_length=variant["eval_context_length"],
    max_ep_len=MAX_EPISODE_LEN,
    hidden_size=variant["embed_dim"],
    n_layer=variant["n_layer"],
    n_head=variant["n_head"],
    n_inner=4 * variant["embed_dim"],
    activation_function=variant["activation_function"],
    n_positions=1024,
    resid_pdrop=variant["dropout"],
    attn_pdrop=variant["dropout"],
    stochastic_policy=True,
    ordering=variant["ordering"],
    init_temperature=variant["init_temperature"],
    target_entropy=target_entropy,
).to(device=args.device)

# ## Rascunhos

# In[8]:


import pickle


# In[9]:


# TODO: save state_mean, state_std

def _load_dataset(env_name):
    dataset_path = f"./data/{env_name}.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
    print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
    print("=" * 50)

    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    trajectories = [trajectories[ii] for ii in sorted_inds]

    return trajectories, state_mean, state_std


# In[10]:


device = args.device
torch.no_grad()

# Load the weights on the model
model.load_state_dict(loaded_model['model_state_dict'])


# Set model to evaluation mode
model.eval()
# Convert model to GPU
model.to(device=args.device);

# In[11]:


# Nao gostei disso pq tem a ver com o Dataset
# TODO: Load
offline_trajs, state_mean, state_std = _load_dataset(args.env)
state_mean = torch.from_numpy(state_mean).to(device=device)
state_std = torch.from_numpy(state_std).to(device=device)

# In[12]:


vec_env = make_pytorch_env(args)

# In[13]:


num_envs = 1
reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001

max_ep_len = 4000
use_mean = True  # False # True
state = vec_env.reset()
unfinished = np.ones(num_envs).astype(bool)
mode = 'normal'  # delayed

# In[14]:


# Not sure:
target_return = [variant['eval_rtg'] * reward_scale] * num_envs

ep_return = target_return

target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
    num_envs, -1, 1
)
timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
    num_envs, -1
)

# In[15]:


states = (
    torch.from_numpy(state)
    .reshape(num_envs, state_dim)
    .to(device=device, dtype=torch.float32)
).reshape(num_envs, -1, state_dim)

actions = torch.zeros(0, device=device, dtype=torch.float32)

rewards = torch.zeros(0, device=device, dtype=torch.float32)

ep_return = target_return
target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
    num_envs, -1, 1
)
timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
    num_envs, -1
)

# episode_return, episode_length = 0.0, 0
episode_return = np.zeros((num_envs, 1)).astype(float)
episode_length = np.full(num_envs, np.inf)

# In[16]:


max_ep_len

# In[17]:


# TODO: read the paper and figure it out if reward state is necessary


# In[18]:


actions.shape

# In[19]:


for t in range(max_ep_len):
    # add padding
    actions = torch.cat(
        [
            actions,
            torch.zeros((num_envs, act_dim), device=device).reshape(
                num_envs, -1, act_dim
            ),
        ],
        dim=1,
    )
    rewards = torch.cat(
        [
            rewards,
            torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
        ],
        dim=1,
    )

    state_pred, action_dist, reward_pred = model.get_predictions(
        (states.to(dtype=torch.float32) - state_mean) / state_std,
        actions.to(dtype=torch.float32),
        rewards.to(dtype=torch.float32),
        target_return.to(dtype=torch.float32),
        timesteps.to(dtype=torch.long),
        num_envs=num_envs,
    )
    state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
    reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

    # the return action is a SquashNormal distribution
    action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
    if use_mean:
        action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
    action = action.clamp(*model.action_range)

    # TODO: nao entendo pq esta gerando um [] a mais e se isso atrapalhou no training
    # print("action: {}".format(action[0]))
    state, reward, done, _ = vec_env.step(action.detach().cpu().numpy()[0])
    #vec_env.render()
    # state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

    # eval_env.step() will execute the action for all the sub-envs, for those where
    # the episodes have terminated, the envs will be reset. Hence we use
    # "unfinished" to track whether the first episode we roll out for each sub-env is
    # finished. In contrast, "done" only relates to the current episode
    # TODO: nao sei pq, mas o unfinished precisa por [0]
    episode_return[unfinished] += reward[unfinished[0]].reshape(-1, 1)
    # episode_return[unfinished] += reward[unfinished[0]].reshape(-1, 1)

    actions[:, -1] = action
    state = (
        torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
    )
    states = torch.cat([states, state], dim=1)
    del state
    # print("states: {}".format(states))
    # TODO: n sei pq, mas tive que por np.array em reward (na vdd sei, apenas 1 evaluate..)
    reward = torch.from_numpy(np.array(reward)).to(device=device).reshape(num_envs, 1)
    # reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
    rewards[:, -1] = reward

    if mode != "delayed":
        pred_return = target_return[:, -1] - (reward * reward_scale)
    else:
        pred_return = target_return[:, -1]
    target_return = torch.cat(
        [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
    )

    timesteps = torch.cat(
        [
            timesteps,
            torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                num_envs, 1
            )
            * (t + 1),
        ],
        dim=1,
    )

    if t == max_ep_len - 1:
        done = np.ones(done.shape).astype(bool)

    if np.any(done):
        ind = np.where(done)[0]
        unfinished[ind] = False
        episode_length[ind] = np.minimum(episode_length[ind], t + 1)

    if not np.any(unfinished):
        break

    calculate_tensors()

# In[ ]:


gc.get_objects()

# In[ ]:


# In[ ]:


calculate_tensors()

# In[ ]:


vec_env.close()

# In[ ]:


print("Episode Return: {}".format(episode_return[0][0]))

# In[ ]:


states.shape

# ## Fim dos Rascunhos

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:



