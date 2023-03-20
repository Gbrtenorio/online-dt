#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!cat ~/.bashrc


# In[2]:


#!mv ../drone_dataset.pkl .


# In[3]:


#!pip3 install --upgrade protobuf==3.20.0 


# In[4]:


#!pip3 install transformers==4.5.1
#!pip3 install -U tokenizers
# The code below just solve many problems lol
#!pip3 uninstall tokenizers -y


# In[5]:


from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym
import d4rl
import torch
import numpy as np

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger

from env import make_pytorch_env

MAX_EPISODE_LEN = 4000 # Warning: there is a similar variable in data.py! 


# In[6]:


import sys
sys.argv = ['']

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--env", type=str, default="drone_dataset")
#parser.add_argument("--env", type=str, default="antmaze-large-diverse-v2")

# model options
parser.add_argument("--K", type=int, default=20)
parser.add_argument("--embed_dim", type=int, default=512)
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--activation_function", type=str, default="relu")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--eval_context_length", type=int, default=5)
# 0: no pos embedding others: absolute ordering
parser.add_argument("--ordering", type=int, default=0)

# shared evaluation options
parser.add_argument("--eval_rtg", type=int, default=3600)
parser.add_argument("--num_eval_episodes", type=int, default=10)

# shared training options
parser.add_argument("--init_temperature", type=float, default=0.1)
#parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
parser.add_argument("--warmup_steps", type=int, default=10000)

# pretraining options
parser.add_argument("--max_pretrain_iters", type=int, default=1)
parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

# finetuning options
parser.add_argument("--max_online_iters", type=int, default=1500)
parser.add_argument("--online_rtg", type=int, default=7200)
parser.add_argument("--num_online_rollouts", type=int, default=1)
parser.add_argument("--replay_size", type=int, default=1000)
parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
parser.add_argument("--eval_interval", type=int, default=10)

# environment options
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
parser.add_argument("--save_dir", type=str, default="./exp")
parser.add_argument("--exp_name", type=str, default="default")

args = parser.parse_args()


# In[7]:


class Experiment:
    def __init__(self, variant):

        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
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
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        self.log_temperature_optimizer = torch.optim.Adam(
            [self.model.log_temperature],
            lr=1e-4,
            betas=[0.9, 0.999],
        )

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        #####env = gym.make(variant["env"])
        env = make_pytorch_env(args)
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        #action_range = [-0.999999, 0.999999]
        
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        
        print("action_range: {}".format(action_range))
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

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

    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")
        print("----------------")
        print("eval_envs: {}".format(eval_envs))
        print("loss_fn: {}".format(loss_fn))
        
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = (
            SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None
        )
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)

            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)

            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)

            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1

    def __call__(self):

        utils.set_seed_everywhere(args.seed)

        import d4rl

        def loss_fn(
            a_hat_dist,     # action_preds
            a,              # action_target
            attention_mask, # padding_mask
            entropy_reg,    # self.model.temperature().detach()
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()
            
            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)
            
            '''
            print("a_hat_dist : {}".format(a_hat_dist))
            print("a : {}".format(a))
            torch.save(a,"a.pt")
            print("a_hat_dist.log_likelihood(a) : {}".format(a_hat_dist.log_likelihood(a)))
            #print("attention_mask : {}".format(attention_mask))
            print("log_likelihood: {}".format(log_likelihood))
            print("loss inside jupyter: {} of type: {}".format(loss,type(loss)))
            '''
            
            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                import d4rl

                #####env = gym.make(env_name)
                env = make_pytorch_env(args)
                env.seed(seed)
                '''
                if hasattr(env.env, "wrapped_env"):
                    env.env.wrapped_env.seed(seed)
                elif hasattr(env.env, "seed"):
                    env.env.seed(seed)
                else:
                    pass
                '''
                '''
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                '''

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")
                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, loss_fn)

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, loss_fn)
            online_envs.close()

        eval_envs.close()


# In[8]:


utils.set_seed_everywhere(args.seed)
experiment = Experiment(vars(args))

print("=" * 50)
experiment()


# In[ ]:


def study_env(env):
    
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6]
        
    print("state_dim: {}".format(state_dim))
    print("act_dim: {}".format(act_dim))
    print("action_range: {}".format(action_range))


# In[ ]:


my_env = make_pytorch_env(args)
their_env = gym.make('antmaze-large-diverse-v2')


# In[ ]:


study_env(my_env)


# In[ ]:


study_env(their_env)


# In[ ]:


my_env.reset()
my_env.step(2)


# In[ ]:


args


# In[ ]:


their_env.action_space


# In[ ]:


their_env.reset()
their_env.step(2)


# In[ ]:


#experiment.variant
#experiment.model.forward


# In[ ]:





# In[ ]:


loss


# In[ ]:


experiment.model.forward


# In[ ]:


experiment.model


# In[ ]:





# In[ ]:





# In[ ]:


import math
math.log(1e-310)


# In[ ]:


action_preds = torch.load('action_preds.pt')


# In[ ]:


a = torch.load("a.pt")


# In[ ]:


action_preds.log_likelihood(a)


# In[ ]:


sefude = action_preds.log_likelihood(a)


# In[ ]:


a


# In[ ]:


a


# In[ ]:


a


# In[ ]:


torch.nan_to_num(sefude)


# In[ ]:


action_preds


# In[ ]:





# In[ ]:


a[0][0]


# In[ ]:


math.log(-0.3)


# In[ ]:





# In[ ]:


action_preds.entropy().mean()


# In[ ]:


action_preds.log_likelihood(10)


# In[ ]:


action_preds.perplexity


# In[ ]:


import torch
state_dim = 4
hidden_size = 512

embed_state = torch.nn.Linear(state_dim, hidden_size).to('cuda')
embed_state_2 = torch.load('embed_state.pt').to('cuda')
states = torch.load('states.pt').to('cuda')
state_embeddings = embed_state(states)
state_embeddings_2 = torch.load('state_embeddings.pt').to('cuda')


# In[ ]:


states[0]


# In[ ]:





# In[ ]:


print("state_embeddings {}".format(state_embeddings))


# In[ ]:


print("state_embeddings 2 {}".format(state_embeddings_2))


# In[ ]:


embed_state.weight


# In[ ]:


embed_state_2.weight


# In[ ]:


embed_state


# In[ ]:


embed_state_2


# In[ ]:


embed_state(states)


# In[ ]:


embed_state_2(states)


# In[ ]:





# In[ ]:


stoppppppppppp


# In[ ]:





# In[ ]:


import torch
torch.__version__


# In[ ]:


get_ipython().system('pip list | grep torch')


# In[ ]:


get_ipython().system('pip3 install torch --upgrade')


# In[ ]:


# Normalizando as rewards pra ver se resolve o problema


# In[ ]:


import pickle

with open('data/drone_dataset.pkl', 'rb') as f:
    my_data = pickle.load(f)
    
with open('data/antmaze-large-diverse-v2.pkl', 'rb') as f:
    their_data = pickle.load(f)


# In[ ]:


for data in my_data:
    rewards = data['actions']
    print("max: {}".format(np.max(rewards)))
    print("min: {}".format(np.min(rewards)))
    print("mean: {}".format(np.mean(rewards)))
    print('----------------')


# In[ ]:


np.shape(my_data[0]['observations'])


# In[ ]:


np.shape(their_data[0]['observations'])


# In[ ]:





# In[ ]:


import pickle

with open('data/drone_dataset.pkl', 'rb') as f:
    my_data = pickle.load(f)
    

for data in my_data:
    
    #data['rewards']   = np.float32(data['rewards'].flatten())
    #data['terminals'] = np.float32(data['terminals'].flatten())
    data['actions'] = np.float32(np.minimum(np.maximum(data['actions'], -1), 1))
    #data['observations'] = np.float32(data['observations'])
    #data['next_observations'] = np.float32(data['next_observations'])
    
with open('data/drone_dataset.pkl', 'wb') as handle:
    pickle.dump(my_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


(v - v.min()) / (v.max() - v.min())


# In[ ]:




