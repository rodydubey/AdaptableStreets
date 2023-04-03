import numpy as np
from copy import deepcopy
from machin.machin.frame.algorithms import MADDPG
from machin.machin.frame.algorithms.utils import safe_call, soft_update, hard_update
import torch as t
import torch.nn as nn
from gym_sumo.actors.agent import Agent
import os
import time
np.random.seed(42)
t.manual_seed(42)

t.set_default_device('cpu')

CustomMADDPG = MADDPG
class CustomMADDPG(MADDPG):
    @staticmethod
    def action_transform_function(raw_output_action, *_, actor=False):
        if raw_output_action.shape[-1]>1:
            if actor:
                transformed_act = gumbel_softmax(raw_output_action, hard=True)
            else:
                transformed_act = onehot_from_logits(raw_output_action, eps=0.0)
            # print('raw output', raw_output_action, transformed_act)
        else:
            transformed_act = raw_output_action
        return {"action": transformed_act}

    @staticmethod
    def _update_sub_policy(
        batch_size,
        batches,
        next_actions_t,
        actor_index,
        policy_index,
        actors,
        actor_targets,
        critics,
        critic_targets,
        critic_visible_actors,
        actor_optims,
        critic_optims,
        update_value,
        update_policy,
        update_target,
        atf,
        acf,
        scf,
        rf,
        criterion,
        discount,
        update_rate,
        update_steps,
        update_counter,
        grad_max,
        visualize,
        visualize_dir,
        backward_func,
    ):
        ensemble_batch = batches[policy_index]
        ensemble_n_act_t = next_actions_t[policy_index]
        visible_actors = critic_visible_actors[actor_index]

        actors[actor_index][policy_index].train()
        critics[actor_index].train()

        with t.no_grad():
            # only select visible actors
            all_next_actions_t = [
                ensemble_n_act_t[a_idx]
                if a_idx != actor_index
                else atf(
                    safe_call(
                        actor_targets[actor_index][policy_index],
                        ensemble_batch[a_idx][3],
                    )[0],
                    ensemble_batch[a_idx][5],
                    actor=False
                )
                for a_idx in visible_actors
            ]
            all_next_actions_t = acf(all_next_actions_t)

            all_actions = [ensemble_batch[a_idx][1] for a_idx in visible_actors]
            all_actions = acf(all_actions)

            all_next_states = [ensemble_batch[a_idx][3] for a_idx in visible_actors]
            all_next_states = scf(all_next_states)

            all_states = [ensemble_batch[a_idx][0] for a_idx in visible_actors]
            all_states = scf(all_states)

        # Update critic network first
        # Generate target value using target critic.
        with t.no_grad():
            reward = ensemble_batch[actor_index][2]
            terminal = ensemble_batch[actor_index][4]
            next_value = safe_call(
                critic_targets[actor_index], all_next_states, all_next_actions_t
            )[0]
            next_value = next_value.view(batch_size, -1)
            y_i = rf(
                reward, discount, next_value, terminal, ensemble_batch[actor_index][5]
            )

        cur_value = safe_call(critics[actor_index], all_states, all_actions)[0]
        value_loss = criterion(cur_value, y_i.to(cur_value.device))

        if update_value:
            critics[actor_index].zero_grad()
            backward_func(value_loss)
            nn.utils.clip_grad_norm_(critics[actor_index].parameters(), grad_max)
            critic_optims[actor_index].step()

        # Update actor network
        all_actions = [ensemble_batch[a_idx][1] for a_idx in visible_actors]
        # find the actor index in the view range of critic
        # Eg: there are 4 actors in total: a_0, a_1, a_2, a_3
        # critic may have access to actor a_1 and a_2
        # then:
        #     visible_actors.index(a_1) = 0
        #     visible_actors.index(a_2) = 1
        # visible_actors.index returns the (critic-)local position of actor
        # in the view range of its corresponding critic.
        all_actions[visible_actors.index(actor_index)] = atf(
            safe_call(
                actors[actor_index][policy_index], ensemble_batch[actor_index][3]
            )[0],
            ensemble_batch[actor_index][5],
            actor=True,
        )
        cur_pol_out = all_actions[visible_actors.index(actor_index)]['action']
        print('CURRENT POL OUT',cur_pol_out)
        # s = all_actions[visible_actors.index(actor_index)]['action']
        # print(min(s.numpy()), max(s.numpy()), s.shape)
        # print(s)
        all_actions = acf(all_actions)
        act_value = safe_call(critics[actor_index], all_states, all_actions)[0]

        # "-" is applied because we want to maximize J_b(u),
        # but optimizer workers by minimizing the target
        act_policy_loss = -act_value.mean()
        act_policy_loss += (t.square(cur_pol_out)).mean() * 1e-3

        if update_policy:
            actors[actor_index][policy_index].zero_grad()
            backward_func(act_policy_loss)
            nn.utils.clip_grad_norm_(
                actors[actor_index][policy_index].parameters(), grad_max
            )
            actor_optims[actor_index][policy_index].step()

        # Update target networks
        if update_target:
            if update_rate is not None:
                soft_update(
                    actor_targets[actor_index][policy_index],
                    actors[actor_index][policy_index],
                    update_rate,
                )
                soft_update(
                    critic_targets[actor_index], critics[actor_index], update_rate
                )
            else:
                if update_counter % update_steps == 0:
                    hard_update(
                        actor_targets[actor_index][policy_index],
                        actors[actor_index][policy_index],
                    )
                    hard_update(critic_targets[actor_index], critics[actor_index])

        actors[actor_index][policy_index].eval()
        critics[actor_index].eval()
        return -act_policy_loss.item(), value_loss.item()
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, discrete=True):
        super().__init__()
        # self.in_func = nn.BatchNorm1d(state_dim, momentum=None)
        # self.in_func.weight.data.fill_(1)
        # self.in_func.bias.data.fill_(0)
        # self.in_func = lambda x: x
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        if discrete:
            # self.output_nonlin = nn.Softmax(dim=1)
            self.output_nonlin = nn.Identity()
        else:
            self.output_nonlin = nn.Sigmoid()

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = self.output_nonlin(self.fc3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        # This critic implementation is shared by the prey(DDPG) and
        # predators(MADDPG)
        # Note: For MADDPG
        #       state_dim is the dimension of all states from all agents.
        #       action_dim is the dimension of all actions from all agents.
        super().__init__()
        # self.in_func = nn.BatchNorm1d(state_dim + action_dim, momentum=None)
        # self.in_func.weight.data.fill_(1)
        # self.in_func.bias.data.fill_(0)
        # self.in_func = lambda x: x

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


def create_actors(observe_dim, action_num, hidden_dim):
    actors = []
    for obs_size, act_size in zip(observe_dim, action_num):
        if act_size == 1:
            discrete = False
        else:
            discrete = True
        actors.append(Actor(obs_size, act_size, hidden_dim, discrete=discrete))
        actors[-1].eval()
    return actors


class MADDPGAgent(Agent):
    def __init__(self, env, observe_dim, action_num, agent_num,
                 hidden_dim, batch_size, actor_learning_rate, critic_learning_rate,
                 tau, gamma):
        self.env = env
        self.observe_dim = observe_dim
        self.action_num = action_num
        self.agent_num = agent_num
        self.tmp_observations_list = [[] for _ in range(agent_num)]

        actors = create_actors(observe_dim, action_num, hidden_dim)
        critic = Critic(sum(observe_dim), sum(action_num), hidden_dim)
        critic.eval()

        self.maddpg = CustomMADDPG(
            actors,
            [deepcopy(actor) for actor in actors],
            [deepcopy(critic) for _ in range(agent_num)],
            [deepcopy(critic) for _ in range(agent_num)],
            t.optim.Adam,
            nn.MSELoss(reduction="sum"),
            critic_visible_actors=[list(range(agent_num))] * agent_num,
            batch_size=batch_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            update_rate=tau,
            discount=gamma,
            # gradient_max=0.5,
            # pool_size=1,
            use_jit=False
        )

    def act(self, states, epnum, explore=True):
        with t.no_grad():

            states = [{'state': sk} for sk in states]
            raw_actions = self.maddpg._act_api_general(states, use_target=False)
            result = []

            decayrate = 100
            decay = np.exp(-epnum/decayrate)
            for i, (action, *others) in enumerate(raw_actions):
                if self.action_num[i] == 1:
                    if not explore:
                        decay *= 0
                    normal_scalar = 0.2
                    noise_uniform = t.randn(1, device=action.device)*normal_scalar
                    action = action + decay*noise_uniform
                    action = t.clip(action, 0.1, 0.9)
                    result.append((action, action))
                else:
                    action_probs = action
                    if explore:
                        action = gumbel_softmax(action_probs, hard=True)
                    else:
                        action = onehot_from_logits(action_probs)
                    result.append((t.argmax(action, dim=1, keepdim=True), action))

                    # batch_size = action.shape[0]
                    # dist = t.distributions.Categorical(probs=action)
                    # # dist = t.distributions.Categorical(logits=action)
                    # action_disc = dist.sample([batch_size, 1]).view(batch_size, 1)
                    # result.append((action_disc, action))
                    # result.append((action_disc, t.nn.functional.one_hot(action_disc, num_classes=action.shape[1]).squeeze(0)))

            actions = [r[0][0] for r in result]
            action_probs = [r[1] for r in result]
        return actions, action_probs

    def prepare_states(self, states):
        states = [
            t.tensor(st, dtype=t.float32).view(1, self.observe_dim[i]) for i, st in enumerate(states)
        ]
        return states

    def train(self):
        return self.maddpg.update()
    
    def store_transition(self, old_states, action_probs, states, rewards, terminals):
        for tmp_observations, ost, act, st, rew, term in zip(
                    self.tmp_observations_list,
                    old_states,
                    action_probs,
                    states,
                    rewards,
                    terminals,
                ):
                    tmp_observations.append(
                        {
                            "state": {"state": ost},
                            "action": {"action": act},
                            "next_state": {"state": st},
                            "reward": float(rew),
                            "terminal": term,
                        }
                    )

    def process_transitions(self):
        self.maddpg.store_episodes(self.tmp_observations_list)
        self.tmp_observations_list = [[] for _ in range(self.agent_num)]

    def get_buffer_size(self):
        return self.maddpg.replay_buffers[0].size()
    
    def save(self, folder):
        date_now = time.strftime("%Y%m%d%H%M")
        full_path = f"{folder}/maddpg_machin_{date_now}"
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        self.maddpg.save(full_path)

    def load(self, folder):
        self.maddpg.load(folder)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = t.autograd.Variable(t.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return t.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(t.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=t.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = t.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=True)
    return -t.log(-t.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return nn.functional.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y