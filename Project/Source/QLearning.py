import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import features
from pysc2.lib import actions

from itertools import product

np.set_printoptions(threshold=np.nan)


class QLearningAgent(base_agent.BaseAgent):
    def __init__(self, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame()
        self.lastObservation = None
        self.lastAction = None
        self.seenActions = set()
        self.actions = list()

    def step(self, obs):
        available_actions = obs.observation["available_actions"]

        missing_actions = [x for x in np.delete(available_actions, 3) if x not in self.seenActions]

        if len(missing_actions) > 0:
            missing_action_args = {a: list(product(*[product(*n) for n in [[list(range(size)) for size in arg.sizes] for arg in self.action_spec.functions[a].args]])) for a in missing_actions}
            missing_actions_with_args = list()
            for i, l in missing_action_args.items():
                for v in l:
                    missing_actions_with_args.append((i, v))

            self.q_table = self.q_table.append(pd.DataFrame(columns=list(range(len(self.actions),len(self.actions)+len(missing_actions_with_args))))).fillna(0)
            self.actions.extend(missing_actions_with_args)

            self.seenActions.update(missing_actions)

        observation = " ".join([str(item) for sublist in obs.observation["screen"][features.SCREEN_FEATURES.unit_type.index] for item in sublist])

        if self.lastObservation and self.lastAction:
            self.learn(self.lastObservation, self.lastAction, obs.reward, observation)

        actionIdx = self.choose_action(observation, available_actions)

        self.lastObservation = observation
        self.lastAction = actionIdx

        return actions.FunctionCall(*self.actions[actionIdx])

    def choose_action(self, observation, available_actions):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            state_action = state_action.sort_values(ascending=False)
            for actionIdx, _ in state_action.iteritems():
                action = self.actions[actionIdx]
                if action[0] in available_actions:
                    return actionIdx

        # choose random action
        action = 0
        for i in range(5):
            check_action = np.random.randint(len(self.actions))
            if self.actions[check_action][0] in available_actions:
                action = check_action
                break
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.q_table.columns.values),
                    index=self.q_table.columns,
                    name=state,
                )
            )
