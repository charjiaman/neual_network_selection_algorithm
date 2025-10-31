'''
function name: datagen.py

dataset generator from https://github.com/joshuahseaton/SubmodOptDatasetGen.git
this dataset generator generate games and each game is explained as the following:
for example, game at index 0 in games is:
{'games':
[{'agents': ['P0', 'P1', 'P2', 'P3'],
'resources': {'R0': 0.4442075380512197, 'R1': 0.29916316432317736, 'R2': 0.30745641486052444, 'R3': 0.7505630712513681, 'R4': 0.7278564019401484, 'R5': 0.6037831697594075},
'action_set': {'P0': ['R2', 'R3', 'R4'], 'P1': ['R1', 'R2', 'R3', 'R5'], 'P2': ['R0', 'R2', 'R4'], 'P3': []},
'optimal_allocations': [{'P0': 'R3', 'P1': 'R5', 'P2': 'R4', 'P3': None}]},
These are the features of game 0:
4 available agents, 5 resources with weights, In the action set, agent P0 can choose either R2, R3 or R4; agent P1 can choose either R1, R2, R3 or R5; agent P2 can choose R0, R2 or R4, agent 3 does not have action.
'''

from __future__ import annotations
import typing
import json
import numpy as np
import random

random.seed(21)
np.random.seed(21)

NUM_RESOURCES = 4
NUM_AGENTS = 2
to_keep = 2
actions_to_remove = NUM_AGENTS * NUM_RESOURCES - to_keep

class GameInstance:
    def __init__(self):
        self.agents = list()
        self.resources = dict()
        self.action_set = dict()
        self.set_of_optimal = list()

    def create_resources(self, resource_list: np.ndarray):
        for i in range(len(resource_list)):
            self.resources.update({"R" + str(i): resource_list[i]})

    def create_agents(self, num_agents: int):
        for i in range(num_agents):
            self.agents.append("P" + str(i))

    def create_action_profiles(self):
        for agent in self.agents:
            for resource in self.resources:
                if agent not in self.action_set:
                    self.action_set.update({agent: []})
                #print(self.action_list)
                #print(agent in self.action_list)
                self.action_set[agent].append(resource)

    def remove_actions(self, num_actions: int):
        agents = self.agents.copy()
        to_remove: int = num_actions if num_actions < len(self.agents) * len(self.resources)\
            else len(self.agents) * len(self.resources)
        for i in range(to_remove):
            agent = random.choice(agents)
            self.action_set[agent].remove(random.choice(self.action_set[agent]))
            if len(self.action_set[agent]) == 0:
                agents.remove(agent)

    def find_optimal_profiles(self):
        arg_max: float = 0.0
        #print(self.to_json())
        a = self.create_initial_profile()
        while a is not None:
            f = self.evaluate(a)
            if f > arg_max:
                arg_max = f
                self.set_of_optimal.clear()
            if f >= arg_max:
                self.set_of_optimal.append(self.convert_action_profile(a.copy()))
            a = self.get_next_action_profile(a)

    def create_initial_profile(self) -> dict:
        a = dict()   #this is the action profile being evaluated
        for agent in self.agents:
            a.update({agent: None})
            if len(self.action_set[agent]) > 0:
                a.update({agent: 0})
        #print(f"a (initial): {a}")
        return a

    def get_next_action_profile(self, a: dict) -> typing.Union[dict, None]:
        valid: bool = False
        for agent in self.agents:
            if a[agent] is not None:  # This skips agents with an empty action set
                a[agent] += 1
                if a[agent] > len(self.action_set[agent]) - 1:
                    if agent is not self.agents[-1]:
                        a[agent] = 0
                else:
                    valid = True
                    break
        #print(f"a: {a}")
        #print(f"valid: {valid}")
        if valid:
            return a
        else:
            return None
            
    def evaluate(self, a: dict) -> float:
        u = set()
        v: float = 0.0

        for agent in self.agents:
            if a[agent] is not None:
                u.add(self.action_set[agent][a[agent]])
        #print(f"u: {u}")
        for resource in u:
            # print(f"resource: {resource}")
            # print(f"self.resources[resource]: {self.resources[resource]}")
            v += self.resources[resource]
        #print(v)
        return v

    def convert_action_profile(self, a: dict) -> dict:
        #print(f"action profile to be converted: {a}")
        for agent in self.agents:
            if a[agent] is not None:
                a[agent] = self.action_set[agent][a[agent]]
        return a

    def to_dict(self):
        output = dict()
        output.update({"agents": self.agents})
        output.update({"resources": self.resources})
        output.update({"action_set": self.action_set})
        output.update({"optimal_allocations": self.set_of_optimal})
        return output

    def to_json(self):
        return json.dumps(self.to_dict())

def tedst():
    g = GameInstance()
    g.create_resources(np.random.rand(NUM_RESOURCES))
    g.create_agents(NUM_AGENTS)
    g.create_action_profiles()
    g.remove_actions(actions_to_remove)
    g.find_optimal_profiles()
    print(g.agents)
    print(g.resources)
    print(g.action_set)
    print(g.to_json())

def SetCoverDataGenerator():
    games = list()
    for i in range(200):
        g = GameInstance()
        g.create_resources(np.random.rand(NUM_RESOURCES))
        g.create_agents(NUM_AGENTS)
        g.create_action_profiles()
        g.remove_actions(actions_to_remove)
        g.find_optimal_profiles()
        games.append(g.to_dict())
    #print({"\n\n games \n": games})
    return json.dumps({"games": games})
