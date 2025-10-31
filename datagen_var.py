'''
function name: datagen_add.py

datagen_add.py is used to for genfixdata.py where padd and conact several different kinds of dataframe together.
refer to original datagen.py
'''

from __future__ import annotations
import typing
import json
import numpy as np
import random

random.seed(21)
np.random.seed(21)

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

def test():
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

def SetCoverDataGenerator(NUM_RESOURCES, NUM_AGENTS, to_keep, NUM_GAMES):
    actions_to_remove = NUM_AGENTS * NUM_RESOURCES - to_keep
    #print('agent', NUM_AGENTS)
    #print('res', NUM_RESOURCES)
    #print('to_keep', to_keep)
    #print('actions to remove', actions_to_remove)
    games = list()
    for i in range(NUM_GAMES):
        g = GameInstance()
        g.create_resources(np.random.rand(NUM_RESOURCES))
        g.create_agents(NUM_AGENTS)
        g.create_action_profiles()
        g.remove_actions(actions_to_remove)
        g.find_optimal_profiles()
        games.append(g.to_dict())
    #print({"\n\n games \n": games})
    return json.dumps({"games": games})
