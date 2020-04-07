This is the code for the paper "NEURAL MESSAGE-PASSING FOR LARGE-SCALE MULTI-AGENT REINFORCEMENT LEARNING", by Dr Kha Vo and Distinguished Professor Chin-Teng Lin.

## Dependencies
```
torch 1.1.0
schnetpack 
ase 3.17.0
torch-scatter 1.4.0
torch_geometric 1.3.2
torch-sparse 0.4.3
torch_cluster 1.4.5
```

## Method
![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/Kha.png)
A graph-based learning method using neural-message passing (NMP) technique is proposed to tackle large-scale multi-agent reinforcement learning (MARL) scenarios. The graph-based technique is embedded into REINFORCE, a simple but powerful policy-gradient RL algorithm. Actions are distributed in each sub-graph of a limited size to better focus on proper information and significantly reduce training time. 

## Scenarios
Three scenarios were implemented (Jungle, Battle, Deception). For custom configuration of each scenario, please manual change the init parameters of class `GridWorld` in the corresponding notebook. For example, in Jungle scenario, initialize the scenario by
```
W = GridWorld(width=WIDTH, height=HEIGHT, agent_types=AGENT_TYPES, 
                  n_agents_each_type=N_AGENTS_EACHTYPE,
                  terminal_frame=FRAMES_PER_EPISODE) 
```
For custom reward mechanism or custom agents' interactions, modify method `transition()` in class `GridWorld`. For instance, the following snippet will check if this predator is next to a food cell or not. If so, increase its property `kill` by `1` and update the current reward as `+1`.
``` 
view = self.get_view(agent.pos, agent.properties['viewrange'], 'prey')
next_to_food = True if view[0,1]>=1 or view[1,0]>=1 or view[1,2]>=1 or view[2,1]>=1 else False
if next_to_food: 
    agent.properties['kills'] += 1
    agent.current_reward += 1
```
![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/scenarios.png)

## Model
Besides REINFORCE (Policy-Gradient), NMP-RL is implemented using `schnetpack` package. The principle of the implementation is shown as below.
- Each moving agent has a state that is a flattened array from a 2D view around its position. Property `viewrange` indicates the depth of the view. For example, `viewrange=1` means that the agent can see up to 1 cell around its position, equivalent to the nearest 9 cells. Adjust this value via `agent['viewrange']`.
- Before a feedforwarding step, `batch_data` is prepared by collecting `states`, `connectivity`, and `distances` from all agents. Here, `states` are tensor for all flattened state of all agents, `connectivity` indicates connected agents within `viewrange` of each other, and `distances` are geometric distances pair-wise.
- During each feedforward iteration, action for each agent is sampled after being feedforwarded via the training network as
```
probs = model(batch_data) # forward pass, # probs shape: n_agents x n_action_options 
m = Categorical(probs)
action_sample = m.sample() # sampled actions of all agents at this time step
action_buffer.append(m.log_prob(action_sample))
```
- During each backpropagation iteration, the log probabilities (variable `log_prob`) of sampled actions are modulated with the long-term returns via variable `returns`, using factor `R=0.99`. 

![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/jungle_init_big.png)

![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/jungle_trained_big.png)

## Training Speed
The training speed with compared methods is shown in the below figure, demonstrating our superiority in speed. To re-implement this result, run script `compare.py`.
![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/perf.png)


