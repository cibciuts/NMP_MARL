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
![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/scenarios.png)

![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/jungle_init_big.png)

![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/jungle_trained_big.png)

![](https://github.com/cibciuts/NMP_MARL/blob/master/figures/perf.png)


