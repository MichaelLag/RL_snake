import numpy as np
import RL_agent_env
import RL_Bridge
from snake_env import my_snake
import pickle

# Agent parameters
agent_info = {
    'agent_type': RL_agent_env.Agent,
    'agent_parameters': {
    'network_config': {
        'state_dim': 147,
        'num_hidden_units': 512,
        'num_actions': 3
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9,
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 5000,
    'minibatch_sz': 16,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001#0.6
    }
}
# Env parameters
environment_info = {'env_type': my_snake,}

# initialize experiment
RL_shell = RL_Bridge.RL_shell_env_agent(environment_info, agent_info)
RL_shell.initialize_agent()

th_inds = [200,500,800,1000,1500,2000,2497,2498,2499,2500]
TH = {i: np.zeros((1,RL_shell.s_size)) for i in th_inds}
W  = pickle.load( open( "weights_{}.p".format(147), "rb" ))
RL_shell.agent.network.set_weights(W)
for i in np.arange(2501):
    RL_shell.initialize_env()
    RL_shell.start()
    print('Iterration...{}'.format(i))
    j = 0
    while not RL_shell.is_terminal():
        j += 1
        state = RL_shell.step()
        if j==1000: # protect from infinite loop
            break
        if i in [500,1000, 2000,2500]:
            RL_shell.env.print_state()
        if i in th_inds:
            TH[i] = np.append(TH[i],state.reshape(1,-1),axis=0)

    print(RL_shell.agent.sum_rewards)
    print(RL_shell.env.life)

W = RL_shell.agent.network.get_weights()

pickle.dump(TH,open( "TH_{}.p".format(agent_info['agent_parameters']['network_config']['state_dim']), "wb" ))
pickle.dump(W,open( "weights_{}.p".format(agent_info['agent_parameters']['network_config']['state_dim']), "wb" ))



