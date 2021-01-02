
class RL_shell_env_agent:
    def __init__(self, environment_info, agent_info):
        self.environment_info = environment_info
        self.agent_info       = agent_info
        self.s_size = None
        self.a_size = None
        print('finish init')


    def is_terminal(self):
        return self.env.terminal

    def initialize_agent(self):
        self.agent_parameters = self.agent_info['agent_parameters']
        self.agent  = self.agent_info['agent_type']()
        self.agent.agent_init(self.agent_parameters)
        self.s_size = self.agent_parameters['network_config']['state_dim']
        self.a_size = self.agent_parameters['network_config']['num_actions']
        print('initialized agent')


    def initialize_env(self):
        self.env    = self.environment_info['env_type'](9)



    def start(self):
        self.agent.agent_start(self.env.state_flat_long())
    def step(self):
        _,terminal,reward = self.env.move_step(self.agent.last_action)
        state = self.env.state_flat_long()

        if not terminal:
            self.agent.agent_step(reward, state)
        else:
            self.agent.agent_end(reward)

        return state
