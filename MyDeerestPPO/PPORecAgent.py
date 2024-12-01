class PPORecAgent():
    """
    Interface for implementing Recurrent  agents-
    """
    def actor(self, hiddens, obs):
        """
        Receives the hidden states and an observation with shape [N_envs, (shape of your data)]
        Returns tuple: actions [N_envs, (shape of your actions)], logprobs [N_envs]
        """
        
        raise NotImplementedError


    def critic(self, obs):
        """
        Receives an observation with shape [N_envs, (shape of your data)]
        Returns estimated returns [N_envs]
        """

        raise NotImplementedError


    def actor_evaluate(self, hiddens, obs, actions):
        """
        Should be able to handle observations and actions with shape [Timesteps, N_Envs, (shape of your data)]
        and must return logprobs with shape [Timesteps, N_envs]
        """

        raise NotImplementedError

    def critic_evaluate(self, obs):
        """
        Should be able to handle observations with shape [Timesteps, N_Envs, (shape of your data)]
        and return estimated returns with shape [Timesteps, N_Envs]
        """

        raise NotImplementedError