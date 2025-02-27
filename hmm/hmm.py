import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p
        self.viterbi_path = []
        self.viterbi_probability = 0

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        What is the probability of the input observation sequence given the HMM parameters?

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """
        # I found this useful for my forward algorithm implementation:
        # https://github.com/AzharuddinKazi/Forward-Algorithm-HMM/blob/master/Forward_Algorithm_HMM.py
        
        ### Step 0. Check inputs and format

        self._check_input_states(input_observation_states)

        # handle edge case of empty input
        if len(input_observation_states) == 0:
            return []
        
        ### Step 1. Initialize variables

        A = self.transition_p
        pi = self.prior_p
        B = self.emission_p
        observations = [self.observation_states_dict[obs] for obs in input_observation_states]

        M = len(observations)
        N = pi.shape[0]
        
        # Initialization
        alpha = np.zeros((M, N))
            
        # Since their are no previous states. the probability of being in state 1 at time 1 is given as product of:
        # 1. Initial Probability of being in state 1
        # 2. Emmision Probability of symbol O(1) being in state 1
        alpha[0, :] = pi * B[:,observations[0]]
            
        ### Step 2. Calculate probabilities

        # if we know the previous state i,then the probability of being in state j at time t+1 is given as product of:
        # 1. Probability of being in state i at time t
        # 2. Transition probability of going from state i to state j
        # 3. Emmision Probability of symbol O(t+1) being in state j

        for t in range(1, M):
            for j in range(N):
                for i in range(N):
                    alpha[t, j] += alpha[t-1, i] * A[i, j] * B[j, observations[t]]
        
        ### Step 3. Return final probability 

        P = np.sum(alpha[M-1,:])

        return P

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        ### Step 0. Check inputs and format

        self._check_input_states(decode_observation_states)

        ### Step 1. Initialize variables
        
        ### Step 2. Calculate Probabilities and Backtrack

        ### Step 3. Traceback

        ### Step 4. Return best hidden state sequence

        # return best_path
    
    def _check_input_states(self, input_states: np.ndarray) -> None:
        if input_states.ndim != 1:
            raise ValueError("Input observation states must be a 1D array.")
        if len(input_states) == 0:
            raise Warning("Input observation states is empty!")
        
        # check if input observation states are of the same type
        input_types = {type(i) for i in input_states}
        if len(input_types) != 1:
            raise ValueError("Input observation states must be of the same type.")

        # check if input observation states are same type as observation states in the model
        if input_types == {type(obs) for obs in self.observation_states}:
            pass
        else:
            raise ValueError("Input observation states must be of the same type as observation states in the model.")
        
        # check if input observation states are in the model
        not_found = []
        for obs in input_states:
            if obs not in self.observation_states_dict.keys():
                not_found.append(obs)
        if len(not_found) > 0:
            raise ValueError(f"Observation state {','.join(not_found)} not found in observation states in HMM.")
