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
        self._num_observations = len(self.observation_states)

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

        ### Step 0. Check inputs and format
        
        input_types = self._check_input_states(input_observation_states)

        ### Step 1. Initialize variables
        
        if input_types == {str}:
            states = [self.observation_states_dict[s] for s in input_observation_states]
        else:
            states = input_observation_states

        start_state = states[0] 
        prob = self.prior_p[start_state]
        prev_state = start_state

        ### Step 2. Calculate probabilities

        for i in range(1, len(states)):
            curr_state = states[i]
            prob *= self.transition_p[prev_state][curr_state]
            prev_state = curr_state

        ### Step 3. Return final probability 
        
        forward_probability = prob

        return forward_probability

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

        input_types = self._check_input_states(decode_observation_states)

        ### Step 1. Initialize variables

        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
                
        if input_types == {str} or input_types == {np.str_}:
            state2index = {state: index for index, state in self.hidden_states_dict.items()}
            obs = [state2index[s] for s in decode_observation_states]
        else:
            obs = decode_observation_states
        
        ### Step 2. Calculate Probabilities and Backtrack

        for t in range(1, len(obs)):
            for s in range(len(self.hidden_states)):
                max_prob, max_state = max(
                    (viterbi_table[t-1][prev_s] * self.transition_p[prev_s][s], prev_s)
                    for prev_s in range(len(self.hidden_states))
                )
            viterbi_table[t][s] = max_prob * self.emission_p[s][obs[t]]
            best_path[t] = max_state

        ### Step 3. Traceback

        best_hidden_state_sequence = np.zeros(len(decode_observation_states), dtype=int)
        best_hidden_state_sequence[-1] = np.argmax(viterbi_table[-1])

        for t in range(len(decode_observation_states) - 2, -1, -1):
            best_hidden_state_sequence[t] = best_path[t + 1]

        ### Step 4. Return best hidden state sequence

        best_hidden_state_sequence = np.array(
            [self.hidden_states[state] for state in best_hidden_state_sequence], dtype=str
        )

        return best_path
    
    def _check_input_states(self, input_states: np.ndarray) -> None:
        if input_states.ndim != 1:
            raise ValueError("Input observation states must be a 1D array.")
        if len(input_states) == 0:
            raise ValueError("Input observation states cannot be empty.")
        
        input_types = {type(i) for i in input_states}
        if len(input_types) != 1:
            raise ValueError("Input observation states must be of the same type.")

        return input_types        
        # if input_types != {str}:
        #     raise ValueError("Input observation states must be strings.")
        
        # for state in input_states:
        #     if state not in self.observation_states_dict:
        #         raise ValueError(f"Input observation state {state} is not in the model.")
