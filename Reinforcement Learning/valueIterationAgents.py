# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()


    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        for i in range(self.iterations):
            tmp_q = util.Counter()
            
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    #calculate maximum Q value for each action
                    maxQ = max([self.computeQValueFromValues(state, action)for action in self.mdp.getPossibleActions(state)])
                    tmp_q[state] = maxQ
                #continue if the state is terminal
                else:
                    continue

            self.values = tmp_q


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        sum = 0

        #calculate Q values using the value function
        #Sum of T(s,a,s')[R(s,a,s')+ gamma (v*(s'))
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            #list of (nextState, prob) pairs
            state_t = transition[0]
            prob = transition[1]
            r = self.mdp.getReward(state, action, state_t)
            #calculate sum based on the formula
            sum += prob * (r + self.discount * self.values[state_t])

        return sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #no legal actions if state is terminal
        if self.mdp.isTerminal(state):
            return None

        #dict to keep pairs of actions and values
        action_value = {}
        
        #compute  values
        for action in self.mdp.getPossibleActions(state):
            action_value[action] = self.computeQValueFromValues(state, action)
        
        #get action (key) corresponding to maximum value 
        return max(action_value, key = action_value.get)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # The difference here is that here we will update only one state in each iteration
    # because of this, we iterate on one state at a time

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            l = len(states)
            state = states[i % l]
            if not self.mdp.isTerminal(state):
                #calculate maximum Q value for each action of the state
                max_q = max([self.getQValue(state,action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = max_q
            else:
                continue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        #dict to keep prodecessors(value) of each state(key)
        predecessors = collections.defaultdict(set)

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for stateAndP in self.mdp.getTransitionStatesAndProbs(state, action):
                    #get (nextState, prob) pairs
                    next_state = stateAndP[0]
                    prob = stateAndP[1]
                    # if probability = 0, there is no point in adding the predecessor since
                    #we will never end up in that state.
                    if prob > 0:
                        predecessors[next_state].add(state)
                
                
        queue = util.PriorityQueue()

        # for each non-terminal state:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                #current value of the state
                s_values = self.values[state]
                                       
                #all Q values possible considering the legal actions
                q_values = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                                       
                #find the difference between current value and maximum Q value
                diff = abs(s_values - max(q_values))
                                       
                #push -diff to the queue because it is a min heap
                queue.push(state, -diff)

        #for all self.iterations
        for i in range(self.iterations):
            #the algorithm ends                       
            if queue.isEmpty():
                break
                
            s = queue.pop()
            
            #update self.values for non-terminal states
            if not self.mdp.isTerminal(s):
                q_values = [self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)]
                self.values[s] = max(q_values)
            
            for p in predecessors[s]:
                #current q value of the state
                p_values = self.values[p]
                
                #all Q values possible considering the legal actions
                q_values = [self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)]
                
                #calculate the difference between current value and maximum Q value of p
                diff = abs(p_values - max(q_values))
                
                #add p to the queue
                if diff > self.theta:
                    queue.update(p, -diff)

