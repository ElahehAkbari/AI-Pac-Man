# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #we need to check if we are running to ghosts.
        #if the distance between pacman and a ghost is less than or equal to 1
        #they might catch pacman and the score will be the lowest possible, -infinity.
        #by the end of this loop, we do not have to worry about handling ghosts anymore.
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if manhattanDistance(newPos,ghostPos) <= 1:
                return float('-inf')

        minDist= float('inf')
        
        #if the number of foods in the next state is less than that in our current state,
        #it means we will end up eating a food and we return the highest possible score, +infinity.
        if len(currentGameState.getFood().asList()) > len(newFood.asList()):
            return float('inf')
        
        #we need to find the closest food in the current food list.
        #we do this by calculating manhattan distance between our new position and a food.
        #the minimum distance is eventually saved.
        for food in newFood.asList():
            tmpDist = manhattanDistance(newPos,food)
            minDist = min(minDist, tmpDist)
                    
        #the smaller the distance, the better! -> we inverse the minimum distance and return it.        
        return 1/(minDist)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex = 0, depth = 0):
        #keep agentIndex(incremented each time) in the acceptable range
        #which is the number of agents (pacman and ghosts)
        #if index is larger than the range, mod is calculated for index.
        if agentIndex >= gameState.getNumAgents():
            agentIndex %= gameState.getNumAgents()
            
        #check if the state is terminal, meaning that pacman either won, lost or depth is zero
        if gameState.isWin() or gameState.isLose() or (agentIndex == 0 and (depth == self.depth)):
            return self.evaluationFunction(gameState)

        #call max_value if the next agent is max (pacman), pacman is the maximizer.
        #depth is incremented in each ply (once all agents take their actions)
        if agentIndex == 0 and depth < self.depth:
            depth += 1
            return self.max_value(gameState, agentIndex, depth)
        #cal min_value if the next agent is min (ghost), ghost is the minimizer.
        else:
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        #saves a value
        v = float('-inf')
        #saves the last action taken
        final_action = "null"
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agentIndex)

        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call minimax function and we pass successor as the first argument, the result is saved in s_value.
        
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agentIndex, action)
            #call minimax on successor
            s_value = self.minimax(successor, agentIndex + 1, depth)
            
            #we save the maximum value in v in each interation
            v = max(v, s_value)

            #we save the action that will eventually be taken to return it later on.
            if v == s_value:
                final_action = action
                
        #we return final action in our penultimate depth.
        if depth == 1:
            return final_action
        
        #v is returned as the value.
        return v

    def min_value(self, gameState, agentIndex, depth):
        #saves a value
        v = float('inf')
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agentIndex)
        
        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call minimax function and we pass successor as the first argument, the result is saved in s_value.
        
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agentIndex, action)
            #call minimax on successor
            s_value = self.minimax(successor, agentIndex + 1, depth)
            
            #save min value in v in each interation
            v = min(v, s_value)
            
        #v is returned as the value.
        return v
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState)
        #util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta(self, gameState, agentIndex = 0, depth = 0, alpha = float("-inf"), beta = float("inf")):
        #keep agentIndex(incremented each time) in the acceptable range
        #which is the number of agents (pacman and ghosts)
        #if index is larger than the range, mod is calculated for index.
        if agentIndex >= gameState.getNumAgents():
            agentIndex %= gameState.getNumAgents()
            
        #check if the state is terminal, meaning that pacman either won, lost or depth is zero
        if gameState.isWin() or gameState.isLose() or (agentIndex == 0 and (depth == self.depth)):
            return self.evaluationFunction(gameState)

        #call max_value if the next agent is max (pacman), pacman is the maximizer.
        #depth is incremented in each ply (once all agents take their actions)
        if agentIndex == 0 and depth < self.depth:
            depth += 1
            return self.max_value(gameState, agentIndex, depth,alpha, beta)
        #cal min_value if the next agent is min (ghost), ghost is the minimizer.
        else:
            return self.min_value(gameState, agentIndex, depth,alpha, beta)
        
    
    def max_value(self, gameState, agent, depth, alpha, beta):
        #saves a value
        v = float('-inf')
        #saves the last action taken
        final_action = "null"
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agent)

        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call alphabeta function and we pass successor as the first argument, the result is saved in s_value.
        
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agent, action)
            #call alpha_beta on successor, passing alpha and beta as well
            s_value = self.alpha_beta(successor, agent + 1, depth, alpha, beta)
            
            #we save the maximum value in v in each interation
            v = max(v, s_value)

            #we save the action that will eventually be taken to return it later on.
            if v == s_value:
                final_action = action
            
            #v is returned as the value if its greater than beta.
            if v > beta:
                return v
            #alpha is calculated as the maximum of alpha and value by far.
            alpha = max(alpha, v)

        #we return final action in our penultimate depth.
        if depth == 1:
            return final_action
        
        #v is returned as the value.
        return v

    def min_value(self, gameState, agent, depth, alpha, beta):
        #saves a value
        v = float('inf')
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agent)

        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call alpha_beta function and we pass successor as the first argument, the result is saved in s_value.
        
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agent, action)
            #call alpha_beta on successor
            s_value = self.alpha_beta(successor, agent + 1, depth, alpha, beta)
            
            #save min value in v in each interation
            v = min(v, s_value)
            
            #v is returned as the value if its smaller than alpha.
            if v < alpha:
                return v
            #beta is calculated as the minimum of beta and value by far.
            beta = min(beta, v)
        #v is returned as the value.
        return v
    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #call alpha_beta on the state.
        return self.alpha_beta(gameState, 0, 0, float("-inf"),float("inf"))
        #util.raiseNotDefined()
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, agentIndex = 0, depth = 0):
        #keep agentIndex(incremented each time) in the acceptable range
        #which is the number of agents (pacman and ghosts)
        #if index is larger than the range, mod is calculated for index.
        if agentIndex >= gameState.getNumAgents():
            agentIndex %= gameState.getNumAgents()
            
        #check if the state is terminal, meaning that pacman either won, lost or depth is zero
        if gameState.isWin() or gameState.isLose() or (agentIndex == 0 and (depth == self.depth)):
            return self.evaluationFunction(gameState)

        #call max_value if the next agent is max (pacman), pacman is the maximizer.
        #depth is incremented in each ply (once all agents take their actions)
        if agentIndex == 0 and depth < self.depth:
            depth += 1
            return self.max_value(gameState, agentIndex, depth)
        #cal min_value if the next agent is min (ghost), ghost is the minimizer.
        else:
            return self.exp_value(gameState, agentIndex, depth)
    
    #this is the same max_value implemented earlier for minimax.
    def max_value(self, gameState, agentIndex, depth):
        #saves a value
        v = float('-inf')
        #saves the last action taken
        final_action = "null"
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agentIndex)
        
        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call expectimax function and we pass successor as the first argument, the result is saved in s_value.
       
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agentIndex, action)
            #call minimax on successor
            s_value = self.expectimax(successor, agentIndex + 1, depth)
            
            #we save the maximum value in v in each interation
            v = max(v, s_value)
            
            #we save the action that will eventually be taken to return it later on.
            if v == s_value:
                final_action = action
                
        #we return final action in our penultimate depth.
        if depth == 1:
            return final_action
        
        #v is returned as the value.
        return v
    
    #this function calculates average score.
    def exp_value(self, gameState, agent, depth):
        #saves a value
        v = 0
        
        #saves all legal actions for our state
        legalActions = gameState.getLegalActions(agent)
        
        #saves the probability of each action, which is 1 in the number of all possible actions.
        p = 1.0/len(gameState.getLegalActions(agent))
        
        #in this loop, we iterate through all legal actions and get the successors. 
        #we then call expectimax function and we pass successor as the first argument, the result is saved in s_value.
        
        for action in legalActions:
            #get successor
            successor = gameState.generateSuccessor(agent, action)
            #call minimax on successor
            s_value = self.expectimax(successor, agent + 1, depth)
            
            #save the value as sum of values multiplied by their probability.
            v += p * s_value
        
        #v is returned as the value.
        return v
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #call expectimax on the state.
        return self.expectimax(gameState,0,0)
        #util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
        curScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        we need to consider distnace from closest food, closest ghost and closest 
        capsules in addition to the number of ghosts. we also need to keep scared times for
        the ghosts in mind. A cobmination of all of these will improve our evaluation function.
        
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFoodList = currentGameState.getFood().asList()
    curCapsules = currentGameState.getCapsules()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    
    #keeps the number of scared ghosts
    scaredNum = 0
    
    #keeps distance from a ghost
    ghostDist = float('inf')
    
    #keeps distance from a capsule
    capDist = float('inf')
    
    #keeps distance from a food
    foodDist = float('inf')

    for ghostState in curGhostStates:
        #based on the discription given, if scaredTimer for a ghost is more than zero,
        #it means that the ghost is still scared, so we increment scaredNum by one.
        if ghostState.scaredTimer > 0:
            scaredNum += 1
       
        #if we don't end up in the ghost's position, we try to find the closest ghost to us.
        #in order to do that, we calculate our distance with the ghost and save the minimum one for later.      
        if curPos != ghostState.getPosition():
            tmpDist = manhattanDistance(ghostState.getPosition(), curPos)
            ghostDist = min(ghostDist, tmpDist)
        #if our current position and the ghost's position are the same, we're dead! in this case, -inf is returned.
        else:
            return float('-inf')
            
    
    #we need to find the closest food (aka minimum distance from a food),so we iterate through current capsules
    #and save the minimum distance.
    for food in curFoodList:
        tmpDist = manhattanDistance(food, curPos)
        foodDist = min(foodDist, tmpDist)
        
    #similar to foods, we need to find the closest capsule,we find and save the minimum distance.
    for capsuleState in curCapsules:
        tmpDist = manhattanDistance(capsuleState, curPos)
        capDist = min(capDist, tmpDist)
    
    #the less food we have left, the higher our score should be.
    foods = 1/(1 + foodDist)
    
    #the less distance we have, the higher our score should be.
    #since minimum distance is calculated so far, we inverse it.
    capDist = 1/(1 + capDist)
    
    #the less the number of our ghosts and the more the distance from the closest one is,
    #the higher our score should be, so we reverese the relation between them.
    ghostDist = 1/(1 + ((len(curGhostStates))/ghostDist))

    
    #add all the numbers.
    return foods + scaredNum + ghostDist + capDist + currentGameState.getScore()

    #util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
