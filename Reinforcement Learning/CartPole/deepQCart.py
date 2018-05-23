from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import gym
import math
import random
from collections import deque
import os.path

class DeepQCartSolver():

    def __init__(self, nEpisodes=2000, minScoreToWin=180, gamma=1.0, epsilon=1.0, minEpsilon=0.01, epsilonDecay=0.995,
                 alpha=0.01, alphaDecay=0.01, batchSize=64):

        self.env = gym.make('CartPole-v0')
        self.nEpisodes = nEpisodes
        self.minScoreToWin = minScoreToWin
        self.memory = deque(maxlen=50000)  # This dataset will feed our neural network with data
        self.gamma = gamma
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.epsilonDecay = epsilonDecay
        self.alpha = alpha
        self.alphaDecay = alphaDecay
        self.batchSize = batchSize

        self.nnModel = self.initNNModel()

    # layerNeurons -> input_nodes, hiddem_nodes, hidden_nodes, output_node
    # activationFunc -> hidden_nodes activation, hidden_nodes activation, output_activation
    def initNNModel(self, layerNeurons=[4, 24, 48, 2], activationFunc=['relu', 'relu', 'linear'], lossFunc='mse'):

        neuralNet = Sequential()

        for i in range(len(layerNeurons) - 1):

            if i == 0:

                # adds the input layer and the first hidden layer
                print(i)
                neuralNet.add(Dense(layerNeurons[i + 1], input_dim=layerNeurons[i], activation=activationFunc[i]))

            else:
                if i != (len(layerNeurons) - 2):

                    # adds all the other hidden layers
                    neuralNet.add(Dense(layerNeurons[i + 1], activation=activationFunc[i]))

                else:
                    # adds the last layer
                    neuralNet.add(Dense(layerNeurons[i+1], activation=activationFunc[i]))

        neuralNet.compile(loss=lossFunc, optimizer=Adam(lr=self.alpha, decay=self.alphaDecay))

        if(os.path.isfile("weights.txt")):
            neuralNet.load_weights("weights.txt")

        return neuralNet

    # transforms a python list into a numpy array so that keras can understand it
    def reshapeState(self, state):
        return np.reshape(state, [1, 4])

    # add a state to our state memory
    def rememberState(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # replay
    def updateNetwork(self, batchSize):

        # tabular version:
        # self.Q[state_old][action] = alpha * (reward + self.gamma * np.max(self.Q[state_new])

        # as we will use batch training, we must gather all the states that we want to pass to the neural net
        x_batch = []

        # we must also gather the labels of these states
        y_batch = []

        # populate the minibatch with a random sample of our past experiences
        minibatch = random.sample(list(self.memory), min(len(self.memory), batchSize))

        # we will update the q-values of each element of this sample
        for state, action, reward, next_state, done in minibatch:

            #print(state)

            # we retrieve the entry that we want to update: (predict() method returns a numpy array)
            target = self.nnModel.predict(self.reshapeState(state))

            # now we are going to update it

            if done:
                # in this case, there is no next_state, so we update the old_state value only with the reward
                target[0][action] = reward
            else:
                # we use the q-learning equation to update the old state's value
                target[0][action] = reward + self.gamma * np.max(self.nnModel.predict(self.reshapeState(next_state)))

            # collecting the sampled states into an array
            x_batch.append(state[0])

            # computing the labels of the states we are going to train the network on
            y_batch.append(target[0])

        # after we finish collecting and updating the sample, we must re-enter them in the network. We do so by training the network on this data.
        self.nnModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        # after updating our network, we should also update the epsilon
        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.epsilonDecay

    # returns the action we should take when we are in a given state
    def getAction(self, state, epsilon):

        # we test whether a random value is smaller than our threshold epsilon. if so, return a random action:
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            # we use our neural network to predict the value of the possible actions from this state and return the one with the highest value
            return np.argmax(self.nnModel.predict(state))

    def getEpsilon(self, t):

        return max(self.minEpsilon, min(1, (1.0 - (math.log10((t + 1) * self.epsilonDecay)))))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.nEpisodes):
            state = self.reshapeState(self.env.reset())
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.getAction(state, self.getEpsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                self.rememberState(state, action, reward, self.reshapeState(next_state), done)
                state = self.reshapeState(next_state)
                i += 1

            scores.append(i)
            avg_score = np.mean(scores)

            if avg_score > self.minScoreToWin:
                print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
                self.nnModel.save_weights("weights.txt",True)
                return e - 100

            if (e % 100 == 0):
                print('Episode {} - Average survival time over last 100 episodes was {} ticks'.format(e, avg_score))

            self.updateNetwork(self.batchSize)

            print("Avg score: {} / Last score: {} \nEpisode: {} Epsilon: {}".format(avg_score, i, e, self.epsilon))

        print('Algorithm was unable to solve after {} epsiodes.'.format(e))

        return e


if __name__ == '__main__':
    agent = DeepQCartSolver()
    agent.run()














