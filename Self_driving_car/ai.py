import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable


## Architecture


class Network(nn.Module):

    ##To create the network, we use linear function
    def  __init__(self,input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    ##To create acivation function, we use relu
    def forward(self,state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
## Experience replay
        
    
##Experience replay creates an array of memory from which our agent learns
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self,event):
        #event is a tuple of length 4 with last state, current state, last action and last reward
        self.memory.append(event)
        # to delete extra memory
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #zip(*) function changes a list from[(0,1,2),(3,4,5)] to [(0,3),(1,4),(2,5)].
        # It is required as we need to conver tensors to a pytorch variable
        samples = zip(*random.sample(self.memory, batch_size))
        
        #once we have the required format we can convert it to pytorch variable usin Variable
        return map(lambda x: Variable(torch.cat(x,0)),samples)
            
###Implementing deep q learning
        
class Dqn():
    
    def __init__(self,input_size, nb_action, gamma):
        self.gamma = gamma
        
        #To create a reward window
        self.reward_window = []
        
        #To initialize model
        self.model = Network(input_size, nb_action)
        
        #to create memory
        self.memory = ReplayMemory(100000)
        
        #To perform stochastic gradient descent
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001)
        
        #to get last state
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        # to get last action
        self.last_action =0
        
        #to get last reward
        self.last_reward = 0
        
    def select_action(self,state):
        
        #Calculating the probabilities
        probs = F.softmax(self.model(Variable(state,volatile = True))*100)
        #Here 7 is temp coeff to increase surity and efficiency
        
        #to select a random prob
        action = probs.multinomial(len(probs))
        
        #as mutinomial function returns in diff format
        return action.data[0,0]
        
    def learn(self,batch_state,batch_next_state,batch_reward, batch_action):
        
        #Gather chooses the action which was choosen and gets its q_value
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1)
        
        #Target
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
                                                                                                                        
        
        # loss function
        td_loss = F.smooth_l1_loss(outputs, target)
        
        #Reset the optimizer
        self.optimizer.zero_grad()
                
        #Backpropagate
        td_loss.backward(retain_graph = True)
        
        #Update the weights
        self.optimizer.step()
        
    def update(self,reward,new_signal):
        
        # Get a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        # Update the memory
        self.memory.push((self.last_state,new_state,torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward])))
        
        # Get the next action by giving states and signals
        action = self.select_action(new_state)
        
        #If memory>100, take a sample of 100, and return both states,reward and action as a batch
        if len(self.memory.memory)>100:
            batch_state,batch_next_state,batch_action,batch_reward = self.memory.sample(100) 
        
        
            #Learn from samples
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window)>1000:
            del self.reward_window[0]
            
            
        return action
    
    def score(self):
        if len(self.reward_window)<1:
            return 0
        else:
            return sum(self.reward_window)/len(self.reward_window)
        
        
        
    def save(self):
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict,
                    },'last_brain.pth')
        
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading the checkpoint......")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Brain loaded")
        else:
            print("No checkpoint found..........")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        