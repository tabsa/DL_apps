#%% Basic example of NN with PyTorch package
# Basic buildings blocks of NN
# Make it as a class for other examples

#%% Import packages
import torch
import torch.nn as nn # nn.Module with basic NN structures

#%% NN class
class nn_model(nn.Module): # Borrows the properties of class nn.Module
    def __init__(self, no_in, no_out, dpout_prob=0.3):
        super(nn_model, self).__init__()
        self.pipe = nn.Sequential(# Most convinent structure
            nn.Linear(no_in, 5), # ff, l1 (in,5)
            nn.ReLU(), # act fun
            nn.Linear(5, 20), # ff, l2 (5,20)
            nn.ReLU(), # act fun
            nn.Linear(20, no_out), # ff, l3 (20,out)
            nn.Dropout(p=dpout_prob),
            nn.Softmax(dim=1)
        )
    def forward(self, x): # Function to call nn.Module.forward
        return self.pipe(x) # Return nn output

#%% main script
if __name__ == '__main__':
    print('Basic examples')
    # Basic applications
    nn_linear = nn.Linear(2, 5) # Linear implements a feed-forward layer (in=2 , out=5) NO intermediate layer
    x = torch.FloatTensor([1, 2]) # tensor with (1, 2) dimensions
    y = nn_linear(x) # NN output with X as input
    print(y) # Output with 5 scalars
    # NN sequential - convinent class that allows to build more complex layers
    nn_seq = nn.Sequential(
        nn.Linear(2,5), # l1(in=2, out=5)
        nn.ReLU(), # retified linear fun - activation function of the l1 with 5 output
        nn.Linear(5,10), # l2 (5,10)
        nn.Dropout(p=0.3),
        nn.Softmax(dim=1)
    )
    print(nn_seq)
    print('\n')
    # Use my class nn_model
    print('My class example')
    nn_example = nn_model (2, 3) # net(in=2, out=3)
    print(nn_example)
    x = torch.FloatTensor([[2, 3]]) # 2-D array
    y = nn_example(x) # output like y := fun(x)
    print(y)
    print("Cuda's availability is %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Data from cuda: %s" % y.to('cuda'))