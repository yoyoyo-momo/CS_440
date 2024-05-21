import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    block = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 5)
    )
    return block


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return torch.nn.CrossEntropyLoss()


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, 2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.hidden1 = torch.nn.Linear(800, 100)
        self.hidden2 = torch.nn.Linear(100, 31)
        self.output = torch.nn.Linear(31, 5)
        

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x = x.reshape(x.shape[0], 3, 31, 31)
        x_temp = self.conv1(x)
        x_temp = self.relu(x_temp)
        x_temp = self.pool(x_temp)
        
        x_temp = self.conv2(x_temp)
        x_temp = self.relu(x_temp)
        # x_temp = self.pool(x_temp)
        
        x_temp = torch.flatten(x_temp, 1)
        x_temp = self.relu(self.hidden1(x_temp))
        x_temp = self.relu(self.hidden2(x_temp))
        y = self.output(x_temp)
        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    loss_fn = create_loss_function()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(epochs):
        for features, labels in train_dataloader:
            y_pred = model(features)
            loss = loss_fn(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ################## Your Code Ends here ##################

    return model
