import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")
    
def trainmodel():

    model = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(in_features=8*8*15, out_features=1))

    # ... and if you do, this initialization might not be relevant any more ...
    model[1].weight.data = initialize_weights()
    model[1].bias.data = torch.zeros(1)
    
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.00001)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(20):
        for x,y in trainloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ... after which, you should save it as "model_ckpt.pkl":
    torch.save(model, 'model_ckpt.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
    