import torch
import torch.nn.functional as F
import torch.nn as nn

def metric_rmse(output,label):
    # this is our MSE loss function 
    return torch.sqrt(torch.mean(torch.square(torch.sub(output,label)))) #MSE as loss


# Write a method called `train_one_epoch` that runs one step using the optimizer.
def train_one_epoch(train_loader, model, device, optimizer, log_interval, epoch, batch_size,loss_func):
    model.train()
    losses = []
    counter = []
    metrics=[]
    
    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)

        # ------------------
        # forward pass
        output=model(img)
        # print(label)
        loss=  loss_func(output, label)# loss function goes here
        
        #zero all gradients
        optimizer.zero_grad()
        
        #do a backward pass
        loss.backward()
        
        #update weights
        optimizer.step()
        # print(loss)
        
        # ------------------
    
        # Record training loss every log_interval and keep counter of total training images seen
        if (i+1) % log_interval == 0:
            losses.append(loss.item())
            counter.append((i * batch_size) + img.size(0) + epoch * len(train_loader.dataset))
            metrics.append(metric_rmse(nn.Sigmoid()(output),label).item())


    return losses, counter, metrics

# Write a method called `test_one_epoch` that evalutes the trained model on the test dataset. Return the average test loss and the number of samples that the model predicts correctly.
def test_one_epoch(test_loader, model, device,loss_func):
    model.eval()
    test_loss = 0
    counter=0
    mse=0
    
    with torch.no_grad():
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)

            # ------------------
            output = model(img)
            counter+=1
            test_loss +=  loss_func(output,label).item()# loss function goes here
            mse+=metric_rmse(nn.Sigmoid()(output), label).item()
            # ------------------
            
    return test_loss/counter, mse/counter

