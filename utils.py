import torch
from torchvision import transforms as T
from torchvision.datasets import FashionMNIST
from PIL import Image

class RandomSampler:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = T.ToTensor()

    def collate_fn(self, node_id):
        node_id = node_id[0]
        img = self.x[node_id]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)
        tgt = self.y[node_id]
        
        return node_id, img, tgt

def load_MNIST():
    train_data = FashionMNIST(root="./data", download=True, train=True, transform=T.ToTensor())
    test_data = FashionMNIST(root="./data", download=True, train=False, transform=T.ToTensor())
    
    return train_data, test_data

def train_SGD(model, optimizer, train_loader, loss_fn, device):
    model.train()

    train_loss = 0.
    train_num = 0
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    
    return train_loss / train_num

def train_SVRG(model, buddy_model, epoch, update_freq, optimizer, aux_optimizer, train_loader, aux_loader, loss_fn, device, total_train_num):
    model.train()

    total_loss = 0.
    if epoch % update_freq == 0:
        # To calculate u, we need to conduct full-batch forward and backward
        buddy_model.train()
        aux_optimizer.update_param_groups(optimizer.get_param_groups())
        aux_optimizer.zero_grad()
        for b_x, b_y in aux_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = buddy_model(b_x)
            loss = loss_fn(output, b_y) * b_x.size(0) / total_train_num  
            loss.backward()
        optimizer.update_u(aux_optimizer.get_param_groups())

    total_loss = 0.
    for b_x, b_y in train_loader:
        optimizer.zero_grad()
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        loss.backward()

        aux_optimizer.zero_grad()
        aux_output = buddy_model(b_x)
        aux_loss = loss_fn(aux_output, b_y)
        aux_loss.backward()
        optimizer.step(aux_optimizer.get_param_groups())

        total_loss += loss.item() * b_x.size(0)
    
    return total_loss / total_train_num

def warm_up(model, train_loader, optimizer, loss_fn, device, total_train_num):
    # calculate u_0
    model.train()
    optimizer.zero_grad()
    for b_x, b_y in train_loader:
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = model(b_x)
        loss += loss_fn(output, b_y) * b_x.size(0)
    loss /= total_train_num 
    loss.backward()
    optimizer.set_u() 

def train_SAGA(model, buddy_model, b_x, b_y, optimizer, aux_optimizer, loss_fn, device, total_train_num):
    model.train()
    buddy_model.train()

    optimizer.zero_grad()
    b_x = b_x.to(device)
    b_y = b_y.to(device)
    output = model(b_x)
    loss = loss_fn(output, b_y) 
    loss.backward() # calculate \Delta F_j(x_k)

    aux_optimizer.zero_grad()
    buddy_model = buddy_model.to(device)
    aux_output = buddy_model(b_x)
    aux_loss = loss_fn(aux_output, b_y)
    aux_loss.backward() # calculate \Delta F_j(\alpha_j^k)
    
    aux_optimizer.update_param_groups(optimizer.get_param_groups()) # update \alpha_j
    optimizer.step(aux_optimizer.get_param_groups()) # update x_k
    optimizer.update_u(aux_optimizer.get_param_groups(), total_train_num) # update u_{k+1} by u_k, \Delta F_j(x_k), Delta F_j(\alpha_j^k)

    return loss.item()

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()

    total_correct_num, test_num = 0, 0
    for b_x, b_y in test_loader:
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = model(b_x)
        total_correct_num += torch.sum(output.argmax(dim=1) == b_y.data).item()
        test_num += b_x.size(0)

    return total_correct_num / test_num



