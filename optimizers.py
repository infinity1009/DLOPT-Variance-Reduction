import copy
from torch.optim import Optimizer
 
class SGD(Optimizer):
    r"""
    Stachastic Gradient Descent
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float): regularization
    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
 
    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super(SGD, self).__init__(params, defaults)
 
    def step(self):
        """Performs a single optimization step.
        """
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                p.data.add_(d_p, alpha=-group["lr"])

class SVRG(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        self.__u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super(SVRG, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups
    
    def update_u(self, next_u):
        """
        update u with full-batch gradients
        """
        if self.__u is None:
            self.__u = copy.deepcopy(next_u)
        for prev_u_group, next_u_group in zip(self.__u, next_u):
            for prev_u_grad, next_u_grad in zip(prev_u_group["params"], next_u_group["params"]):
                prev_u_grad.grad = next_u_grad.grad.clone()

    def step(self, tilde_param_groups):
        """
        tilde_param_groups: parameter groups of $\tilde{X}$.
        """
        for param_group, tilde_group, u_group in zip(self.param_groups, tilde_param_groups, self.__u):
            weight_decay = param_group["weight_decay"]
            
            for param, tilde_param, u_param in zip(param_group["params"], tilde_group["params"], u_group["params"]):
                if param.grad is None:
                    continue 
                if tilde_param.grad is None:
                    continue
                d_total = param.grad - tilde_param.grad + u_param.grad 
                if weight_decay != 0:
                    d_total.add_(param.data, alpha=weight_decay)
                
                param.data.add_(d_total, alpha=-param_group["lr"])

class SAGA(Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        self.__u = None
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super(SAGA, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups
    
    def set_u(self):
        if self.__u is not None:
            raise ValueError(f"u is not None. Please use update_u method to refresh u.")
        new_u = self.get_param_groups()
        self.__u = copy.deepcopy(new_u)
        for prev_u_group, next_u_group in zip(self.__u, new_u):
            for prev_u_grad, next_u_grad in zip(prev_u_group["params"], next_u_group["params"]):
                prev_u_grad.grad = next_u_grad.grad.clone()
    
    def update_u(self, alpha_param_groups, total_train_num):
        for param_group, alpha_param_group, u_param_group in zip(self.param_groups, alpha_param_groups, self.__u):
            for param, alpha_param, u_param in zip(param_group["params"], alpha_param_group["params"], u_param_group["params"]):
                d_grad = param.grad.clone() - alpha_param.grad.clone()
                u_param.grad.add_(d_grad, alpha=1./total_train_num)

    def step(self, alpha_param_groups):
        for param_group, alpha_param_group, u_group in zip(self.param_groups, alpha_param_groups, self.__u):
            weight_decay = param_group["weight_decay"]
            
            for param, alpha_param, u_param in zip(param_group["params"], alpha_param_group["params"], u_group["params"]):
                if param.grad is None:
                    continue 
                if alpha_param.grad is None:
                    continue
                d_total = param.grad - alpha_param.grad + u_param.grad 
                if weight_decay != 0:
                    d_total.add_(param.data, alpha=weight_decay)
                
                param.data.add_(d_total, alpha=-param_group["lr"])

class AUXILIARYOpt(Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(AUXILIARYOpt, self).__init__(params, defaults)

    def get_param_groups(self):
        return self.param_groups
    
    def update_param_groups(self, new_param_groups):
        """
        update alpha_j^{k+1} = x_k
        """
        for old_param_group, new_param_group in zip(self.param_groups, new_param_groups):
            for old_param, new_param in zip(old_param_group["params"], new_param_group["params"]):
                old_param.data = new_param.data.clone()