import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import argparse

from utils import load_MNIST, evaluate
from network import ConvNet
from optimizers import SGD, SVRG, SAGA, AUXILIARYOpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variance Reduction Optimizers.")
    parser.add_argument("--optimizer_type", type=str, default="SGD", choices=["SGD", "SVRG", "SAGA"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--update_freq", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--simple_model", action="store_true", default=False)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"
    train_data, test_data = load_MNIST()
    if args.optimizer_type == "SAGA":
        # FORCE TRAIN BATCH_SIZE = 1 FOR SAGA
        from utils import RandomSampler
        sampler = RandomSampler(train_data.data, train_data.targets)
        train_loader = DataLoader(range(train_data.data.size(0)), batch_size=1, shuffle=True, collate_fn=sampler.collate_fn, num_workers=2)
    else:
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=4096, shuffle=False, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    train_loss_all = []
    test_acc_all = []
    time_all = []

    if args.optimizer_type == "SGD":
        if args.simple_model:
            model = ConvNet([4, 8], [64, 32])
        else:
            model = ConvNet()
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        save_dir = os.path.join("./results", f"{args.optimizer_type}_{args.epochs}_{str(args.lr)}_{str(args.weight_decay)}_{args.simple_model}")
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        
        from utils import train_SGD
        
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_SGD(model, optimizer, train_loader, loss_fn, device)
            train_loss_all.append(train_loss)

            test_acc = evaluate(model, test_loader, device)
            test_acc_all.append(test_acc)
            t1 = time.time()
            time_all.append(t1-t0)

            print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, test acc: {test_acc:.4f}, time: {t1-t0:.2f} sec")
            if (epoch + 1) % args.save_every == 0:
                np.savez(os.path.join(save_dir, f"{epoch}.npz"), train_loss=np.array(train_loss_all), test_acc=np.array(test_acc_all), time=np.array(time_all))

    elif args.optimizer_type == "SVRG":
        if args.simple_model:
            model = ConvNet([4, 8], [64, 32])
        else:
            model = ConvNet()
        buddy_model = copy.deepcopy(model)
        model = model.to(device)
        buddy_model = buddy_model.to(device)
        optimizer = SVRG(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        aux_optimizer = AUXILIARYOpt(buddy_model.parameters())
        aux_loader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=4)
        total_train_num = train_data.data.size(0)
        save_dir = os.path.join("./results", f"{args.optimizer_type}_{args.epochs}_{str(args.lr)}_{str(args.weight_decay)}_{args.update_freq}_{args.simple_model}")
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

        from utils import train_SVRG

        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_SVRG(model, buddy_model, epoch, args.update_freq, optimizer, aux_optimizer, train_loader, aux_loader, loss_fn, device, total_train_num)
            train_loss_all.append(train_loss)

            test_acc = evaluate(model, test_loader, device)
            test_acc_all.append(test_acc)
            t1 = time.time()
            time_all.append(t1-t0)

            print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, test acc: {test_acc:.4f}, time: {t1-t0:.2f} sec")
            if (epoch + 1) % args.save_every == 0:
                np.savez(os.path.join(save_dir, f"{epoch}.npz"), train_loss=np.array(train_loss_all), test_acc=np.array(test_acc_all), time=np.array(time_all))

    else:
        total_train_num = train_data.data.size(0)
        if args.simple_model is False:
            raise ValueError("SAGA only supports simple model.")
        model = ConvNet([4, 8], [64, 32])
        buddy_models = [copy.deepcopy(model).to(device) for _ in range(total_train_num)]
        model = model.to(device)
        optimizer = SAGA(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        aux_optimizers = [AUXILIARYOpt(buddy_model.parameters()) for buddy_model in buddy_models]
        save_dir = os.path.join("./results", f"{args.optimizer_type}_{args.epochs}_{str(args.lr)}_{str(args.weight_decay)}")
        if os.path.exists(save_dir) is False:
            os.mkdir(save_dir)

        from utils import warm_up, train_SAGA
        warm_up_loader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=4)
        warm_up(model, warm_up_loader, optimizer, loss_fn, device, total_train_num)
        
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = 0.
            for id, b_x, b_y in train_loader:
                b_x, b_y = b_x.unsqueeze(0), b_y.unsqueeze(0)
                train_loss += train_SAGA(model, buddy_models[id], b_x, b_y, optimizer, aux_optimizers[id], loss_fn, device, total_train_num)
            train_loss /= total_train_num
            train_loss_all.append(train_loss)

            test_acc = evaluate(model, test_loader, device)
            test_acc_all.append(test_acc)
            t1 = time.time()
            time_all.append(t1-t0)

            print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, test acc: {test_acc:.4f}, time: {t1-t0:.2f} sec")
            if (epoch + 1) % args.save_every == 0:
                np.savez(os.path.join(save_dir, f"{epoch}.npz"), train_loss=np.array(train_loss_all), test_acc=np.array(test_acc_all), time=np.array(time_all))


