import os
from queue import Queue

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from aihwkit.optim import AnalogSGD, AnalogAdam

'Available Datasets if more wanted Add here and in Frontend Selection'
DATASET_SPECS = {
    "MNIST": {"cls": datasets.MNIST, "channels": 1, "size": 28, "mean": (0.1307,), "std": (0.3081,)},
    "FashionMNIST": {
        "cls": datasets.FashionMNIST, "channels": 1, "size": 28, "mean": (0.2860,), "std": (0.3530,),
    },
    "CIFAR10": {
        "cls": datasets.CIFAR10, "channels": 3, "size": 32,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616),
    },
}

# Analog weights live on the RPU tile and are hardware-bounded by the device's conductance
# range (w_min/w_max) -- they physically cannot run away. The bias (digital by aihwkit's
# MappingParameter.digital_bias default) is a plain, unbounded torch.nn.Parameter with no such
# ceiling. Giving it its own, much smaller learning rate keeps it trainable without letting it
# diverge at learning rates that are otherwise fine for the self-limiting analog weights.
BIAS_LR_SCALE = 0.5

def train(model, device, train_loader, optimizer):
    '''takes the pyTorch/HWKIT model and performs a train, with loss metric cross entropy (may be extended later)

    Args:
        model (_type_): pyTorch/HWKIT model
        device (_type_): device
        train_loader (_type_): train loader
        optimizer (_type_): optimizer

    Returns:
        float: avg accuracy over Epoch, avg loss over Epoch
    '''
    model.train()
    total_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # Deliberately clips only the digital bias gradients. The analog tile updates bypass
        # .grad entirely -- AnalogSGD.step() replays the stored activation/error vectors as
        # pulsed rank-1 updates on the tile -- so this line cannot touch them. That is
        # physically consistent: a global gradient-norm clip across all crossbars would
        # require hardware that reads every gradient before pulsing, which defeats the
        # parallel analog update. The unbounded digital biases are the ones that need the
        # safety net here (together with BIAS_LR_SCALE).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(data)
        correct += output.argmax(dim=1).eq(target).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    return accuracy, avg_loss


def test(model, device, test_loader):
    '''takes the model performs test also uses cross_entropy

    Args:
        model (_type_): pyTorch/HWKIT model
        device (_type_): device
        test_loader (_type_): test loader

    Returns:
        float: accuracy during test epoch, loss during test epoch
    '''
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return accuracy, test_loss


class AnalogFramework:
    '''Device-agnostic-to-the-caller training harness. Takes a ready-built
    analog model + RPU_CONFIG from the caller and owns everything else:
    data loading, the train/test loop, and pushing live progress to a
    queue. Swapping network architecture never requires touching this
    class -- only the model passed in changes.'''

    def __init__(self,model,rpu_config,epochs=100,train_batch_size=32,test_batch_size=256,train_epoch_size=8000,test_epoch_size=1000,lr=0.01,optimizer="AnalogSGD",dataset="MNIST",device=None,queue=None,cancel_event=None,):
        '''
        Args:
            model (_type_): See Variable name
            rpu_config (_type_): See Variable name
            epochs (int, optional): See Variable name. Defaults to 100.
            train_batch_size (int, optional): See Variable name. Defaults to 32.
            test_batch_size (int, optional): See Variable name. Defaults to 256.
            train_epoch_size (int, optional): See Variable name. Defaults to 8000.
            test_epoch_size (int, optional): See Variable name. Defaults to 1000.
            lr (float, optional): Learning rate. Defaults to 0.01.
            optimizer (str, optional): See Variable name. Defaults to "AnalogSGD".
            dataset (str, optional): See Variable name. Defaults to "MNIST".
            device (_type_, optional): See Variable name. Defaults to None.
            queue (_type_, optional): Queue where changes are pushed. Defaults to None.
            cancel_event (threading.Event, optional): checked at the start of every epoch --
                if set, run() stops early (keeping every epoch already pushed to `queue`)
                instead of raising. Defaults to None (never cancels).
        '''
        self.model = model
        self.rpu_config = rpu_config
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_epoch_size = train_epoch_size
        self.test_epoch_size = test_epoch_size
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.optim = optimizer
        self.dataset = dataset
        self.queue = queue if queue is not None else Queue()
        self.cancel_event = cancel_event

    def run(self):
        '''running the simulation on cpu (currently only cpu supported)
        '''
        if self.device.type == "cpu":
            # os.cpu_count() reports the whole machine, not this process's share of it.
            # Under SLURM -- or simply several batch_runner workers on one box -- that
            # oversubscribes badly: every worker asks for every core. Prefer an explicit
            # override, then the CPUs actually scheduled to us (Linux only), and only
            # fall back to the machine count.
            override = os.environ.get("HWKIT_NUM_THREADS")
            if override:
                torch.set_num_threads(int(override))
            elif hasattr(os, "sched_getaffinity"):
                torch.set_num_threads(len(os.sched_getaffinity(0)))
            else:
                torch.set_num_threads(os.cpu_count())

        spec = DATASET_SPECS[self.dataset]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(spec["mean"], spec["std"]),
        ])

        dataset_cls = spec["cls"]
        full_train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
        test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
        test_dataset = Subset(test_dataset, range(self.test_epoch_size))
        train_data, train_targets = zip(*full_train_dataset)
        test_data, test_targets = zip(*test_dataset)
        full_train_data = torch.stack(train_data)
        full_train_targets = torch.tensor(train_targets)
        test_dataset = TensorDataset(torch.stack(test_data), torch.tensor(test_targets))

        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

        model = self.model.to(self.device)

        optimizer = self.getoptimizer(self.optim, model, self.lr)
        optimizer.regroup_param_groups(model)

        self.queue.put(('init', self.epochs, self.train_batch_size, self.lr))

        for epoch in range(1, self.epochs + 1):
            if self.cancel_event is not None and self.cancel_event.is_set():
                break

            epoch_indices = torch.randperm(len(full_train_data))[:self.train_epoch_size]
            train_dataset = TensorDataset(full_train_data[epoch_indices], full_train_targets[epoch_indices])
            train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

            train_acc, train_loss = train(model, self.device, train_loader, optimizer)
            test_acc, _ = test(model, self.device, test_loader)
            self.queue.put(('run', test_acc, train_acc, epoch, train_loss))

        self.queue.put(('stop',))
    
    def getoptimizer (self, optim, model, lr):
        '''Available Optimizers if more wanted Add here and in Frontend Selection

        Bias parameters (see network.py -- every AnalogLinear/AnalogConv2d layer has
        bias=True) are split into their own param group at BIAS_LR_SCALE * lr, since they
        are digital/unbounded unlike the analog weights -- see BIAS_LR_SCALE's comment.

        Args:
            optim (str): Optimizer (allowed)
            parameters (_type_): parameters
            lr (float): Learning rate
        Returns:
            optimizer
        '''
        bias_params = [p for name, p in model.named_parameters() if name.endswith("bias")]
        other_params = [p for name, p in model.named_parameters() if not name.endswith("bias")]
        param_groups = [{"params": other_params}, {"params": bias_params, "lr": lr * BIAS_LR_SCALE}]

        if optim == "AnalogSGD":
            optimizer = AnalogSGD(param_groups, lr)
        elif optim == "AnalogAdam":
            optimizer = AnalogAdam(param_groups, lr)
        else:
            raise ValueError("Optimizer not Allowed")
        return optimizer