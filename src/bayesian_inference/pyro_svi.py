import time

import torch
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
import pyro.optim as pyroopt
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform, Delta

DEBUG=False

def model(network, x_data, y_data):

    priors = {}
    for key, value in network.state_dict().items():
        loc = torch.zeros_like(value)
        scale = torch.ones_like(value)
        prior = Normal(loc=loc, scale=scale)
        priors.update({str(key):prior})

    lifted_module = pyro.random_module("module", network, priors)()

    with pyro.plate("data", len(x_data)):
        logits = lifted_module(x_data)
        lhat = nnf.log_softmax(logits, dim=-1)
        obs = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


def guide(network, x_data, y_data=None):

    dists = {}
    for key, value in network.state_dict().items():
        loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
        scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
        distr = Normal(loc=loc, scale=softplus(scale))
        dists.update({str(key):distr})

    lifted_module = pyro.random_module("module", network, dists)()
    
    with pyro.plate("data", len(x_data)):
        logits = lifted_module(x_data)
        preds = nnf.softmax(logits, dim=-1)

    return preds

def train(network, dataloaders, criterion, device, num_iters, is_inception=False):

    network.to(device)
    network.model = model
    network.guide = guide

    elbo = TraceMeanField_ELBO()
    optimizer = pyro.optim.Adam({"lr":0.001})
    svi = SVI(network.model, network.guide, optimizer, loss=elbo)

    val_acc_history = []

    since = time.time()
    for epoch in range(num_iters):

        loss=0.0

        print('Epoch {}/{}'.format(epoch, num_iters - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                network.train()  # Set model to training mode
            else:
                network.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs, labels  = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase=='train'):

                    loss = svi.step(network, x_data=inputs, y_data=labels)
                    outputs = network.forward(inputs)
                    _, preds = torch.max(outputs, 1)

                    if DEBUG:
                        print(network.basenet.state_dict()['conv1.weight'][0,0,:5])
                        print(pyro.get_param_store()["outw_mu"][:5])

                minibatch_size = inputs.size(0)
                running_loss += loss * minibatch_size
                running_corrects += torch.sum(preds == labels.data)

            num_minibatches =  len(dataloaders[phase].dataset)
            epoch_loss = running_loss / num_minibatches
            epoch_acc = running_corrects.double() / num_minibatches

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            val_acc_history.append(epoch_acc)

        print()

    print("\nLearned variational params:\n")
    print(pyro.get_param_store().get_all_param_names())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return network, val_acc_history

def forward(network, inputs, n_samples, sample_idxs=None):

    if sample_idxs:
        if len(sample_idxs) != n_samples:
            raise ValueError("Number of sample_idxs should match number of samples.")
    else:
        sample_idxs = list(range(n_samples))

    logits = []  
    for sample_idx in sample_idxs:
        pyro.set_rng_seed(sample_idx)
        guide_trace = poutine.trace(network.guide).get_trace(network, inputs)   
        logits.append(guide_trace.nodes['_RETURN']['value'])

    logits = torch.stack(logits)
    return logits

def save(path, filename):
    param_store = pyro.get_param_store()
    print(f"\nlearned params = {param_store.get_all_param_names()}")
    param_store.save(path + filename + ".pt")

def load(path, filename):
    param_store = pyro.get_param_store()
    param_store.load(path + filename + ".pt")
    for key, value in param_store.items():
        param_store.replace_param(key, value, value)
    
    print("\nLoading: ", path + filename)

def to(device):
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)
