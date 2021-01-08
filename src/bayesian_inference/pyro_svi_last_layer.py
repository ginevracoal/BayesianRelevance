import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
import pyro.optim as pyroopt
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform, Delta

DEBUG=False

def model(bayesian_network, x_data, y_data):

    net = bayesian_network.basenet
    w, b, w_name, b_name = bayesian_network._last_layer(net)

    for weights_name in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
        pyro.get_param_store()[weights_name].requires_grad=True

    outw_prior = Normal(loc=torch.zeros_like(w), scale=torch.ones_like(w))
    outb_prior = Normal(loc=torch.zeros_like(b), scale=torch.ones_like(b))

    priors = {w_name: outw_prior, b_name: outb_prior}
    lifted_module = pyro.random_module("module", net, priors)()

    with pyro.plate("data", len(x_data)):
        logits = lifted_module(x_data)
        lhat = nnf.log_softmax(logits, dim=-1)
        cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    
    return cond_model

def guide(bayesian_network, x_data, y_data=None):

    w, b, w_name, b_name = bayesian_network._last_layer(bayesian_network.basenet)

    outw_mu = torch.randn_like(w)
    outw_sigma = torch.randn_like(w)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

    outb_mu = torch.randn_like(b)
    outb_sigma = torch.randn_like(b)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

    priors = {w_name: outw_prior, b_name: outb_prior}
    lifted_module = pyro.random_module("module", bayesian_network.basenet, priors)()

    with pyro.plate("data", len(x_data)):
        logits = lifted_module(x_data)
        probs = nnf.softmax(logits, dim=-1)

    return probs

def train(bayesian_network, dataloaders, device, num_iters, is_inception=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = pyro.optim.Adam({"lr":0.001})
    bayesian_network.to(device)

    network = bayesian_network.basenet
    since = time.time()
    elbo = TraceMeanField_ELBO()

    bayesian_network.model = model
    bayesian_network.guide = guide

    svi = SVI(bayesian_network.model, bayesian_network.guide, optimizer, loss=elbo)

    val_acc_history = []

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

                with torch.set_grad_enabled(phase == 'train'):
                    # Get model logits and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    loss += svi.step(bayesian_network, x_data=inputs, y_data=labels)
                    logits = bayesian_network.forward(inputs, n_samples=1)
                    _, preds = torch.max(logits, 1)

                    if DEBUG:
                        print(bayesian_network.basenet.state_dict()['conv1.weight'][0,0,:5])
                        print(pyro.get_param_store()["outw_mu"][:5])

                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            val_acc_history.append(epoch_acc)

        print()

    print("\nLearned variational params:\n")
    print(pyro.get_param_store().get_all_param_names())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_acc_history

def forward(bayesian_network, inputs, n_samples, sample_idxs=None):

    if sample_idxs:
        if len(sample_idxs) != n_samples:
            raise ValueError("Number of sample_idxs should match number of samples.")
    else:
        sample_idxs = list(range(n_samples))

    logits = []  
    for sample_idx in sample_idxs:
        pyro.set_rng_seed(sample_idx)
        guide_trace = poutine.trace(bayesian_network.guide).get_trace(bayesian_network, inputs)   
        logits.append(guide_trace.nodes['_RETURN']['value'])

    logits = torch.stack(logits)
    return logits

def save(bayesian_network, path, filename):
    param_store = pyro.get_param_store()
    print(f"\nlearned params = {param_store.get_all_param_names()}")
    param_store.save(path + filename + ".pt")

def load(bayesian_network, path, filename):
    param_store = pyro.get_param_store()
    param_store.load(path + filename + ".pt")
    for key, value in param_store.items():
        param_store.replace_param(key, value, value)
    
    bayesian_network.model = model
    bayesian_network.guide = guide

    print("\nLoading: ", path + filename)

def set_params_updates():
    for weights_name in pyro.get_param_store():
        if weights_name not in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
            pyro.get_param_store()[weights_name].requires_grad=False

def to(device):
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)
