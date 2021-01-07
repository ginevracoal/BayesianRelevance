import time
from utils.data import save_to_pickle

import torch
import torch.nn as nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
import pyro.optim as pyroopt
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform, Delta
from pyro.contrib.autoguide import AutoLaplaceApproximation

DEBUG=False

def model(bayesian_network, x_data, y_data):

    net = bayesian_network.basenet
    w, b, w_name, b_name = bayesian_network._last_layer(net)

    w.requires_grad=True
    b.requires_grad=True

    outw_prior = Normal(loc=torch.zeros_like(w), scale=torch.ones_like(w))
    outb_prior = Normal(loc=torch.zeros_like(b), scale=torch.ones_like(b))

    outw = pyro.sample(w_name, outw_prior)
    outb = pyro.sample(b_name, outb_prior)

    # print(outw.shape, outb.shape)

    with pyro.plate("data", len(x_data)):
        output = bayesian_network.rednet(x_data).squeeze()
        yhat = torch.matmul(output, outw.t()) + outb 
        lhat = nnf.log_softmax(yhat, dim=-1)
        cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
    
    return cond_model

def train(bayesian_network, dataloaders, device, num_iters, is_inception=False):

    criterion = nn.CrossEntropyLoss()
    optimizer = pyro.optim.Adam({"lr":0.001})
    bayesian_network.to(device)

    network = bayesian_network.basenet
    bayesian_network.model = model
    since = time.time()

    guide = AutoLaplaceApproximation(bayesian_network.model)
    elbo = Trace_ELBO()
    svi = SVI(bayesian_network.model, guide, optimizer, loss=elbo)

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
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    loss += svi.step(bayesian_network, x_data=inputs, y_data=labels)
                    bayesian_network.delta_guide = guide.get_posterior()

                    outputs = bayesian_network.forward(inputs)
                    _, preds = torch.max(outputs, 1)

                    if DEBUG:
                        print(bayesian_network.basenet.state_dict()['conv1.weight'][0,0,:5])
                        print(pyro.sample("posterior", bayesian_network.delta_guide)[:5]) # = guide.loc

                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # if DEBUG:
            #     print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            val_acc_history.append(epoch_acc)

        print()

    bayesian_network.laplace_posterior = guide.laplace_approximation(bayesian_network, inputs, labels)

    print("\nLearned variational params:\n")
    print(pyro.get_param_store().get_all_param_names())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_acc_history

def forward(bayesian_network, inputs, n_samples, seeds=None):

    if seeds:
        if len(seeds) != n_samples:
            raise ValueError("Number of seeds should match number of samples.")
    else:
        seeds = list(range(n_samples))

    out_batch = bayesian_network.rednet(inputs).squeeze(3).squeeze(2)

    if hasattr(bayesian_network, 'laplace_posterior'):
        # after training use Laplace posterior approximation

        outputs = []  
        for seed in seeds:
            pyro.set_rng_seed(seed)
            posterior = pyro.sample("posterior", bayesian_network.laplace_posterior)
            out_w = posterior['fc.weight']
            out_b = posterior['fc.bias']
            output = torch.matmul(out_batch, out_w.t()) + out_b
            outputs.append(output)

        outputs = torch.stack(outputs)

    else:
        # during training use delta function at MAP estimate
        map_weights = pyro.sample("posterior", bayesian_network.delta_guide)

        layer_size = out_batch.shape[1]
        out_w = map_weights[:bayesian_network.num_classes*layer_size]
        out_w = out_w.reshape(bayesian_network.num_classes, layer_size)
        out_b = map_weights[bayesian_network.num_classes*layer_size:]

        outputs = torch.matmul(out_batch.squeeze(), out_w.t()) + out_b
        outputs = outputs.unsqueeze(0)

        if DEBUG:
            print("out_w[:5] =", out_w[:5])

    return outputs

def save(bayesian_network, path, filename):
    save_to_pickle(bayesian_network.laplace_posterior, path, filename+".pkl")

def load(bayesian_network, path, filename):
    bayesian_network.laplace_posterior = load_from_pickle(path+filename+".pkl")

def to(bayesian_network, device):
    if hasattr(bayesian_network, "laplace_posterior"):
        bayesian_network.laplace_posterior = bayesian_network.laplace_posterior.to(device)
