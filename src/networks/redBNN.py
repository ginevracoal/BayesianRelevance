"""
Neural network with bayesian last layer.
"""

import argparse
import os
import numpy as np
import pandas as pd 
import copy
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.distributions.constraints as constraints
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
import pyro.optim as pyroopt
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform
from pyro.nn import PyroModule

from utils.data import *
from utils.savedir import *
from networks.baseNN import baseNN


DEBUG=False


redBNN_settings = {"model_0":{"dataset":"mnist", "inference":"svi", "hidden_size":512, 
                            "n_inputs":60000, "epochs":5, "lr":0.01, 
                            "activation":"leaky", "architecture":"conv", "baseNN_idx":0},
                   "model_1":{"dataset":"fashion_mnist", "inference":"svi", "hidden_size":1024, 
                            "n_inputs":60000, "epochs":5, "lr":0.01, 
                            "activation":"leaky", "architecture":"fc2", "baseNN_idx":1},
                    }
                   
def get_hyperparams(model_dict):

    if model_dict["inference"] == "svi":
        return {"epochs":model_dict["epochs"], "lr":model_dict["lr"]}

    elif model_dict["inference"] == "hmc":
        return {"hmc_samples":model_dict["hmc_samples"], "warmup":model_dict["warmup"]}


class redBNN(PyroModule):

    def __init__(self, dataset_name, inference, hyperparams, base_net):
        super(redBNN, self).__init__()
        self.dataset_name = dataset_name
        self.hidden_size=base_net.hidden_size
        self.architecture=base_net.architecture
        self.activation=base_net.activation
        self.inference = inference
        self.basenet = base_net
        self.hyperparams = hyperparams
        self.name = self.set_name()

    def set_name(self):

        name = str(self.basenet.dataset_name)+"_redBNN_hid="+str(self.hidden_size)+\
                    "_arch="+str(self.architecture)+"_act="+str(self.activation)

        if self.inference == "svi":
            return name+"_ep="+str(self.hyperparams["epochs"])+"_lr="+\
                   str(self.hyperparams["lr"])+"_"+str(self.inference)

        elif self.inference == "hmc":
            return name+"_samp="+str(self.hyperparams["hmc_samples"])+\
                   "_warm="+str(self.hyperparams["warmup"])+"_"+str(self.inference)

    def model(self, x_data, y_data):

        net=self.basenet

        if self.inference == "svi":
            for weights_name in pyro.get_param_store():
                if weights_name not in ["outw_mu","outw_sigma","outb_mu","outb_sigma"]:
                    pyro.get_param_store()[weights_name].requires_grad=False

        outw_prior = Normal(loc=torch.zeros_like(net.out.weight), 
                            scale=torch.ones_like(net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(net.out.bias), 
                            scale=torch.ones_like(net.out.bias))
        
        priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
        lifted_module = pyro.random_module("module", net, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            lhat = nnf.log_softmax(logits, dim=-1)
            cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

    def guide(self, x_data, y_data=None):

        net=self.basenet

        outw_mu = torch.randn_like(net.out.weight)
        outw_sigma = torch.randn_like(net.out.weight)
        outw_mu_param = pyro.param("outw_mu", outw_mu)
        outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

        outb_mu = torch.randn_like(net.out. bias)
        outb_sigma = torch.randn_like(net.out.bias)
        outb_mu_param = pyro.param("outb_mu", outb_mu)
        outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

        priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
        lifted_module = pyro.random_module("module", net, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
        
        return nnf.softmax(logits, dim=-1)

    def save(self, savedir):
        filename=self.name+"_weights"

        if self.inference == "svi":
            os.makedirs(savedir, exist_ok=True)

            self.basenet.to("cpu")
            self.to("cpu")
            param_store = pyro.get_param_store()
            print(f"\nlearned params = {param_store.get_all_param_names()}")

            fullpath=os.path.join(savedir, filename+".pt")
            print("\nSaving: ", fullpath)
            param_store.save(fullpath)

        elif self.inference == "hmc":
            savedir=os.path.join(savedir, "weights")
            os.makedirs(savedir, exist_ok=True)  

            self.basenet.to("cpu")
            self.to("cpu")

            for idx, weights in enumerate(self.posterior_samples):
                fullpath=os.path.join(savedir, filename+"_"+str(idx)+".pt")    
                torch.save(weights.state_dict(), fullpath)

    def load(self, savedir, device):
        filename=self.name+"_weights"

        if self.inference == "svi":
            os.makedirs(savedir, exist_ok=True)

            param_store = pyro.get_param_store()
            param_store.load(os.path.join(savedir, filename + ".pt"))
            for key, value in param_store.items():
                param_store.replace_param(key, value.to(device), value)
            print("\nLoading ", os.path.join(savedir, filename + ".pt"))

        elif self.inference == "hmc":
            savedir=os.path.join(savedir, "weights")
            os.makedirs(savedir, exist_ok=True)  

            self.posterior_samples=[]
            for idx in range(self.n_samples):
                net_copy = copy.deepcopy(self.basenet)
                fullpath=os.path.join(savedir, filename+"_"+str(idx)+".pt")    
                net_copy.load_state_dict(torch.load(fullpath))
                self.posterior_samples.append(net_copy)  

            if len(self.posterior_samples)!=self.n_samples:
                raise AttributeError("wrong number of posterior models")

        self.to(device)
        self.basenet.to(device)
        self.device=device

    def forward(self, inputs, n_samples, sample_idxs=None, training=False, expected_out=True, 
                explain=False, rule=None):

        if sample_idxs:
            if len(sample_idxs) != n_samples:
                raise ValueError("Number of sample_idxs should match number of samples.")
        else:
            sample_idxs = list(range(n_samples))

        if self.inference == "svi":

            if DEBUG:
                print("\nguide_trace =", 
                    list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

            preds = []  

            if training:
                for _ in range(n_samples):
                    guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                    preds.append(guide_trace.nodes['_RETURN']['value'])

            else:
                for seed in sample_idxs:
                    pyro.set_rng_seed(seed)
                    guide_trace = poutine.trace(self.guide).get_trace(inputs)  

                    weights = {}
                    for key, value in self.basenet.state_dict().items():

                        w = guide_trace.nodes[str(f"module$$${key}")]["value"]
                        weights.update({str(key):w})

                    self.basenet.load_state_dict(weights)
                    self.basenet.to(self.device)
                    preds.append(self.basenet.forward(inputs, explain=explain, rule=rule))

            if DEBUG:
                print("\nmodule$$$model.0.weight shoud be fixed:\n", 
                      guide_trace.nodes['module$$$model.0.weight']['value'][0,0,:3])
                print("\noutw_mu shoud be fixed:\n", guide_trace.nodes['outw_mu']['value'][:3])
                print("\nmodule$$$out.weight shoud change:\n", 
                      guide_trace.nodes['module$$$out.weight']['value'][0][:3]) 

        elif self.inference == "hmc":

            if n_samples>len(self.posterior_samples):
                raise ValueError("Too many samples. Max available samples =", len(self.posterior_samples))

            if explain:
                preds = []
                # posterior_predictive = list(self.posterior_samples.values())
                posterior_predictive = self.posterior_samples
                for seed in sample_idxs:
                    net = posterior_predictive[seed]
                    preds.append(net.forward(inputs, explain=explain, rule=rule))

            else:
                preds = []
                # posterior_predictive = list(self.posterior_samples.values())
                posterior_predictive = self.posterior_samples
                for seed in sample_idxs:
                    net = posterior_predictive[seed]
                    preds.append(net.forward(inputs))
        
        logits = torch.stack(preds)
        return logits.mean(0) if expected_out else logits

    def _train_hmc(self, train_loader, savedir, device): # todo: check inferred weights 
        print("\n == redBNN HMC training ==")

        num_samples, warmup_steps = (self.hyperparams["hmc_samples"], self.hyperparams["warmup"])
        print("\nnum_chains =", len(train_loader), "\n")
        pyro.clear_param_store()
        batch_samples = 1 

        kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        mcmc = MCMC(kernel=kernel, num_samples=batch_samples, warmup_steps=warmup, num_chains=1)

        self.posterior_samples=[]
        state_dict_keys = ['out.weight','out.bias']
        start = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            labels = y_batch.to(device).argmax(-1)
            mcmc.run(x_batch, labels)

            posterior_sample = mcmc.get_samples(batch_samples)
            net_copy = copy.deepcopy(self.basenet)

            model_dict=OrderedDict({})
            for weights_key in state_dict_keys:
                model_dict.update({weights_key:posterior_sample['module$$$'+weights_key][0]})

            net_copy.load_state_dict(model_dict)
            self.posterior_samples.append(net_copy)

            if DEBUG:
                print(net_copy.state_dict()['out.weight'][0,:5])

        execution_time(start=start, end=time.time())

        out_weight, out_bias = (torch.cat(out_weight), torch.cat(out_bias))
        self.posterior_samples = {"module$$$out.weight":out_weight, "module$$$out.bias":out_bias}

        execution_time(start=start, end=time.time())     
        self.save(savedir)

    def _train_svi(self, train_loader, savedir, device):
        print("\n == redBNN SVI training ==")

        epochs, lr = (self.hyperparams["epochs"], self.hyperparams["lr"])

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)

        loss_list = []
        accuracy_list = []

        start = time.time()
        for epoch in range(epochs):
            loss = 0.0
            correct_predictions = 0.0

            for x_batch, y_batch in train_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss += svi.step(x_data=x_batch, y_data=y_batch.argmax(dim=-1))

                outputs = self.forward(x_batch, n_samples=10, training=True).to(device)
                predictions = outputs.argmax(-1)
                labels = y_batch.argmax(-1)
                correct_predictions += (predictions == labels).sum().item()
            
            if DEBUG:
                print("\n", pyro.get_param_store()["model.0.weight_loc"][0][:5])
                print("\n",predictions[:10],"\n", labels[:10])

            total_loss = loss / len(train_loader.dataset)
            accuracy = 100 * correct_predictions / len(train_loader.dataset)

            print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.2f} \t accuracy: {accuracy:.2f}", 
                  end="\t")

            loss_list.append(loss)
            accuracy_list.append(accuracy)


            if DEBUG:
                print("\nmodule$$$model.0.weight should be fixed:\n",
                      pyro.get_param_store()["module$$$model.0.weight"][0][0][:3])
                print("\noutw_mu should change:\n", pyro.get_param_store()["outw_mu"][:3])


        execution_time(start=start, end=time.time())
        self.save(savedir)

        plot_loss_accuracy(dict={'loss':loss_list, 'accuracy':accuracy_list},
                           path=os.path.join(savedir, self.name+"_training.png"))

    def train(self, train_loader, savedir, device):
        self.to(device)
        self.basenet.to(device)
        self.device=device

        if self.inference == "svi":
            self._train_svi(train_loader, savedir, device)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, savedir, device)

    def evaluate(self, test_loader, device, n_samples):
        self.to(device)
        self.basenet.to(device)

        with torch.no_grad():

            correct_predictions = 0.0
            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                outputs = self.forward(x_batch, n_samples=n_samples)
                predictions = outputs.to(device).argmax(-1)
                labels = y_batch.to(device).argmax(-1)
                correct_predictions += (predictions == labels).sum().item()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy

