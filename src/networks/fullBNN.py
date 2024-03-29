"""
Bayesian Neural Network model
"""

import argparse
import copy
import keras
import numpy as np
import os
import pandas as pd 

from collections import OrderedDict

import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as nnf
import torch.optim as torchopt

from torch import nn
softplus = torch.nn.Softplus()

import pyro
import pyro.optim as pyroopt

from pyro import poutine
from pyro.distributions import Categorical, Normal, OneHotCategorical, Uniform
from pyro.infer import Predictive, SVI, TraceMeanField_ELBO, Trace_ELBO
from pyro.infer.mcmc import HMC, MCMC, NUTS
from pyro.nn import PyroModule

from networks.baseNN import baseNN
from utils.data import *
from utils.model_settings import fullBNN_settings
from utils.savedir import *


DEBUG=False

class BNN(PyroModule):

    def __init__(self, dataset_name, hidden_size, activation, architecture, inference, 
                 epochs, lr, hmc_samples, warmup, input_shape, output_size):
        super(BNN, self).__init__()
        self.dataset_name = dataset_name
        self.inference = inference
        self.architecture = architecture
        self.epochs = epochs
        self.lr = lr
        self.hmc_samples = 20 if DEBUG else hmc_samples
        self.warmup = 5 if DEBUG else warmup
        self.step_size = 0.5
        self.num_steps = 10
        self.basenet = baseNN(dataset_name=dataset_name, input_shape=input_shape, 
                              output_size=output_size, hidden_size=hidden_size, 
                              activation=activation, architecture=architecture, 
                              epochs=epochs, lr=lr)

        self.name = self.get_name()
        self.n_layers = self.basenet.n_layers

    def get_name(self, n_inputs=None):
        
        name = str(self.dataset_name)+"_fullBNN_"+str(self.inference)+"_hid="+\
               str(self.basenet.hidden_size)+"_act="+str(self.basenet.activation)+\
               "_arch="+str(self.basenet.architecture)

        if n_inputs:
            name = name+"_inp="+str(n_inputs)

        if self.inference == "svi":
            return name+"_ep="+str(self.epochs)+"_lr="+str(self.lr)
        elif self.inference == "hmc":
            return name+"_samp="+str(self.hmc_samples)+"_warm="+str(self.warmup)+\
                   "_stepsize="+str(self.step_size)+"_numsteps="+str(self.num_steps)

    def model(self, x_data, y_data):

        priors = {}
        for key, value in self.basenet.state_dict().items():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

        lifted_module = pyro.random_module("module", self.basenet, priors)()

        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)
            logits = nnf.log_softmax(logits, dim=-1)
            obs = pyro.sample("obs", Categorical(logits=logits), obs=y_data)

    def guide(self, x_data, y_data=None):

        dists = {}
        for key, value in self.basenet.state_dict().items():
            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value, dtype=torch.float32)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value, dtype=torch.float32))
            distr = Normal(loc=loc, scale=softplus(scale))
            dists.update({str(key):distr})

        lifted_module = pyro.random_module("module", self.basenet, dists)()
        
        with pyro.plate("data", len(x_data)):
            logits = lifted_module(x_data)

        return logits 

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
            for idx in range(self.hmc_samples):
                net_copy = copy.deepcopy(self.basenet)
                fullpath=os.path.join(savedir, filename+"_"+str(idx)+".pt")    
                net_copy.load_state_dict(torch.load(fullpath))
                self.posterior_samples.append(net_copy)  

            if len(self.posterior_samples)!=self.hmc_samples:
                raise AttributeError("wrong number of posterior models")

        self.to(device)
        self.basenet.to(device)

    def _set_correct_layer_idx(self, layer_idx):
        return self.basenet._set_correct_layer_idx(layer_idx)

    def forward(self, inputs, n_samples=10, sample_idxs=None, training=False,
                expected_out=True, softmax=False, layer_idx=-1, *args, **kwargs):

        # change external attack libraries behavior #
        n_samples = self.n_samples if hasattr(self, "n_samples") else n_samples
        sample_idxs = self.sample_idxs if hasattr(self, "sample_idxs") else sample_idxs
        # avg_posterior = self.avg_posterior if hasattr(self, "avg_posterior") else avg_posterior
        layer_idx = self.layer_idx if hasattr(self, "layer_idx") else layer_idx
        #############################################

        if sample_idxs:
            if len(sample_idxs) != n_samples:
                raise ValueError("Number of sample_idxs should match number of samples.")
        else:
            sample_idxs = list(range(n_samples))

        if self.inference == "svi":

            preds = []  

            if training:

                guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                out = guide_trace.nodes['_RETURN']['value']
                preds.append(out)

            else:
                # for seed in sample_idxs:
                #     pyro.set_rng_seed(seed)

                #     guide_trace = poutine.trace(self.guide).get_trace(inputs)  

                #     sampled_dict = {}
                #     for key in self.basenet.state_dict().keys():
                #         dist = Normal(loc=guide_trace.nodes[key+"_loc"]["value"], 
                #                       scale=softplus(guide_trace.nodes[key+"_scale"]["value"]))
                #         weights = pyro.sample(key, dist)
                #         sampled_dict.update({key:weights}) 

                #     basenet_copy = copy.deepcopy(self.basenet)
                #     basenet_copy.load_state_dict(sampled_dict)
                #     out = basenet_copy.forward(inputs, layer_idx=layer_idx, *args, **kwargs)

                #     if softmax:
                #         out = nnf.softmax(out, dim=-1)
                #     preds.append(out)

                for seed in sample_idxs:
                    pyro.set_rng_seed(seed)
                    guide_trace = poutine.trace(self.guide).get_trace(inputs)  

                    weights = {}
                    for key, value in self.basenet.state_dict().items():

                        w = guide_trace.nodes[str(f"module$$${key}")]["value"]
                        weights.update({str(key):w})

                    basenet_copy = copy.deepcopy(self.basenet)
                    basenet_copy.load_state_dict(weights)

                    out = basenet_copy.forward(inputs, layer_idx=layer_idx, *args, **kwargs)
                    if softmax:
                        out = nnf.softmax(out, dim=-1)
                    preds.append(out)

        elif self.inference == "hmc":

            if n_samples>len(self.posterior_samples):
                raise ValueError("Too many samples. Max available samples =", len(self.posterior_samples))

            preds = []
            posterior_predictive = self.posterior_samples
            for seed in sample_idxs:
                net = posterior_predictive[seed]
                out = net.forward(inputs, layer_idx=layer_idx, *args, **kwargs)
                if softmax:
                    out = nnf.softmax(out, dim=-1)
                preds.append(out)
        
        preds = torch.stack(preds)
        return preds.mean(0) if expected_out else preds

    def _train_hmc(self, train_loader, n_samples, warmup, step_size, num_steps, savedir, device):
        print("\n == fullBNN HMC training ==")
        pyro.clear_param_store()

        if DEBUG:
            num_batches = 1
            batch_samples = 2
            warmup = 2
        else:
            num_batches = int(len(train_loader.dataset)/train_loader.batch_size)
            batch_samples = int(n_samples/num_batches)+1

        print("\nn_batches =", num_batches,"\tbatch_samples =", batch_samples)

        # kernel = HMC(self.model, step_size=step_size, num_steps=num_steps)
        kernel = NUTS(self.model, adapt_step_size=True)
        mcmc = MCMC(kernel=kernel, num_samples=batch_samples, warmup_steps=warmup, num_chains=1)

        self.posterior_samples=[]
        state_dict_keys = list(self.basenet.state_dict().keys())
        start = time.time()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).argmax(-1)
            mcmc_run = mcmc.run(x_batch, y_batch)

            posterior_samples = mcmc.get_samples(batch_samples)
            # print('module$$$model.1.weight:\n', posterior_samples['module$$$model.1.weight'][:,0,:5])

            for sample_idx in range(batch_samples):
                net_copy = copy.deepcopy(self.basenet)

                model_dict=OrderedDict({})
                for weight_idx, weights in enumerate(posterior_samples.values()):
                    model_dict.update({state_dict_keys[weight_idx]:weights[sample_idx]})
                
                net_copy.load_state_dict(model_dict)
                self.posterior_samples.append(net_copy)

        execution_time(start=start, end=time.time())     
        self.save(savedir)

    def _train_svi(self, train_loader, epochs, lr, savedir, device):
        print("\n == fullBNN SVI training ==")

        optimizer = pyro.optim.Adam({"lr":lr})
        elbo = Trace_ELBO()
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

                outputs = self.forward(x_batch, training=True).to(device)
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

        execution_time(start=start, end=time.time())
        self.save(savedir)

        plot_loss_accuracy(dict={'loss':loss_list, 'accuracy':accuracy_list},
                           path=os.path.join(savedir, self.name+"_training.png"))

    def train(self, train_loader, savedir, device):
        self.to(device)
        self.basenet.to(device)

        if self.inference == "svi":
            self._train_svi(train_loader, self.epochs, self.lr, savedir, device)

        elif self.inference == "hmc":
            self._train_hmc(train_loader, self.hmc_samples, self.warmup,
                            self.step_size, self.num_steps, savedir, device)

    def evaluate(self, test_loader, device, avg_posterior=False, n_samples=10, sample_idxs=None):
        self.to(device)
        self.basenet.to(device)

        with torch.no_grad():

            correct_predictions = 0.0
            for x_batch, y_batch in test_loader:

                x_batch = x_batch.to(device)
                outputs = self.forward(x_batch, n_samples=n_samples, sample_idxs=sample_idxs)
                predictions = outputs.to(device).argmax(-1)
                labels = y_batch.to(device).argmax(-1)
                correct_predictions += (predictions == labels).sum().item()

            accuracy = 100 * correct_predictions / len(test_loader.dataset)
            print("Accuracy: %.2f%%" % (accuracy))
            return accuracy
            
    def get_logits(self, *args, **kwargs):
        return self.forward(layer_idx=-1, *args, **kwargs)