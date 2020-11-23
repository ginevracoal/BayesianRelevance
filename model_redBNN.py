"""
Neural network with bayesian last layer.
"""

import os
import argparse
import numpy as np
from utils_data import *
from model_baseNN import baseNN

import torch
from torch import nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro import poutine
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import OneHotCategorical, Normal, Categorical


DEBUG=False

saved_redBNNs = {"model_0":{"dataset":"mnist", "inference":"svi", "hidden_size":512, 
                 			"baseNN_inputs":60000, "baseNN_epochs":10, "baseNN_lr":0.001,
                 			"BNN_inputs":60000, "BNN_epochs":5, "BNN_lr":0.01, 
                 			"activation":"leaky", "architecture":"conv"},
     			 "model_1":{"dataset":"fashion_mnist", "inference":"svi", "hidden_size":1024, 
                 			"baseNN_inputs":60000, "baseNN_epochs":15, "baseNN_lr":0.001,
                 			"BNN_inputs":60000, "BNN_epochs":5, "BNN_lr":0.01, 
                 			"activation":"leaky", "architecture":"conv"}}


def get_hyperparams(model_dict):

	if model_dict["inference"] == "svi":
		return {"epochs":model_dict["BNN_epochs"], "lr":model_dict["BNN_lr"]}

	elif model_dict["inference"] == "hmc":
		return {"hmc_samples":model_dict["hmc_samples"], "warmup":model_dict["warmup"]}


class redBNN(nn.Module):

	def __init__(self, dataset_name, inference, hyperparams, base_net):
		super(redBNN, self).__init__()
		self.dataset_name = dataset_name
		self.hidden_size=base_net.hidden_size
		self.architecture=base_net.architecture
		self.activation=base_net.activation
		self.inference = inference
		self.base_net = base_net
		self.hyperparams = hyperparams
		self.name = self.set_name()

	def set_name(self):

		name = str(self.base_net.dataset_name)+"_redBNN_hid="+str(self.hidden_size)+\
					"_arch="+str(self.architecture)+"_act="+str(self.activation)

		if self.inference == "svi":
			return name+"_ep="+str(self.hyperparams["epochs"])+"_lr="+\
			       str(self.hyperparams["lr"])+"_"+str(self.inference)

		elif self.inference == "hmc":
			return name+"_samp="+str(self.hyperparams["hmc_samples"])+\
			       "_warm="+str(self.hyperparams["warmup"])+"_"+str(self.inference)

	def model(self, x_data, y_data):
		net = self.base_net

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
		net = self.base_net 

		outw_mu = torch.randn_like(net.out.weight)
		outw_sigma = torch.randn_like(net.out.weight)
		outw_mu_param = pyro.param("outw_mu", outw_mu)
		outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
		outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)

		outb_mu = torch.randn_like(net.out.	bias)
		outb_sigma = torch.randn_like(net.out.bias)
		outb_mu_param = pyro.param("outb_mu", outb_mu)
		outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
		outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)

		priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
		lifted_module = pyro.random_module("module", net, priors)()

		with pyro.plate("data", len(x_data)):
			logits = lifted_module(x_data)
			probs = nnf.softmax(logits, dim=-1)

		return probs
 
	def save(self, n_inputs):

		filepath, filename = (TESTS+self.name+"/", self.name+"_weights")
		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		if self.inference == "svi":
			self.base_net.to("cpu")
			self.to("cpu")
			param_store = pyro.get_param_store()
			print("\nSaving: ", filepath + filename +".pt")
			print(f"\nlearned params = {param_store.get_all_param_names()}")
			param_store.save(filepath + filename +".pt")

		elif self.inference == "hmc":
			self.base_net.to("cpu")
			self.to("cpu")
			save_to_pickle(data=self.posterior_samples, path=filepath, filename=filename+".pkl")

	def load(self, n_inputs, device, rel_path):

		filepath, filename = (rel_path+self.name+"/", self.name+"_weights")

		if self.inference == "svi":
			param_store = pyro.get_param_store()
			param_store.load(filepath + filename + ".pt")
			print("\nLoading ", filepath + filename + ".pt\n")

		elif self.inference == "hmc":
			posterior_samples = load_from_pickle(filepath + filename + ".pkl")
			self.posterior_samples = posterior_samples
			print("\nLoading ", filepath + filename + ".pkl\n")

		self.base_net.to(device)

	def forward(self, inputs, n_samples, seeds=None, training=False, out_prob=True, 
		        *args, **kwargs):

		if seeds:
			if len(seeds) != n_samples:
				raise ValueError("Number of seeds should match number of samples.")
		else:
			seeds = list(range(n_samples))

		if self.inference == "svi":

			if DEBUG:
				print("\nguide_trace =", 
					list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

			preds = []  

			if training:
				guide_trace = poutine.trace(self.guide).get_trace(inputs)   
				preds.append(guide_trace.nodes['_RETURN']['value'])

			else:
				for seed in seeds:
					pyro.set_rng_seed(seed)
					guide_trace = poutine.trace(self.guide).get_trace(inputs)   
					preds.append(guide_trace.nodes['_RETURN']['value'])

			if DEBUG:
				print("\nmodule$$$model.0.weight shoud be fixed:\n", 
					  guide_trace.nodes['module$$$model.0.weight']['value'][0,0,:3])
				print("\noutw_mu shoud be fixed:\n", guide_trace.nodes['outw_mu']['value'][:3])
				print("\nmodule$$$out.weight shoud change:\n", 
					  guide_trace.nodes['module$$$out.weight']['value'][0][:3]) 

		elif self.inference == "hmc":

			if DEBUG:
				print("\nself.base_net.state_dict() keys = ", self.base_net.state_dict().keys())

			preds = []
			n_samples = min(n_samples, len(self.posterior_samples["module$$$out.weight"]))
			for i in range(n_samples):

				state_dict = self.base_net.state_dict()
				out_w = self.posterior_samples["module$$$out.weight"][i]
				out_b = self.posterior_samples["module$$$out.bias"][i]
				state_dict.update({"out.weight":out_w, "out.bias":out_b})
				self.base_net.load_state_dict(state_dict)
				
				preds.append(self.base_net.forward(inputs))

				if DEBUG:
					print("\nl2.0.weight should be fixed:\n", 
						  self.base_net.state_dict()["l2.0.weight"][0,0,:3])
					print("\nout.weight should change:\n", self.base_net.state_dict()["out.weight"][0][:3])	
		
		output_probs = torch.stack(preds)
		return output_probs if out_prob else output_probs.mean(0)


	def _train_hmc(self, train_loader, device):
		print("\n == redBNN HMC training ==")

		num_samples, warmup_steps = (self.hyperparams["hmc_samples"], self.hyperparams["warmup"])
		# kernel = HMC(self.model, step_size=0.0855, num_steps=4)
		kernel = NUTS(self.model)
		batch_samples = int(num_samples*train_loader.batch_size/len(train_loader.dataset))+1
		print("\nSamples per batch =", batch_samples)
		hmc = MCMC(kernel=kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)

		start = time.time()

		out_weight, out_bias = ([],[])
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device).argmax(-1)
			hmc.run(x_batch, y_batch)
			out_weight.append(hmc.get_samples()["module$$$out.weight"])
			out_bias.append(hmc.get_samples()["module$$$out.bias"])

		execution_time(start=start, end=time.time())

		out_weight, out_bias = (torch.cat(out_weight), torch.cat(out_bias))
		self.posterior_samples = {"module$$$out.weight":out_weight, "module$$$out.bias":out_bias}

		self.save(n_inputs=len(train_loader.dataset), hyperparams=hyperparams)
		return self.posterior_samples

	def _train_svi(self, train_loader, device):
		print("\n == redBNN SVI training ==")

		epochs, lr = (self.hyperparams["epochs"], self.hyperparams["lr"])

		optimizer = pyro.optim.Adam({"lr":lr})
		elbo = TraceMeanField_ELBO()
		svi = SVI(self.model, self.guide, optimizer, loss=elbo)

		start = time.time()
		for epoch in range(epochs):
			loss = 0.0
			correct_predictions = 0.0

			for x_batch, y_batch in train_loader:

				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				loss += svi.step(x_data=x_batch, y_data=y_batch)

				probs = self.forward(x_batch, n_samples=1, training=True).to(device).mean(0)
				predictions = probs.argmax(-1)
				correct_predictions += (predictions == y_batch).sum()
			
				if probs.mean(0).sum().abs() < 0.9:
					raise ValueError("Error in softmax probs")

			total_loss = loss / len(train_loader.dataset)
			accuracy = 100 * correct_predictions / len(train_loader.dataset)
			print(f"\n[Epoch {epoch + 1}]\t loss: {total_loss:.8f} \t accuracy: {accuracy:.2f}", 
				  end="\t")

			if DEBUG:
				print("\nmodule$$$model.0.weight should be fixed:\n",
					  pyro.get_param_store()["module$$$model.0.weight"][0][0][:3])
				print("\noutw_mu should change:\n", pyro.get_param_store()["outw_mu"][:3])

		execution_time(start=start, end=time.time())
		hyperparams = {"epochs":epochs, "lr":lr}	
		self.save(n_inputs=len(train_loader.dataset))

	def train(self, train_loader, device):
		self.to(device)
		self.base_net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		if self.inference == "svi":
			self._train_svi(train_loader, device)

		elif self.inference == "hmc":
			self._train_hmc(train_loader, device)

	def evaluate(self, test_loader, device, n_samples):
		self.to(device)
		self.base_net.to(device)
		random.seed(0)
		pyro.set_rng_seed(0)

		with torch.no_grad():

			correct_predictions = 0.0

			for x_batch, y_batch in test_loader:

				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device).argmax(-1)
				predictions = self.forward(x_batch, n_samples=n_samples, out_prob=False).argmax(-1)
				correct_predictions += (predictions == y_batch).sum()

			accuracy = 100 * correct_predictions / len(test_loader.dataset)
			print("\nAccuracy: %.2f%%" % (accuracy))
			return accuracy


def main(args):

	m = saved_redBNNs["model_"+str(args.model_idx)]
	rel_path = DATA if args.load_dir=="DATA" else TESTS

	if args.device=="cuda":
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	### baseNN 
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=m["dataset"], batch_size=128, 
										 n_inputs=m["baseNN_inputs"], shuffle=True)

	nn = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
		        epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
		        activation=m["activation"], architecture=m["architecture"])
	nn.load(rel_path=rel_path, device=args.device)
	
	if args.test is True:
		nn.evaluate(test_loader=test_loader, device=args.device)

	### redBNN
	train_loader, test_loader, inp_shape, out_size = \
							data_loaders(dataset_name=m["dataset"], batch_size=128, 
										 n_inputs=m["BNN_inputs"], shuffle=True)
	hyp = get_hyperparams(m)
	bnn = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=nn, hyperparams=hyp)

	if args.train is True:
		bnn.train(train_loader=train_loader, device=args.device)
	else:
		bnn.load(n_inputs=m["BNN_inputs"], device=args.device, rel_path=rel_path)
	
	if args.test is True:
		bnn.evaluate(test_loader=test_loader, device=args.device, n_samples=100)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_BNNs dict")
    parser.add_argument("--train", default=True, type=eval, help="if True train else load")
    parser.add_argument("--test", default=True, type=eval, help="test set evaluation")
    parser.add_argument("--load_dir", default="DATA", type=str, help="DATA, TESTS")
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")	
    main(args=parser.parse_args())