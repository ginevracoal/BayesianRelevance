# import pyro
# from pyro import poutine
# import pyro.optim as pyroopt
# from pyro.contrib.autoguide import AutoLaplaceApproximation
# from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
# from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform, Delta

from networks.torchvision.baseNN import *
from utils.savedir import *
from utils.data import *

import bayesian_inference.pyro_svi_last_layer as pyro_svi
import bayesian_inference.pyro_laplace_last_layer as pyro_laplace

DEBUG=False

class torchvisionBNN(torchvisionNN):

    def __init__(self, model_name, dataset_name, inference):
        super(torchvisionBNN, self).__init__(model_name, dataset_name)

        self.inference = inference
        self.name = str(model_name)+"_redBNN_"+str(inference)+"_"+str(dataset_name)

    def to(self, device):
        """
        Sends network to device.
        """
        self.basenet = self.basenet.to(device)
        self.rednet = self.rednet.to(device)

        if self.inference=="svi":
            pyro_svi.to(device)

        if self.inference=="laplace":
            pyro_laplace.to(self, device)

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        """
        Loads pretrained models, sets parameters for training and specifies last layer weights 
        as the only ones that need to be inferred.
        """
        params_to_update = super(torchvisionBNN, self).initialize_model(model_name, num_classes,
                                                     feature_extract, use_pretrained)

        self.rednet = nn.Sequential(*list(self.basenet.children())[:-1])
        return params_to_update

    def set_params_updates(self, model, feature_extract):
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.

        if self.inference == "svi":
            pyro_svi.set_params_updates()

        params_to_update = model.parameters()
        print("\nParams to learn:")

        count = 0
        params_to_update = []

        for name,param in model.named_parameters():
            if param.requires_grad == True:
                if feature_extract:
                    params_to_update.append(param)
                print("\t", name)
                count += param.numel()

        print("Total n. of params =", count)

        return params_to_update

    def train(self, dataloaders, criterion, optimizer, device, num_iters=10, is_inception=False):
        """
        dataloaders: dictionary containing 'train', 'test' and 'val' dataloaders
        criterion: loss function
        optimizer: SGD optimizer
        device: "cpu" or "cuda" device 
        num_iters: number of training iterations
        is_inception: flag for Inception v3 model
        """
        if self.inference == "svi":
            pyro_svi.train(self, dataloaders, criterion, optimizer, device, num_iters, is_inception)

        elif self.inference == "laplace":
            pyro_laplace.train(self, dataloaders, criterion, optimizer, device, num_iters, is_inception)

        else:
            raise NotImplementedError

    def _last_layer(self, net):

        if self.model_name == "resnet":
            w, b = net.fc.weight, net.fc.bias
            w_name, b_name = 'fc.weight', 'fc.bias'

        elif self.model_name == "alexnet":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        elif self.model_name == "vgg":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        return w, b, w_name, b_name

    def model(self, x_data, y_data):


        if self.inference=="laplace":

            return pyro_laplace.model(self, x_data, y_data)

        elif self.inference=="svi":

            return pyro_svi.model(self, x_data, y_data)

        else:
            raise AssertionError("Wrong inference method")

    def forward(self, inputs, n_samples=10, seeds=None, out_prob=False):
    
        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")
        else:
            seeds = list(range(n_samples))

        if self.inference=="laplace":
            output_probs = pyro_laplace.forward(self, inputs, n_samples, seeds)

        elif self.inference=="svi":
            output_probs = pyro_svi.forward(self, inputs, n_samples, seeds)

        else:
            raise NotImplementedError
        
        return output_probs if out_prob else output_probs.mean(0)

    def save(self, savedir, num_iters):
        path=TESTS+savedir+"/"
        self.to("cpu")

        filename=self.name+"_iters="+str(num_iters)+"_weights.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.inference=="svi":
            pyro_svi.save(path, filename)

        elif self.inference=="laplace":
            pyro_laplace.save(path, filename)

        print("\nSaving: ", path + filename)

    def load(self, savedir, num_iters, device):
        path=TESTS+savedir+"/"
        filename=self.name+"_iters="+str(num_iters)+"_weights.pt"

        if self.inference=="svi":
            pyro_svi.load(path, filename)

        elif self.inference=="laplace":
            pyro_laplace.load(path, filename)

        self.to(device)
