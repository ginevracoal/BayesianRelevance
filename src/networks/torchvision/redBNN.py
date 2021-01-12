from utils.data import *
from utils.savedir import *
from networks.torchvision.baseNN import *

import bayesian_inference.last_layer.pyro_svi as pyro_svi
import bayesian_inference.last_layer.pyro_laplace as pyro_laplace
import bayesian_inference.last_layer.stochastic_gradient_langevin_dynamics as sgld

DEBUG=False

class redBNN(baseNN):

    def __init__(self, architecture, dataset_name, inference):
        super(redBNN, self).__init__(architecture, dataset_name)

        self.inference = inference
        self.name = str(architecture)+"_redBNN_"+str(inference)+"_"+str(dataset_name)

    def initialize_model(self, baseNN, architecture, num_classes, feature_extract, use_pretrained=True):
        """
        Load pretrained models, set parameters for training and specify last layer weights 
        as the only ones that need to be inferred.
        """
        self.basenet = baseNN.basenet 
        self.rednet = nn.Sequential(*list(self.basenet.children())[:-1])
        self.last_layer = nn.Sequential(list(self.basenet.children())[-1])

        self.num_classes = baseNN.num_classes
        self.input_size = baseNN.input_size

    def set_params_updates(self, model, feature_extract):
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.

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

        if self.inference == "svi":
            pyro_svi.set_params_updates()

        return params_to_update

    def train(self, dataloaders, device, num_iters=10, is_inception=False):
        """
        dataloaders: dictionary containing 'train', 'test' and 'val' dataloaders
        criterion: loss function
        optimizer: SGD optimizer
        device: "cpu" or "cuda" device 
        num_iters: number of training iterations
        is_inception: flag for Inception v3 model
        """
        if self.inference=="svi":
            pyro_svi.train(self, dataloaders, device, num_iters, is_inception)

        elif self.inference=="laplace":
            pyro_laplace.train(self, dataloaders, device, num_iters, is_inception)

        elif self.inference=="sgld":
            sgld.train(self, dataloaders, device, num_iters, is_inception)

        else:
            raise NotImplementedError

    def _last_layer(self, net):

        if self.architecture == "resnet":
            w, b = net.fc.weight, net.fc.bias
            w_name, b_name = 'fc.weight', 'fc.bias'

        elif self.architecture == "alexnet":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        elif self.architecture == "vgg":
            w, b = net.classifier[6].weight, net.classifier[6].bias
            w_name, b_name = 'classifier[6].weight', 'classifier[6].bias'

        return w, b, w_name, b_name

    def model(self, x_data, y_data):

        if self.inference=="svi":
            return pyro_svi.model(bayesian_network=self, x_data=x_data, y_data=y_data)

        elif self.inference=="laplace":
            return pyro_laplace.model(bayesian_network=self, x_data=x_data, y_data=y_data)

    def guide(self, x_data, y_data=None):

        if self.inference=="svi":
            return pyro_svi.guide(bayesian_network=self, x_data=x_data, y_data=y_data)


    def forward(self, inputs, n_samples=None, sample_idxs=None, expected_out=True):

        if hasattr(self, 'n_samples'):
            n_samples = self.n_samples
        else:
            if n_samples is None:
                raise ValueError("Set the number of posterior samples.")

        if self.inference=="svi":
            logits = pyro_svi.forward(self, inputs, n_samples, sample_idxs)

        elif self.inference=="laplace":
            logits = pyro_laplace.forward(self, inputs, n_samples, sample_idxs)
       
        elif self.inference=="sgld":
            logits = sgld.forward(self, inputs, n_samples, sample_idxs)
       
        else:
            raise NotImplementedError
        
        return logits.mean(0) if expected_out else logits

    def save(self, savedir):#, num_iters):
        path=TESTS+savedir+"/"
        self.to("cpu")

        filename=self.name+"_iters="+str(num_iters)+"_weights"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.inference=="svi":
            pyro_svi.save(self, path, filename)

        elif self.inference=="laplace":
            pyro_laplace.save(self, path, filename)

        elif self.inference=="sgld":
            path=path+filename+"/"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sgld.save(self, num_iters, path, filename)

    # def load(self, savedir, num_iters, device):
    def load(self, savedir, device):
        path=TESTS+savedir+"/"
        # filename=self.name+"_iters="+str(num_iters)+"_weights"
        filename=self.name+"_weights"

        if self.inference=="svi":
            pyro_svi.load(self, path, filename)

        elif self.inference=="laplace":
            pyro_laplace.load(self, path, filename)

        elif self.inference=="sgld":
            path=path+filename+"/"
            sgld.load(self, num_iters, path, filename)

        self.to(device)

    def to(self, device):
        """
        Send network to device.
        """
        self.basenet = self.basenet.to(device)
        self.rednet = self.rednet.to(device)
        self.last_layer = self.last_layer.to(device)

        if self.inference=="svi":
            pyro_svi.to(device)

        if self.inference=="laplace":
            pyro_laplace.to(self, device)