import pyro
from pyro import poutine
import pyro.optim as pyroopt
# from pyro.nn import PyroModule
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.contrib.autoguide import AutoLaplaceApproximation
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, Predictive
from pyro.distributions import OneHotCategorical, Normal, Categorical, Uniform, Delta

from networks.torchvision.NN import *
from utils.savedir import *

DEBUG=False

class torchvisionBNN(torchvisionNN):

    def __init__(self, model_name, dataset_name, inference):
        super(torchvisionBNN, self).__init__(model_name, dataset_name)

        self.inference = inference
        self.name = "finetuned_"+str(model_name)+"_"+str(inference)+"_"+str(dataset_name)

    def to(self, device):
        self.basenet = self.basenet.to(device)
        self.rednet = self.rednet.to(device)

        for k, v in pyro.get_param_store().items():
            pyro.get_param_store()[k] = v.to(device)

        if hasattr(self, "delta_guide"):
            self.delta_guide = self.delta_guide.to(device)

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):

        model_ft, input_size = super(torchvisionBNN, self).initialize_model(model_name, num_classes,
                                                     feature_extract, use_pretrained)

        self.rednet = nn.Sequential(*list(model_ft.children())[:-1])
        return model_ft, input_size

    def train(self, dataloaders, criterion, optimizer, device, num_iters=10, is_inception=False):

        self.to(device)

        network = self.basenet
        since = time.time()
        elbo = TraceMeanField_ELBO()

        if self.inference=="svi":
            guide = self.guide

        elif self.inference=="laplace":
            guide = AutoLaplaceApproximation(self.model)

        svi = SVI(self.model, guide, optimizer, loss=elbo)

        val_acc_history = []

        best_network_wts = copy.deepcopy(network.state_dict())
        best_acc = 0.0

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
                        
                        if self.inference=="laplace":
                            # loss += svi.step(x_data=inputs, y_data=labels)
                            self.delta_guide = guide.laplace_approximation(inputs, labels)
                            pyro.param("auto_loc", guide.get_posterior())

                            # print(pyro.get_param_store().get_all_param_names())

                        elif self.inference=="svi":
                            loss += svi.step(x_data=inputs, y_data=labels)

                        outputs = self.forward(inputs)
                        _, preds = torch.max(outputs, 1)

                        if DEBUG:
                            print("\n", pyro.get_param_store()["outw_mu"])    

                    running_loss += loss * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if DEBUG:
                    print(list(poutine.trace(self.guide).get_trace(inputs).nodes.keys()))

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_network_wts = copy.deepcopy(network.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        print("\nLearned variational params:\n")
        print(pyro.get_param_store().get_all_param_names())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.basenet.load_state_dict(best_network_wts)
        return self.basenet, val_acc_history

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

        net = self.basenet
        w, b, w_name, b_name = self._last_layer(net)

        if self.inference=="laplace":

            w.requires_grad=True
            b.requires_grad=True

            outw_prior = Normal(loc=torch.zeros_like(w), scale=torch.ones_like(w))
            outb_prior = Normal(loc=torch.zeros_like(b), scale=torch.ones_like(b))

            outw = pyro.sample(w_name, outw_prior)
            outb = pyro.sample(b_name, outb_prior)

            # print(outw.shape, outb.shape)

            with pyro.plate("data", len(x_data)):
                output = self.rednet(x_data).squeeze()
                yhat = torch.matmul(output, outw.t()) + outb 
                lhat = nnf.log_softmax(yhat, dim=-1)
                cond_model = pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
                return cond_model

        elif self.inference=="svi":

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

        else:
            raise AssertionError("Wrong inference method")


    def guide(self, x_data, y_data=None):
 
        if self.inference=="svi":

            w, b, w_name, b_name = self._last_layer(self.basenet)

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
            lifted_module = pyro.random_module("module", self.basenet , priors)()

            with pyro.plate("data", len(x_data)):
                logits = lifted_module(x_data)
                probs = nnf.softmax(logits, dim=-1)

            return probs

    def forward(self, inputs, n_samples=10, seeds=None, out_prob=False):
    
        if seeds:
            if len(seeds) != n_samples:
                raise ValueError("Number of seeds should match number of samples.")
        else:
            seeds = list(range(n_samples))

        if self.inference=="laplace":

            out_batch = self.rednet(inputs).squeeze()
            layer_size = out_batch.shape[1]

            preds = []  
            for seed in seeds:
                pyro.set_rng_seed(seed)
                posterior = pyro.sample("posterior", self.delta_guide)
                out_w = posterior['fc.weight']
                out_b = posterior['fc.bias']
                print(out_b)
                output_probs = torch.matmul(out_batch, out_w.t()) + out_b
                preds.append(output_probs)

            output_probs = torch.stack(preds)

            # posterior = self.delta_guide

            # out_w = posterior[:self.num_classes*layer_size]
            # out_w = out_w.reshape(self.num_classes, layer_size)
            # out_b = posterior[self.num_classes*layer_size:]

            # # print(out_batch, out_w, out_b)
            # output_probs = torch.matmul(out_batch.squeeze(), out_w.t()) + out_b
            # output_probs = output_probs.unsqueeze(0)

        elif self.inference=="svi":

            preds = []  
            for seed in seeds:
                pyro.set_rng_seed(seed)
                guide_trace = poutine.trace(self.guide).get_trace(inputs)   
                preds.append(guide_trace.nodes['_RETURN']['value'])

            output_probs = torch.stack(preds)

            # print(output_probs.mean(0).sum(1))

        else:
            raise NotImplementedError
        
        return output_probs if out_prob else output_probs.mean(0)

    def save(self, num_iters):
        path=TESTS+self.name+"/"
        self.to("cpu")
        
        if self.inference=="svi":

            filename=self.name+"_iters="+str(num_iters)+"_weights.pt"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            param_store = pyro.get_param_store()
            print(f"\nlearned params = {param_store.get_all_param_names()}")
            param_store.save(path + filename)

        elif self.inference=="laplace":

            filename=self.name+"_iters="+str(num_iters)+"_delta_guide.pkl"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            save_to_pickle(self.delta_guide, path, filename)

        print("\nSaving: ", path + filename)

    def load(self, num_iters, device):
        path=TESTS+self.name+"/"

        if self.inference=="svi":
            filename=self.name+"_iters="+str(num_iters)+"_weights.pt"
            param_store = pyro.get_param_store()
            param_store.load(path + filename)
            for key, value in param_store.items():
                param_store.replace_param(key, value, value)

        elif self.inference=="laplace":

            filename=self.name+"_iters="+str(num_iters)+"_delta_guide.pkl"
            self.delta_guide = load_from_pickle(path+filename)

        self.to(device)

        print("\nLoading: ", path + filename)