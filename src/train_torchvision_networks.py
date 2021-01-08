from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from utils.torchvision import *
from networks.torchvision.baseNN import *
from networks.torchvision.redBNN import *
from utils.data import load_from_pickle
from utils.seeding import *

import attacks.torchvision_gradient_based as grad_based
import attacks.deeprobust as deeprobust

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="resnet", help="resnet, alexnet, vgg")
parser.add_argument("--dataset", type=str, default="animals10", 
                    help="imagenette, imagewoof, animals10, hymenoptera")
parser.add_argument("--debug", type=eval, default="False")
parser.add_argument("--bayesian", type=eval, default="True")
parser.add_argument("--inference", type=str, default="svi", help="laplace, svi")
parser.add_argument("--samples", type=int, default=10, help="Number of posterior samples in the Bayesian case.")
parser.add_argument("--inputs", type=int, default=None, help="Number of input images. None loads all the available ones.")
parser.add_argument("--iters", type=int, default=5, help="Number of training iterations.")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--savedir", type=str, default=None)
args = parser.parse_args()

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


batch_size = 128

if args.debug:
    savedir="debug"
    n_inputs=10
    iters=2
    n_samples=2

else:
    n_inputs=args.inputs
    iters=args.iters
    n_samples=args.samples

    if args.savedir:
        savedir = args.savedir 

    else:
        if args.bayesian:
            savedir = args.model+"_redBNN_"+args.dataset+"_"+args.inference+"_iters="+str(iters)
        else:
            savedir = args.model+"_baseNN_"+args.dataset+"_iters="+str(iters)


if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_workers=0 if args.device=="cuda" else 4
device = torch.device(args.device)

dataloaders_dict, num_classes, _ = load_data(dataset_name=args.dataset, phases=['train','val'],
                                    batch_size=batch_size, n_inputs=n_inputs, num_workers=num_workers)

if args.bayesian:
    model = torchvisionBNN(model_name=args.model, dataset_name=args.dataset, inference=args.inference)
    model.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.to(device)

    model.train(dataloaders_dict, num_iters=iters, device=device)
    model.save(savedir, iters)

else:

    model = torchvisionNN(model_name=args.model, dataset_name=args.dataset)
    params_to_update = model.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.to(device)

    model.train(dataloaders_dict, params_to_update, num_iters=iters, device=device)
    model.save(savedir, iters)
