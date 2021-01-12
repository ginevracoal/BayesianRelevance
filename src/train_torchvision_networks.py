from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from utils.torchvision import *
from networks.torchvision.baseNN import *
from networks.torchvision.redBNN import *
from utils.data import load_from_pickle
from utils.seeding import *
from utils.savedir import _get_torchvision_savedir

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="animals10", 
                    help="imagenette, imagewoof, animals10, hymenoptera")
parser.add_argument("--inputs", type=int, default=None, help="Number of training images. None loads all the available ones.")
parser.add_argument("--model", type=str, default="redBNN", help="baseNN, redBNN")
parser.add_argument("--architecture", type=str, default="resnet", help="resnet, alexnet, vgg")
parser.add_argument("--iters", type=int, default=10, help="Number of training iterations.")
parser.add_argument("--inference", type=str, default="svi", help="(redBNN only) Inference method: laplace, svi, sgld")
parser.add_argument("--samples", type=int, default=10, help="(redBNN only) Number of posterior samples.")
parser.add_argument("--base_iters", type=int, default=2, help="(redBNN only) Number of training iterations for the basenet.")
parser.add_argument("--debug", type=eval, default="False", help="Run the script in debugging mode.")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
args = parser.parse_args()

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


n_inputs, iters, n_samples = (20, 2, 2) if args.debug else (args.n_inputs, args.iters, args.n_samples)

savedir = _get_torchvision_savedir(args.architecture, args.dataset, args.architecture, args.inference, args.iters, 
                                    args.debug)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_workers=0 if args.device=="cuda" else 4
device = torch.device(args.device)

batch_size = 1 if args.inference=="laplace" else 128
dataloaders_dict, num_classes, _ = load_data(dataset_name=args.dataset, phases=['train','val'],
                                    batch_size=batch_size, n_inputs=n_inputs, num_workers=num_workers)

if args.model=="baseNN":
    model = baseNN(architecture=args.architecture, dataset_name=args.dataset)
    params_to_update = model.initialize_model(architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.to(device)

    model.train(dataloaders_dict, params_to_update, num_iters=iters, device=device)
    model.save(savedir, iters)

elif args.model=="redBNN":
    basenet = baseNN(architecture=args.architecture, dataset_name=args.dataset)
    basenet.initialize_model(architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    basenet_savedir =  _get_torchvision_savedir("baseNN", args.dataset, args.architecture, 
                                                None, args.base_iters, None, args.debug)

    basenet.load(basenet_savedir, args.base_iters, device)

    model = redBNN(architecture=args.architecture, dataset_name=args.dataset, inference=args.inference)
    model.initialize_model(basenet, architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.to(device)

    model.train(dataloaders_dict, num_iters=iters, device=device)
    model.save(savedir, iters)

else:
    raise ValueError

