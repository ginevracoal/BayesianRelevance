from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from utils.torchvision import *
from networks.torchvision.baseNN import *
from networks.torchvision.redBNN import *
from utils.data import load_from_pickle
from utils.seeding import *
from utils.savedir import _get_torchvision_savedir

import attacks.torchvision_gradient_based as grad_based
import attacks.deeprobust as deeprobust

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
parser.add_argument("--attack_library", type=str, default="deeprobust", help="grad_based, deeprobust")
parser.add_argument("--attack_method", type=str, default="fgsm", help="Attack name: fgsm, pgd.")
parser.add_argument("--load_attack", type=eval, default="False")
parser.add_argument("--debug", type=eval, default="False", help="Run the script in debugging mode.")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
args = parser.parse_args()


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


batch_size = 128

n_inputs, iters, n_samples = (20, 2, 2) if args.debug else (args.n_inputs, args.iters, args.n_samples)

savedir = get_savedir(args.architecture, args.dataset, args.architecture, args.inference, args.iters, 
                                    args.debug)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

num_workers=0 if args.device=="cuda" else 4
device = torch.device(args.device) if args.attack_library=="grad_based" else args.device

dataloaders_dict, num_classes, im_random_idxs = load_data(dataset_name=args.dataset, phases=['test'], 
                                    batch_size=batch_size, n_inputs=n_inputs, num_workers=num_workers)

atk_lib = eval(args.attack_library)
attack = atk_lib.attack
load_attack = atk_lib.load_attack
evaluate_attack = atk_lib.evaluate_attack

if args.model=="baseNN":
    model = baseNN(architecture=args.architecture, dataset_name=args.dataset)
    params_to_update = model.initialize_model(architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.load(savedir, device)

    if args.load_attack:
        adversarial_data = load_attack(method=args.attack_method, savedir=savedir)
        adversarial_data = adversarial_data[im_random_idxs['test']]
    else:
        adversarial_data = attack(network=model, dataloader=dataloaders_dict['test'], 
                    method=args.attack_method, device=device, savedir=savedir)

    evaluate_attack(network=model, dataloader=dataloaders_dict['test'], adversarial_data=adversarial_data, 
                    device=device, method=args.attack_method, savedir=savedir)

elif args.model=="redBNN":
    basenet = baseNN(architecture=args.architecture, dataset_name=args.dataset)
    basenet.initialize_model(architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    basenet_savedir =  get_savedir("baseNN", args.dataset, args.architecture, 
                                                None, args.base_iters, args.debug)
    basenet.load(basenet_savedir, args.base_iters, device)

    model = redBNN(architecture=args.architecture, dataset_name=args.dataset, inference=args.inference)
    model.initialize_model(basenet, architecture=args.architecture, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model.load(savedir, device)

    if args.load_attack:
        adversarial_data = load_attack(method=args.attack_method, n_samples=n_samples, savedir=savedir)  
        adversarial_data = adversarial_data[im_random_idxs['test']]
    else:
        adversarial_data = attack(network=model, dataloader=dataloaders_dict['test'], 
                     method=args.attack_method, n_samples=n_samples, device=device, savedir=savedir)

    evaluate_attack(network=model, dataloader=dataloaders_dict['test'], adversarial_data=adversarial_data, 
                    n_samples=n_samples, device=device, method=args.attack_method, savedir=savedir)

else:
    raise ValueError

