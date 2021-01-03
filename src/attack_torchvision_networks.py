from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser

from utils.torchvision import *
from networks.torchvision.baseNN import *
from networks.torchvision.redBNN import *
from utils.data import load_from_pickle
from utils.seeding import *
from attacks.torchvision_attacks import *

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="resnet", help="resnet, alexnet, vgg")
parser.add_argument("--dataset", type=str, default="animals10", 
                    help="imagenette, imagewoof, animals10, hymenoptera")
parser.add_argument("--bayesian", type=eval, default="True")
parser.add_argument("--inference", type=str, default="svi", help="laplace, svi")
parser.add_argument("--train", type=eval, default="True")
parser.add_argument("--attack", type=eval, default="True")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--iters", type=int, default=15)
parser.add_argument("--attack_method", type=str, default="fgsm")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--debug", type=eval, default="False")
args = parser.parse_args()

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

################
# common setup #
################

device = torch.device(args.device)
criterion = nn.CrossEntropyLoss()

batch_size = 128
dataloaders_dict, num_classes = load_data(dataset_name=args.dataset, batch_size=batch_size, debug=args.debug)

############## 
# Initialize #
############## 

if args.bayesian is False:

    model_nn = torchvisionNN(model_name=args.model, dataset_name=args.dataset)
    model_nn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model_nn.to(device)
    params_nn = set_params_updates(model_nn.basenet, feature_extract=True)
    optimizer_nn = optim.Adam(params_nn, lr=0.001)
    iters = 1 if args.debug else args.iters

    # print(model_nn.basenet)

    if args.train:
        model_nn.train(dataloaders_dict, criterion, optimizer_nn, num_iters=iters, device=device)
        model_nn.save(iters)
    else:
        model_nn.load(iters, device)

    if args.attack:
        nn_attack = attack(network=model_nn, dataloader=dataloaders_dict['test'], 
            method=args.attack_method, device=device)
    else:
        nn_attack = load_attack(network=model_nn, method=args.attack_method)

    evaluate_attack(network=model_nn, dataloader=dataloaders_dict['test'], 
                    adversarial_data=nn_attack, device=device)

else:

    model_bnn = torchvisionBNN(model_name=args.model, dataset_name=args.dataset, inference=args.inference)
    model_bnn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model_bnn.to(device)
    set_params_updates(model_bnn.basenet, feature_extract=True)
    optimizer_bnn = pyro.optim.Adam({"lr":0.001})
    iters = 1 if args.inference=="laplace" else args.iters
    iters = 1 if args.debug else iters

    # print(model_bnn.basenet)

    if args.train:
        model_bnn.train(dataloaders_dict, criterion, optimizer_bnn, num_iters=iters, device=device)
        model_bnn.save(iters)
    else:
        model_bnn.load(iters, device)

    n_samples = 1 if args.debug else args.n_samples

    if args.attack:
        bnn_attack = attack(network=model_bnn, dataloader=dataloaders_dict['test'], 
            method=args.attack_method, n_samples=n_samples, device=device)
    else:
        bnn_attack = load_attack(network=model_bnn, method=args.attack_method, n_samples=n_samples)

    evaluate_attack(network=model_bnn, dataloader=dataloaders_dict['test'], adversarial_data=bnn_attack, 
                    n_samples=n_samples, device=device)
