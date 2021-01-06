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
parser.add_argument("--debug", type=eval, default="False")
parser.add_argument("--train", type=eval, default="True")
parser.add_argument("--attack", type=eval, default="True")
parser.add_argument("--bayesian", type=eval, default="True")
parser.add_argument("--inference", type=str, default="svi", help="laplace, svi")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--n_inputs", type=int, default=None)
parser.add_argument("--iters", type=int, default=15)
parser.add_argument("--attack_method", type=str, default="fgsm")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--savedir", type=str, default=None)
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

if args.debug:
    savedir="debug"
    n_inputs=100
    iters=1
    n_samples=1

else:
    n_inputs=args.n_inputs
    iters=args.iters
    n_samples=args.n_samples

    if args.savedir:
        savedir = args.savedir 

    else:
        if args.bayesian:
            savedir = args.model+"_redBNN_"+args.dataset+"_"+args.inference+"_iters="+str(args.iters)
        else:
            savedir = args.model+"_baseNN_"+args.dataset+"_iters="+str(args.iters)

num_workers=0 if args.device=="cuda" else 4
dataloaders_dict, num_classes, im_random_idxs = load_data(dataset_name=args.dataset, 
                                    batch_size=batch_size, n_inputs=n_inputs, num_workers=num_workers)

############## 
# Initialize #
############## 

if args.bayesian:
    model_bnn = torchvisionBNN(model_name=args.model, dataset_name=args.dataset, inference=args.inference)
    model_bnn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model_bnn.to(device)
    optimizer_bnn = pyro.optim.Adam({"lr":0.001})

    # print(model_bnn.basenet)

    if args.train:
        model_bnn.train(dataloaders_dict, criterion, optimizer_bnn, num_iters=iters, device=device)
        model_bnn.save(savedir, iters)
    else:
        model_bnn.load(savedir, iters, device)

    if args.attack:
        bnn_attack = attack(network=model_bnn, dataloader=dataloaders_dict['test'], 
                     method=args.attack_method, n_samples=n_samples, device=device, savedir=savedir)
    else:
        bnn_attack = load_attack(method=args.attack_method, n_samples=n_samples, savedir=savedir)

        if len(bnn_attack)>n_inputs:
            bnn_attack = bnn_attack[im_random_idxs['test']]
        else:
            print("Max number of available attacks is ", len(bnn_attack))

    evaluate_attack(network=model_bnn, dataloader=dataloaders_dict['test'], adversarial_data=bnn_attack, 
                    n_samples=n_samples, device=device, method=args.attack_method, savedir=savedir)

else:

    model_nn = torchvisionNN(model_name=args.model, dataset_name=args.dataset)
    params_nn = model_nn.initialize_model(model_name=args.model, num_classes=num_classes, 
                                                feature_extract=True, use_pretrained=True)
    model_nn.to(device)
    optimizer_nn = optim.Adam(params_nn, lr=0.001)

    # print(model_nn.basenet)

    if args.train:
        model_nn.train(dataloaders_dict, criterion, optimizer_nn, num_iters=iters, device=device)
        model_nn.save(savedir, iters)
    else:
        model_nn.load(savedir, iters, device)

    if args.attack:
        nn_attack = attack(network=model_nn, dataloader=dataloaders_dict['test'], 
                    method=args.attack_method, device=device, savedir=savedir)
    else:
        nn_attack = load_attack(method=args.attack_method, savedir=savedir)
        
        if len(nn_attack)>n_inputs:
            nn_attack = nn_attack[im_random_idxs['test']]
        else:
            print("Max number of available attacks is ", len(nn_attack))

    evaluate_attack(network=model_nn, dataloader=dataloaders_dict['test'], adversarial_data=nn_attack, 
                    device=device, method=args.attack_method, savedir=savedir)

