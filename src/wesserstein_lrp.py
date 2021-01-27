import os
import torch
import argparse
import numpy as np
import torchvision

from utils.data import *
from utils.networks import *
from utils.savedir import *
from utils.seeding import *
from attacks.gradient_based import *
from attacks.gradient_based import load_attack
from utils.lrp import *
from plot.lrp_heatmaps import *
import plot.lrp_distributions as plot_lrp

from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_inputs", default=1000, type=int, help="Number of test points")
parser.add_argument("--topk", default=50, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="union", type=str, help="intersection, union, average")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_samples_list=[1,5,10] #[10,50,100] if args.model_idx<=1 else [5,10,50]
n_inputs=60 if args.debug else args.n_inputs
topk=10 if args.debug else args.topk
# n_samples=2 if args.debug else args.n_samples

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

### Load models

model = baseNN_settings["model_"+str(args.model_idx)]

_, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=model["dataset"], 
                                                            shuffle=True, n_inputs=n_inputs)
model_savedir = get_savedir(model="baseNN", dataset=model["dataset"], architecture=model["architecture"], 
                      debug=args.debug, model_idx=args.model_idx)
detnet = baseNN(inp_shape, num_classes, *list(model.values()))
detnet.load(savedir=model_savedir, device=args.device)

if args.model=="fullBNN":

    m = fullBNN_settings["model_"+str(args.model_idx)]

    model_savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
                                model_idx=args.model_idx, debug=args.debug)

    bayesnet = BNN(m["dataset"], *list(m.values())[1:], inp_shape, num_classes)
    bayesnet.load(savedir=model_savedir, device=args.device)

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/wesserstein/")

### Deterministic explanations_attacks

if args.load:
    det_attack = load_from_pickle(path=savedir, filename="det_attack")
    lrp = load_from_pickle(path=savedir, filename="det_lrp")
    attack_lrp = load_from_pickle(path=savedir, filename="det_attack_lrp")
    pxl_idxs = load_from_pickle(path=savedir, filename="det_pxl_idxs")
    successful_idxs = load_from_pickle(path=savedir, filename="det_successful_idxs")

else:

    det_attack = attack(net=detnet, x_test=images, y_test=y_test, device=args.device, method=args.attack_method)
    successful_idxs = attack_evaluation(net=detnet, x_test=images, x_attack=det_attack, 
                                        y_test=y_test, device=args.device, return_successful_idxs=True)[3]

    lrp = compute_explanations(images, detnet, rule=args.rule)
    attack_lrp = compute_explanations(det_attack, detnet, rule=args.rule)
    pxl_idxs = lrp_robustness(original_heatmaps=lrp, adversarial_heatmaps=attack_lrp, 
                              topk=topk, method=args.lrp_method)[1]

    save_to_pickle(det_attack, path=savedir, filename="det_attack")
    save_to_pickle(lrp, path=savedir, filename="det_lrp")
    save_to_pickle(attack_lrp, path=savedir, filename="det_attack_lrp")
    save_to_pickle(pxl_idxs, path=savedir, filename="det_pxl_idxs")
    save_to_pickle(successful_idxs, path=savedir, filename="det_successful_idxs")

plot_attacks_explanations(images=images, explanations=lrp, attacks=det_attack, 
                          attacks_explanations=attack_lrp, rule=args.rule, savedir=savedir, 
                          pxl_idxs=pxl_idxs, filename="det_lrp_attacks", layer_idx=-1)

det_wess_dist = lrp_wesserstein_distance(lrp, attack_lrp, pxl_idxs)

### Bayesian explanations

bay_attack=[]
bay_lrp=[]
bay_attack_lrp=[]
bay_pxl_idxs=[]
bay_successful_idxs=[]

if args.load:

    for idx, n_samples in enumerate(n_samples_list):

        bay_attack.append(load_from_pickle(path=savedir, filename="bay_attack"))
        bay_lrp.append(load_from_pickle(path=savedir, filename="bay_lrp"))
        bay_attack_lrp.append(load_from_pickle(path=savedir, filename="bay_attack_lrp"))
        bay_pxl_idxs.append(load_from_pickle(path=savedir, filename="bay_pxl_idxs"))
        bay_successful_idxs.append(load_from_pickle(path=savedir, filename="bay_successful_idxs"))

else:

    for idx, n_samples in enumerate(n_samples_list):

        bay_attack.append(attack(net=bayesnet, x_test=images, y_test=y_test, n_samples=n_samples,
                            device=args.device, method=args.attack_method))
        bay_successful_idxs.append(attack_evaluation(net=bayesnet, x_test=images, x_attack=bay_attack[idx],
                               y_test=y_test, device=args.device, n_samples=n_samples, 
                               return_successful_idxs=True)[3])
        
        bay_lrp.append(compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples))
        bay_attack_lrp.append(compute_explanations(bay_attack[idx], bayesnet, 
                                                   rule=args.rule, n_samples=n_samples))
        bay_pxl_idxs.append(lrp_robustness(original_heatmaps=bay_lrp[idx], 
                                           adversarial_heatmaps=bay_attack_lrp[idx], 
                                           topk=topk, method=args.lrp_method)[1])

    save_to_pickle(bay_attack, path=savedir, filename="bay_attack")
    save_to_pickle(bay_lrp, path=savedir, filename="bay_lrp")
    save_to_pickle(bay_attack_lrp, path=savedir, filename="bay_attack")
    save_to_pickle(bay_pxl_idxs, path=savedir, filename="bay_pxl_idxs")
    save_to_pickle(bay_successful_idxs, path=savedir, filename="bay_successful_idxs")


bay_wess_dist=[]
for samp_idx, n_samples in enumerate(n_samples_list):

    plot_attacks_explanations(images=images, explanations=bay_lrp[samp_idx], attacks=bay_attack[samp_idx], 
                              attacks_explanations=bay_attack_lrp[samp_idx], pxl_idxs=bay_pxl_idxs[samp_idx],
                              rule=args.rule, savedir=savedir, 
                              filename="bay_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

    bay_wess_dist.append(lrp_wesserstein_distance(bay_lrp[samp_idx], bay_attack_lrp[samp_idx], 
                                                   bay_pxl_idxs[samp_idx]))


### Plot

# filename=args.rule+"_lrp_robustness"+m["dataset"]+"_images="+str(n_inputs)+\
#          "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)

# plot_lrp.lrp_robustness_distributions(lrp_robustness=det_lrp_robustness, 
#                                       bayesian_lrp_robustness=post_lrp_robustness, 
#                                       mode_lrp_robustness=mode_lrp_robustness,
#                                       n_samples_list=n_samples_list,
#                                       savedir=savedir, filename="dist_"+filename)

# plot_lrp.lrp_robustness_scatterplot(adversarial_robustness=det_softmax_robustness, 
#                                     bayesian_adversarial_robustness=bay_softmax_robustness,
#                                     mode_adversarial_robustness=mode_softmax_robustness,
#                                     lrp_robustness=det_lrp_robustness, 
#                                     bayesian_lrp_robustness=post_lrp_robustness,
#                                     mode_lrp_robustness=mode_lrp_robustness,
#                                     n_samples_list=n_samples_list,
#                                     savedir=savedir, filename="scatterplot_"+filename)