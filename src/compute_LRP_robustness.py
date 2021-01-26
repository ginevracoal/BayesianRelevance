import os
import argparse
import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as nnf
import torch.optim as torchopt
import torch.nn.functional as F

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
parser.add_argument("--topk", default=100, type=int, help="Top k most relevant pixels.")
parser.add_argument("--model_idx", default=0, type=int, help="Choose model idx from pre defined settings")
parser.add_argument("--model", default="fullBNN", type=str, help="baseNN, fullBNN, redBNN")
parser.add_argument("--inference", default="svi", type=str, help="svi, hmc")
parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
parser.add_argument("--lrp_method", default="intersection", type=str, help="intersection, union, average")
parser.add_argument("--rule", default="epsilon", type=str, help="Rule for LRP computation.")
parser.add_argument("--layer_idx", default=-1, type=int, help="Layer idx for LRP computation.")
parser.add_argument("--load", default=False, type=eval, help="Load saved computations and evaluate them.")
parser.add_argument("--explain_attacks", default=False, type=eval)
parser.add_argument("--debug", default=False, type=eval, help="Run script in debugging mode.")
parser.add_argument("--device", default='cpu', type=str, help="cpu, cuda")  
args = parser.parse_args()

n_samples_list=[10,50,100] if args.model_idx<=1 else [5,10,50]
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

# elif args.model=="redBNN":

#     m = redBNN_settings["model_"+str(args.model_idx)]
#     base_m = baseNN_settings["model_"+str(m["baseNN_idx"])]

#     _, _, x_test, y_test, inp_shape, num_classes = load_dataset(dataset_name=m["dataset"], 
#                                                                 n_inputs=n_inputs)

#     savedir = get_savedir(model=args.model, dataset=m["dataset"], architecture=m["architecture"], 
#                           debug=args.debug, model_idx=args.model_idx)
#     basenet = baseNN(inp_shape, num_classes, *list(base_m.values()))
#     basenet_savedir = get_savedir(model="baseNN", dataset=m["dataset"], 
#                       architecture=m["architecture"], debug=args.debug, model_idx=m["baseNN_idx"])
#     basenet.load(savedir=basenet_savedir, device=args.device)

#     hyp = get_hyperparams(m)
#     net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=basenet, hyperparams=hyp)
# ## todo: load net

else:
    raise NotImplementedError

images = x_test.to(args.device)
labels = y_test.argmax(-1).to(args.device)
savedir = os.path.join(model_savedir, "lrp/robustness/")

### Deterministic explanations

if args.load:
    det_softmax_robustness = load_from_pickle(path=savedir, filename="det_softmax_robustness")
    det_lrp_robustness = load_from_pickle(path=savedir, filename="det_lrp_robustness")

else:

    det_attack = attack(net=detnet, x_test=images, y_test=y_test, device=args.device, method=args.attack_method)
    det_softmax_robustness = attack_evaluation(net=detnet, x_test=images, x_attack=det_attack, 
                                               y_test=y_test, device=args.device)[2].detach().cpu().numpy()

    lrp = compute_explanations(images, detnet, rule=args.rule)
    attack_lrp = compute_explanations(det_attack, detnet, rule=args.rule)
    det_lrp_robustness, pxl_idxs = lrp_robustness(original_heatmaps=lrp, adversarial_heatmaps=attack_lrp, 
                                                  topk=topk, method=args.lrp_method)

    # lrp_attack = attack(net=detnet, x_test=lrp, y_test=y_test, device=args.device, method=args.attack_method)

    plot_attacks_explanations(images=images, explanations=lrp, attacks=det_attack, 
                              attacks_explanations=attack_lrp, #explanations_attacks=lrp_attack,
                              rule=args.rule, savedir=savedir, pxl_idxs=pxl_idxs,
                              filename="det_lrp_attacks", layer_idx=-1)

    save_to_pickle(det_softmax_robustness, path=savedir, filename="det_softmax_robustness")
    save_to_pickle(det_lrp_robustness, path=savedir, filename="det_lrp_robustness")

### Bayesian explanations

bay_softmax_robustness=np.zeros((len(n_samples_list), n_inputs))
post_lrp_robustness=np.zeros((len(n_samples_list), n_inputs))
# avg_lrp_robustness=np.zeros((len(n_samples_list), n_inputs))

if args.load:

    for idx, n_samples in enumerate(n_samples_list):
        bay_softmax_robustness[idx] = load_from_pickle(path=savedir, 
                                      filename="softmax_robustness_samp="+str(n_samples))    
        post_lrp_robustness[idx] = load_from_pickle(path=savedir, 
                                      filename="post_lrp_robustness_samp="+str(n_samples))
        # avg_lrp_robustness[idx] = load_from_pickle(path=savedir, 
        #                               filename="avg_lrp_robustness_samp="+str(n_samples))
    
    mode_softmax_robustness = load_from_pickle(path=savedir, filename="mode_softmax_robustness_samp="+str(n_samples))
    mode_lrp_robustness = load_from_pickle(path=savedir, filename="mode_lrp_robustness_samp="+str(n_samples))

else:

    for idx, n_samples in enumerate(n_samples_list):

        bay_attack = attack(net=bayesnet, x_test=images, y_test=y_test, n_samples=n_samples,
                          device=args.device, method=args.attack_method)
        bay_softmax_robustness[idx] = attack_evaluation(net=bayesnet, x_test=images, x_attack=bay_attack,
                               y_test=y_test, device=args.device, n_samples=n_samples)[2].detach().cpu().numpy()
        
        post_lrp = compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples)
        post_attack_lrp = compute_explanations(bay_attack, bayesnet, rule=args.rule, n_samples=n_samples)
        post_lrp_robustness[idx], pxl_idxs = lrp_robustness(original_heatmaps=post_lrp, 
                                                      adversarial_heatmaps=post_attack_lrp, 
                                                      topk=topk, method=args.lrp_method)

        plot_attacks_explanations(images=images, explanations=post_lrp, attacks=bay_attack, 
                                  attacks_explanations=post_attack_lrp, #sexplanations_attacks=avg_lrp_attack,
                                  rule=args.rule, savedir=savedir, pxl_idxs=pxl_idxs,
                                  filename="post_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

        # avg_lrp = compute_avg_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples)
        # avg_attack_lrp = compute_avg_explanations(bay_attack, bayesnet, rule=args.rule, n_samples=n_samples)
        # avg_lrp_robustness[idx], pxl_idxs = lrp_robustness(original_heatmaps=avg_lrp, 
        #                                               adversarial_heatmaps=avg_attack_lrp, 
        #                                               topk=topk, method=args.lrp_method)

        # # avg_lrp_attack = attack(net=bayesnet, x_test=avg_lrp, y_test=y_test, 
        # #                  device=args.device, method=args.attack_method, n_samples=n_samples)

        # plot_attacks_explanations(images=images, explanations=avg_lrp, attacks=bay_attack, 
        #                           attacks_explanations=avg_attack_lrp, #explanations_attacks=avg_lrp_attack,
        #                           rule=args.rule, savedir=savedir, pxl_idxs=pxl_idxs,
        #                           filename="avg_lrp_attacks_samp="+str(n_samples), layer_idx=-1)
        
        save_to_pickle(bay_softmax_robustness[idx], path=savedir, 
                        filename="softmax_robustness_samp="+str(n_samples))
        save_to_pickle(post_lrp_robustness[idx], path=savedir, 
                        filename="post_lrp_robustness_samp="+str(n_samples))
        # save_to_pickle(avg_lrp_robustness[idx], path=savedir, 
        #                 filename="avg_lrp_robustness_samp="+str(n_samples))

    mode_attack = attack(net=bayesnet, x_test=images, y_test=y_test, n_samples=n_samples,
                      device=args.device, method=args.attack_method, avg_posterior=True)
    mode_softmax_robustness = attack_evaluation(net=bayesnet, x_test=images, x_attack=mode_attack,
                           y_test=y_test, device=args.device, n_samples=n_samples)[2].detach().cpu().numpy()
    mode_lrp = compute_explanations(images, bayesnet, rule=args.rule, n_samples=n_samples, avg_posterior=True)
    mode_attack_lrp = compute_explanations(mode_attack, bayesnet, rule=args.rule, 
                                            n_samples=n_samples, avg_posterior=True)
    mode_lrp_robustness, mode_pxl_idxs = lrp_robustness(original_heatmaps=mode_lrp, 
                                                    adversarial_heatmaps=mode_attack_lrp, 
                                                    topk=topk, method=args.lrp_method)
    # mode_lrp_attack = attack(net=bayesnet, x_test=mode_lrp, y_test=y_test, avg_posterior=True,
    #                   device=args.device, method=args.attack_method, n_samples=n_samples)

    plot_attacks_explanations(images=images, explanations=mode_lrp, attacks=mode_attack, 
                              attacks_explanations=mode_attack_lrp, #explanations_attacks=mode_lrp_attack,
                              rule=args.rule, savedir=savedir, pxl_idxs=mode_pxl_idxs,
                              filename="mode_lrp_attacks_samp="+str(n_samples), layer_idx=-1)

    save_to_pickle(mode_softmax_robustness, path=savedir, filename="mode_softmax_robustness_samp="+str(n_samples))
    save_to_pickle(mode_lrp_robustness, path=savedir, filename="mode_lrp_robustness_samp="+str(n_samples))


### Plot

filename=args.rule+"_lrp_robustness"+m["dataset"]+"_images="+str(n_inputs)+\
         "_samples="+str(n_samples)+"_pxls="+str(topk)+"_atk="+str(args.attack_method)

plot_lrp.lrp_robustness_distributions(lrp_robustness=det_lrp_robustness, 
                                      bayesian_lrp_robustness=post_lrp_robustness, 
                                      mode_lrp_robustness=mode_lrp_robustness,
                                      n_samples_list=n_samples_list,
                                      savedir=savedir, filename="dist_"+filename)

plot_lrp.lrp_robustness_scatterplot(adversarial_robustness=det_softmax_robustness, 
                                    bayesian_adversarial_robustness=bay_softmax_robustness,
                                    mode_adversarial_robustness=mode_softmax_robustness,
                                    lrp_robustness=det_lrp_robustness, 
                                    bayesian_lrp_robustness=post_lrp_robustness,
                                    mode_lrp_robustness=mode_lrp_robustness,
                                    n_samples_list=n_samples_list,
                                    savedir=savedir, filename="scatterplot_"+filename)