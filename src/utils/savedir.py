import os
import sys
import time

DATA = "../data/"
TESTS = "../experiments/" 
ATK_DIR = "attacks/"

def get_model_savedir(model, dataset, architecture, iters=None, inference=None, baseiters=None,
                      model_idx=None, layer_idx=None, debug=False, torchvision=False, attack_method=None):

    if torchvision:
        model = str(model)+"_torchvision"

    savedir = model+"/"+dataset+"_"+architecture

    if iters is not None:
        savedir+="_iters="+str(iters)

    if model_idx is not None:
        savedir+="_idx="+str(model_idx)

    if inference is not None:
        savedir+="_"+inference

    if baseiters is not None:
        savedir+="_baseiters="+str(baseiters)

    if layer_idx:
        savedir+="_layeridx="+str(layer_idx)

    if attack_method:
        savedir+="_atk="+str(attack_method)

    if debug:
        return os.path.join(TESTS, "debug/", savedir)
    else:
        return os.path.join(TESTS, savedir)

def get_lrp_savedir(model_savedir, attack_method, rule, lrp_method=None, layer_idx=None):
    """
    model_savedir: original model directory.
    attack_method: method used for computing the attacks.
    rule: chosen LRP rule.
    lrp_method: Bayesian method for computing the LRP, by default computes the avg heatmap.
    layer_idx: LRP is computed at layer_idx, which is at the last layer by default.
    """

    savedir = str(attack_method)+"/"

    savedir += str(lrp_method)+"_"+str(rule)+"_lrp/" if lrp_method else str(rule)+"_lrp/"

    if layer_idx is not None:
        savedir += "pkl_layer_idx="+str(layer_idx)

    return os.path.join(model_savedir, savedir)

def get_atk_filename_savedir(attack_method, model_savedir, atk_mode=False, n_samples=None):

    if atk_mode:
        filename = str(attack_method)+"_mode_attack"
    else:
        if n_samples:
            filename = str(attack_method)+"_attackSamp="+str(n_samples)+"_attack" 
        else:
            filename = str(attack_method)+"_attack"

    savedir = os.path.join(model_savedir, str(attack_method)+"/"+ATK_DIR)

    return filename, savedir