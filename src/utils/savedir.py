import os
import sys
import time

DATA = "../data/"
TESTS = "../experiments/" 
ATK_DIR = "attacks/"

def get_model_savedir(model, dataset, architecture, iters=None, inference=None, baseiters=None,
                      model_idx=None, debug=False, torchvision=False):

    savedir = model+"/"+dataset+"_"+architecture

    if iters is not None:
        savedir+="_iters="+str(iters)

    if model_idx is not None:
        savedir+="_idx="+str(model_idx)

    if inference is not None:
        savedir+="_"+inference

    if baseiters is not None:
        savedir+="_baseiters="+str(baseiters)

    if torchvision:
        savedir+="torchvision/"

    if debug:
        return os.path.join(TESTS, "debug/", savedir)
    else:
        return os.path.join(TESTS, savedir)

def get_lrp_savedir(model_savedir, attack_method, layer_idx, normalize=False):

    if normalize:
        return os.path.join(model_savedir, str(attack_method)+"/lrp/pkl_layer_idx="+str(layer_idx)+"_norm/")
    else:
        return os.path.join(model_savedir, str(attack_method)+"/lrp/pkl_layer_idx="+str(layer_idx)+"/")

def get_atk_filename_savedir(attack_method, model_savedir, atk_mode=False, n_samples=None):

    filename = str(attack_method)+"_attackSamp="+str(n_samples)+"_attack" if n_samples else str(attack_method)+"_attack"
    if atk_mode:
        filename+="_mode"

    savedir = os.path.join(model_savedir, str(attack_method)+"/"+ATK_DIR)

    return filename, savedir