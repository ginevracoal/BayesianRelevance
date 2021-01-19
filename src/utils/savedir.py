import os
import sys
import time

DATA = "../data/"
TESTS = "../experiments/" 
LRP_DIR="lrp"
ATK_DIR="attacks/"

def get_savedir(model, dataset, architecture, iters=None, inference=None, baseiters=None,
                model_idx=None, debug=False):

    savedir = model+"/"+dataset+"_"+architecture

    if iters is not None:
        savedir=savedir+"_iters="+str(iters)

    if model_idx is not None:
        savedir=savedir+"_idx="+str(model_idx)

    if inference is not None:
        savedir=savedir+"_"+inference

    if baseiters is not None:
        savedir=savedir+"_baseiters="+str(baseiters)

    if debug:
        return os.path.join(TESTS, "debug/", savedir)
    else:
        return os.path.join(TESTS, savedir)


def lrp_savedir(layer_idx):
    return LRP_DIR