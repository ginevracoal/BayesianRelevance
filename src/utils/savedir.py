import time
import sys

DATA = "../data/"
TESTS = "../experiments/" 

def get_savedir(model, dataset, architecture, iters, inference=None, baseiters=None, model_idx=None, debug=False):

    if debug:
        return TESTS+"debug"
    else:
        savedir = model+"/"+dataset+"_"+architecture+"_iters="+str(iters)

        if model_idx is not None:
            savedir=savedir+"_idx="+str(model_idx)

        if inference is not None:
            savedir = savedir+"_"+inference

        if baseiters is not None:
            savedir = savedir+"_baseiters="+str(baseiters)

        return TESTS+savedir

