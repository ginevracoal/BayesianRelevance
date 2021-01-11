import time
import sys

DATA = "../data/"
TESTS = "../experiments/" 

def _get_savedir(model, dataset, architecture, inference, iters, baseiters, debug):

    if debug:
        return "debug"

    else:
        if inference:
            return model+"/"+dataset+"_"+architecture+"_"+inference+"_iters="+str(iters)+"_baseiters="+str(baseiters)
        else:
            return model+"/"+dataset+"_"+architecture+"_iters="+str(iters)

def _get_torchvision_savedir(model, dataset, architecture, inference, iters, baseiters, debug):

    if debug:
        return "debug"

    else:
        if inference:
            return model+"/"+dataset+"_"+architecture+"_"+inference+"_iters="+str(iters)+"_baseiters="+str(baseiters)
        else:
            return model+"/"+dataset+"_"+architecture+"_iters="+str(iters)