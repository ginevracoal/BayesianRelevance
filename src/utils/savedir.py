import time
import sys

DATA = "../data/"
TESTS = "../experiments/" #+str(time.strftime('%Y-%m-%d'))+"/"

# def _get_savedir(model, dataset, architecture, inference, iters, debug):

    # todo

def _get_torchvision_savedir(model, dataset, architecture, inference, iters, debug):

    if debug:
        return "debug"

    else:
        if inference:
            return model+"/"+dataset+"_"+architecture+"_"+inference+"_iters="+str(iters)
        else:
            return model+"/"+dataset+"_"+architecture+"_iters="+str(iters)