import time
import sys

DATA = "../data/"
TESTS = "../experiments/" 

def get_savedir(model, dataset, architecture, iters, inference=None, baseiters=None, model_idx=None, debug=False):

    if debug:
        return "debug"
    else:
        savedir = model+"/"+dataset+"_"+architecture+"_iters="+str(iters)

        if model_idx is not None:
            savedir=savedir+"_idx="+str(model_idx)

        if inference is not None:
            savedir = savedir+"_"+inference

        if baseiters is not None:
            savedir = savedir+"_baseiters="+str(baseiters)

        return savedir

# def _get_torchvision_savedir(model, dataset, architecture, inference, iters, baseiters, debug):

#     if debug:
#         return "debug"

#     else:
#         if inference:
#             return model+"/"+dataset+"_"+architecture+"_"+inference+"_iters="+str(iters)+"_baseiters="+str(baseiters)
#         else:
#             return model+"/"+dataset+"_"+architecture+"_iters="+str(iters)