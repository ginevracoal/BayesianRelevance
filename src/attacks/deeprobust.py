import torch
import numpy as np
from tqdm import tqdm

from utils.savedir import *
from utils.data import save_to_pickle, load_from_pickle
from utils.torchvision import plot_grid_attacks
from attacks.robustness_measures import softmax_robustness
import attacks.torchvision_gradient_based as torchvision_atks

from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.YOPOpgd import FASTPGD
from deeprobust.image.attack.deepfool import DeepFool


def attack(network, dataloader, method, device, savedir, n_samples=None, hyperparams=None):
    network.to(device)
    network.n_samples = n_samples
    print(f"\n{method} attack")

    if method == "fgsm":
        adversary = FGSM
        adversary_params = {'epsilon': 0.2, 'order': np.inf, 'clip_max': None, 'clip_min': None}

    elif method == "pgd":
        adversary = PGD
        adversary_params = {'epsilon': 0.2, 'clip_max': 1.0, 'clip_min': 0.0, 'print_process': False}

    elif method == "cw":
        adversary = CarliniWagner
        adversary_params = {'confidence': 1e-4, 'clip_max': 1, 'clip_min': 0, 'max_iterations': 1000,
                            'initial_const': 1e-2, 'binary_search_steps': 5, 'learning_rate': 5e-3,
                            'abort_early': True,}

    elif method == "nattack":
        adversary = NATTACK
        adversary_params = {}

    elif method == "yopo":
        adversary = FASTPGD
        adversary_params = {}

    elif method == "deepfool":
        adversary = DeepFool
        adversary_params = {}

    adversarial_data=[]

    # todo: sistemare calcolo gradienti

    for images, labels in tqdm(dataloader):

        for idx, image in enumerate(images):
            image = image.unsqueeze(0)
            label = labels[idx].argmax(-1).unsqueeze(0)

            adversary(network, device)
            perturbed_image = adversary.generate(image, label, **adversary_params)
            perturbed_image = torch.clamp(perturbed_image, 0., 1.)
            adversarial_data.append(perturbed_image)

    adversarial_data = torch.cat(adversarial_data)
    save_attack(savedir=savedir, adversarial_data=adversarial_data, method=method, n_samples=n_samples)
    return adversarial_data

def evaluate_attack(savedir, network, dataloader, adversarial_data, device, method, n_samples=None):
    """ Evaluates the network on the original dataloader and its perturbed version. 
    When using a Bayesian network `n_samples` should be specified for the evaluation.     
    """
    network.to(device)
    print(f"\nEvaluating against the attacks.")

    original_images_list = []   

    with torch.no_grad():

        original_outputs = []
        original_correct = 0.0
        adversarial_outputs = []
        adversarial_correct = 0.0

        for idx, (images, labels) in enumerate(dataloader):

            for image in images:
                original_images_list.append(image)

            images, labels = images.to(device), labels.to(device)
            attacks = adversarial_data[idx:idx+len(images)]

            out = network(images, n_samples)
            original_correct += torch.sum(out.argmax(-1) == labels).item()
            original_outputs.append(out)

            out = network(attacks, n_samples)
            adversarial_correct += torch.sum(out.argmax(-1) == labels).item()
            adversarial_outputs.append(out)

        original_accuracy = 100 * original_correct / len(dataloader.dataset)
        adversarial_accuracy = 100 * adversarial_correct / len(dataloader.dataset)
        print(f"\ntest accuracy = {original_accuracy:.2f}\tadversarial accuracy = {adversarial_accuracy:.2f}",
              end="\t")

        original_outputs = torch.cat(original_outputs)
        adversarial_outputs = torch.cat(adversarial_outputs)
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

    _plot_attack(savedir, original_images_list, adversarial_data, method, n_samples)

    return original_accuracy, adversarial_accuracy, softmax_rob

def _plot_attack(savedir, original_images_list, adversarial_data, method, n_samples=None):
    method = method+"_deeprobust"
    torchvision_atks._plot_attack(savedir, original_images_list, adversarial_data, method, n_samples)

def save_attack(savedir, adversarial_data, method, n_samples=None):
    filename = _get_attacks_filename(savedir, method, n_samples)
    save_to_pickle(data=adversarial_data, path=TESTS+savedir+"/", filename=filename+"_deeprobust.pkl")

def load_attack(savedir, method, n_samples=None):

    filename = _get_attacks_filename(savedir, method, n_samples)
    return load_from_pickle(TESTS+savedir+"/" +filename+"_deeprobust.pkl")

def _get_attacks_filename(savedir, method, n_samples=None):

    if n_samples:
        return savedir+"_samp="+str(n_samples)+"_"+str(method)+"_attack_deeprobust"
    else:
        return savedir+"_"+str(method)+"_attack_deeprobust"