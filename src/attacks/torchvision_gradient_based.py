from utils.savedir import *
from plot.attacks import plot_grid_attacks
from attacks.gradient_based import *

def attack(network, dataloader, method, device, savedir, n_samples=None, hyperparams=None):
    """ Attacks the given dataloader using the input network and the chosen attack method. 
    When using a Bayesian network `n_samples` should be specified for the attack.
    """

    network.to(device)
    print(f"\n{method} attack")
    adversarial_data = []
    
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        for idx, image in enumerate(images):
            image = image.unsqueeze(0)
            label = labels[idx].argmax(-1).unsqueeze(0)

            if method == "fgsm":
                perturbed_image = fgsm_attack(net=network, image=image, label=label, 
                                              hyperparams=hyperparams, n_samples=n_samples)
            elif method == "pgd":
                perturbed_image = pgd_attack(net=network, image=image, label=label, 
                                              hyperparams=hyperparams, n_samples=n_samples)

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

        # print(original_outputs.shape, adversarial_outputs.shape)

    _plot_attack(savedir, original_images_list, adversarial_data, method, n_samples)

    return original_accuracy, adversarial_accuracy, softmax_rob

def _plot_attack(savedir, original_images_list, adversarial_data, method, n_samples=None):

    idxs = np.random.choice(len(original_images_list), 10, replace=False)
    original_images_plot = torch.stack([original_images_list[i].permute(1, 2, 0) for i in idxs])
    perturbed_images_plot = torch.stack([adversarial_data[i].permute(1, 2, 0) for i in idxs])
    
    filename = _get_attacks_filename(savedir, method, n_samples)

    plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
                      perturbed_images=perturbed_images_plot.detach().cpu(), 
                      filename=filename+".png", savedir=TESTS+savedir+"/")

def save_attack(savedir, adversarial_data, method, n_samples=None):
    
    filename = _get_attacks_filename(savedir, method, n_samples)
    save_to_pickle(data=adversarial_data, path=TESTS+savedir+"/", filename=filename+".pkl")

def load_attack(savedir, method, n_samples=None):

    filename = _get_attacks_filename(savedir, method, n_samples)
    return load_from_pickle(TESTS+savedir+"/" +filename+".pkl")

def _get_attacks_filename(savedir, method, n_samples=None):

    if n_samples:
        return savedir+"_samp="+str(n_samples)+"_"+str(method)+"_attack"
    else:
        return savedir+"_"+str(method)+"_attack"
