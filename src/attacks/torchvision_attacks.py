from utils.savedir import *
from utils.torchvision import plot_grid_attacks
from attacks.gradient_based import *

def attack(network, dataloader, method, device, n_samples=None, hyperparams=None):
    """ Attacks the given dataloader using the input network and the chosen attack method. 
    When using a Bayesian network `n_samples` should be specified for the attack.
    """

    network.to(device)
    print(f"\n{method} attack")
    original_images_list = []
    adversarial_data = []
    
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        for idx, image in enumerate(inputs):
            image = image.unsqueeze(0)
            label = labels[idx].argmax(-1).unsqueeze(0)

            if method == "fgsm":
                perturbed_image = fgsm_attack(net=network, image=image, label=label, 
                                              hyperparams=hyperparams, n_samples=n_samples)
            elif method == "pgd":
                perturbed_image = pgd_attack(net=network, image=image, label=label, 
                                              hyperparams=hyperparams, n_samples=n_samples)

            original_images_list.append(image.squeeze(0))
            adversarial_data.append(perturbed_image)

    adversarial_data = torch.cat(adversarial_data)

    path = TESTS+network.name+"/" 

    if n_samples:
        filename = network.name+"_"+str(method)+"_attackSamp="+str(n_samples)+"_attack.pkl"
    else:
        filename = network.name+"_"+str(method)+"_attack.pkl"

    save_to_pickle(data=adversarial_data, path=path, filename=filename)

    idxs = np.random.choice(len(original_images_list), 10, replace=False)
    original_images_plot = torch.stack([original_images_list[i].permute(1, 2, 0) for i in idxs])
    perturbed_images_plot = torch.stack([adversarial_data[i].permute(1, 2, 0) for i in idxs])
    plot_grid_attacks(original_images=original_images_plot.detach().cpu(), 
                      perturbed_images=perturbed_images_plot.detach().cpu(), 
                      filename=network.name+"_"+str(method)+".png", savedir=path)

    return adversarial_data

def evaluate_attack(network, dataloader, adversarial_data, device, n_samples=None):
    """ Evaluates the network on the original dataloader and its perturbed version. 
    When using a Bayesian network `n_samples` should be specified for the evaluation.     
    """
    network.to(device)
    print(f"\nEvaluating against the attacks.")

    with torch.no_grad():

        original_outputs = []
        original_correct = 0.0
        adversarial_outputs = []
        adversarial_correct = 0.0

        for idx, (images, labels) in enumerate(dataloader):

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
        print(f"\ntest accuracy = {original_accuracy}\tadversarial accuracy = {adversarial_accuracy}",
              end="\t")

        original_outputs = torch.cat(original_outputs)
        adversarial_outputs = torch.cat(adversarial_outputs)
        softmax_rob = softmax_robustness(original_outputs, adversarial_outputs)

        # print(original_outputs.shape, adversarial_outputs.shape)

    return original_accuracy, adversarial_accuracy, softmax_rob

def load_attack(network, method, n_samples):
    path = TESTS+network.name+"/" 

    if n_samples:
        filename = network.name+"_"+str(method)+"_attack"+"_samp="+str(n_samples)+".pkl"
    else:
        filename = network.name+"_"+str(method)+"_attack.pkl"

    return load_from_pickle(path+filename)