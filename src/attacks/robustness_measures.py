import torch
import torch.nn.functional as nnf

DEBUG=False

def softmax_difference(original_predictions, adversarial_predictions):
    """
    Compute the difference between predictions and adversarial 
    predictions.
    """

    # original_predictions = nnf.softmax(original_predictions, dim=-1)
    # adversarial_predictions = nnf.softmax(adversarial_predictions, dim=-1)

    if original_predictions.abs().max()>1 or original_predictions.abs().max()>1:
        raise ValueError("Pass softmax outputs")

    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("Input arrays should have the same length.")

    if DEBUG:
        print("\n\n", original_predictions[0], "\t", adversarial_predictions[0], end="\n\n")

    abs_diff = (original_predictions-adversarial_predictions).abs()
    return abs_diff

def softmax_robustness(original_outputs, adversarial_outputs, norm="linf"):
    """ 
    Robustness = 1 - norm(softmax difference between predictions).
    This robustness measure is strictly dependent on the epsilon chosen for the 
    perturbations.
    """

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)

    if norm=="l2":
        softmax_differences = torch.norm(softmax_differences, dim=-1)

    elif norm=="linf":
        softmax_differences = softmax_differences.max(dim=-1)[0]

    robustness = (torch.ones_like(softmax_differences)-softmax_differences)
    print(f"avg softmax robustness = {robustness.mean().item():.2f}", end="\t")
    print(f"(min = {robustness.min().item():.2f} max = {robustness.max().item():.2f})")
    return robustness


