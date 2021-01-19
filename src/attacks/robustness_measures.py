import torch
import torch.nn.functional as nnf

DEBUG=False

def softmax_difference(original_predictions, adversarial_predictions):
    """
    Compute the expected l-inf norm of the difference between predictions and adversarial 
    predictions. This is also a point-wise robustness measure.
    """

    original_predictions = nnf.softmax(original_predictions, dim=-1)
    adversarial_predictions = nnf.softmax(adversarial_predictions, dim=-1)

    if len(original_predictions) != len(adversarial_predictions):
        raise ValueError("\nInput arrays should have the same length.")

    if DEBUG:
        print("\n\n", original_predictions[0], "\t", adversarial_predictions[0], end="\n\n")

    softmax_diff = original_predictions-adversarial_predictions
    softmax_diff_norms = softmax_diff.abs().max(dim=-1)[0]

    if softmax_diff_norms.min() < 0. or softmax_diff_norms.max() > 1.:
        raise ValueError("Softmax difference should be in [0,1]")

    return softmax_diff_norms

def softmax_robustness(original_outputs, adversarial_outputs):
    """ 
    This robustness measure is global and it is strictly dependent on the epsilon chosen for the 
    perturbations.
    """

    softmax_differences = softmax_difference(original_outputs, adversarial_outputs)
    robustness = (torch.ones_like(softmax_differences)-softmax_differences)
    print(f"avg softmax robustness = {robustness.mean().item():.2f}")
    return robustness


