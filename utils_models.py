from utils_data import * 
from model_redBNN import *
from model_bnn import BNN, saved_BNNs
from model_baseNN import baseNN, saved_baseNNs


def load_data_net(model_idx, model_type, n_inputs, device, load_dir=DATA):

    rel_path = DATA if load_dir=="DATA" else TESTS

    if model_type=="fullBNN":
        
        dataset_name, model = saved_BNNs["model_"+str(model_idx)]
        _, test_loader, inp_shape, out_size = data_loaders(dataset_name=dataset_name, batch_size=128, 
                                                           n_inputs=n_inputs, shuffle=True)
                        
        net = BNN(dataset_name, *list(model.values()), inp_shape, out_size)
        net.load(device=device, rel_path=rel_path)

    elif model_type=="redBNN":

        m = saved_redBNNs["model_"+str(model_idx)]
        dataset_name = m["dataset"]
        _, test_loader, inp_shape, out_size = data_loaders(dataset_name=m["dataset"], batch_size=128, 
                                                 n_inputs=n_inputs, shuffle=True)

        nn = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
                    epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
                    activation=m["activation"], architecture=m["architecture"])
        nn.load(rel_path=rel_path, device=device)

        hyp = get_hyperparams(m)
        net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=nn, hyperparams=hyp)
        net.load(n_inputs=m["BNN_inputs"], device=device, rel_path=rel_path)

    else:
        raise AssertionError("Wrong model name.")

    return dataset_name, test_loader, net