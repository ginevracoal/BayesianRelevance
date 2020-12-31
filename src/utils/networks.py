from utils.data import * 
from networks.baseNN import *
from networks.fullBNN import *
from networks.redBNN import *


def load_test_net(model_idx, model_type, n_inputs, device, load_dir, return_data_loader=True):

    if model_type=="baseNN":

        model = baseNN_settings["model_"+str(model_idx)]

        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=model["dataset"], 
                                                           n_inputs=n_inputs)[2:]

        net = baseNN(inp_shape, out_size, *list(model.values()))
        net.load(device=device, rel_path=load_dir)

    elif model_type=="fullBNN":
        
        dataset_name, model = fullBNN_settings["model_"+str(model_idx)]
        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=dataset_name, 
                                                           n_inputs=n_inputs)[2:]
                        
        net = BNN(dataset_name, *list(model.values()), inp_shape, out_size)
        net.load(device=device, rel_path=load_dir)

    elif model_type=="redBNN":

        m = redBNN_settingss["model_"+str(model_idx)]
        dataset_name = m["dataset"]
        x_test, y_test, inp_shape, out_size = load_dataset(dataset_name=dataset_name, 
                                                           n_inputs=n_inputs)[2:]

        nn = baseNN(dataset_name=m["dataset"], input_shape=inp_shape, output_size=out_size,
                    epochs=m["baseNN_epochs"], lr=m["baseNN_lr"], hidden_size=m["hidden_size"], 
                    activation=m["activation"], architecture=m["architecture"])
        nn.load(rel_path=load_dir, device=device)

        hyp = get_hyperparams(m)
        net = redBNN(dataset_name=m["dataset"], inference=m["inference"], base_net=nn, hyperparams=hyp)
        net.load(n_inputs=m["BNN_inputs"], device=device, rel_path=load_dir)

    else:
        raise AssertionError("Wrong model name.")

    if return_data_loader:
        test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=128, shuffle=True)
        return test_loader, net

    else:
        return (x_test, y_test), net