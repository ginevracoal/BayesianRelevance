"""
Attacking BNN with increasing strenght and varying the number of samples.
The same posterior samples are used when evaluating against the attacks.
"""

from adversarialAttacks import *
from model_bnn import *
from model_redBNN import *
from utils_models import load_test_net 

def build_eps_attacks_df(bnn, device, method, x_test, y_test, epsilon_list, n_samples_list):

    df = pandas.DataFrame(columns=["attack_method", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "n_samples"])

    row_count = 0
    for epsilon in epsilon_list:
        for n_samples in n_samples_list:

            df_dict = {"epsilon":epsilon, "attack_method":method, "n_samples":n_samples}

            x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, 
                              device=device, method=method, filename=bnn.name, 
                              n_samples=n_samples, hyperparams={"epsilon":epsilon})

            test_acc, adv_acc, softmax_rob = attack_evaluation(net=bnn, x_test=x_test, 
                        n_samples=n_samples, x_attack=x_attack, y_test=y_test, device=device)
            
            for pointwise_rob in softmax_rob:
                df_dict.update({"test_acc":test_acc, "adv_acc":adv_acc,
                                "softmax_rob":pointwise_rob.item()})

                df.loc[row_count] = pandas.Series(df_dict)
                row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS+bnn.name+"/"), exist_ok=True)
    df.to_csv(TESTS+bnn.name+"/increasing_eps_"+str(method)+"_attack.csv", 
              index = False, header=True)
    return df

def load_eps_attacks_df(bnn, method, load_dir):
    return pandas.read_csv(load_dir+"/"+str(bnn.name)+"/increasing_eps_"+str(method)+"_attack.csv")


def lineplot_increasing_eps(df, bnn, method):
    print(df.head())
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

    sns.set_style("darkgrid")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orangered","darkred","black"])
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.suptitle(f"{method} attack on {dataset}")
    sns.lineplot(data=df, x="epsilon", y="adv_acc",  style="n_samples", ax=ax[0])
    sns.lineplot(data=df, x="epsilon", y="softmax_rob", style="n_samples", ax=ax[1])
    
    filename = bnn.name+"/increasing_eps_"+str(method)+"_attack.png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def main(args):

    epsilon_list=[0.1, 0.15, 0.2, 0.25, 0.3]
    n_samples_list=[1, 10, 50]

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    load_dir = DATA if args.load_dir=="DATA" else TESTS
    (x_test, y_test), bnn = load_test_net(model_idx=args.model_idx, model_type=args.model_type, 
                        device=args.device, load_dir=load_dir, n_inputs=args.n_inputs,
                        return_data_loader=False)

    if args.attack:

        df = build_eps_attacks_df(bnn=bnn, device=args.device, 
                                 x_test=x_test, y_test=y_test, method=args.attack_method, 
                                 n_samples_list=n_samples_list, epsilon_list=epsilon_list)
    else:
        df = load_eps_attacks_df(bnn=bnn, method=args.attack_method, load_dir=args.load_dir)

    lineplot_increasing_eps(df, dataset=dataset, method=args.attack_method)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=100, type=int, help="inputs to be attacked")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_NNs")
    parser.add_argument("--model_type", default="fullBNN", type=str, help="fullBNN, redBNN, laplBNN")
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--attack", default=True, type=eval)
    parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--load_dir", default='DATA', type=str, help="DATA, TESTS")  
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")   
    main(args=parser.parse_args())