"""
Attacking BNN with increasing strenght and varying the number of samples.
The same posterior samples are used when evaluating against the attacks.
"""

from adversarialAttacks import *
from model_bnn import *
from model_redBNN import *
from utils_models import load_test_net 
import pandas as pd

def build_eps_attacks_df(bnn, device, method, x_test, y_test, epsilon_list, 
                         attack_samples_list, defence_samples_list):

    df = pd.DataFrame(columns=["attack_method", "epsilon", "test_acc", "adv_acc", 
                                   "softmax_rob", "attack_samples", "defence_samples"])

    row_count = 0
    for epsilon in epsilon_list:
        for attack_samples in attack_samples_list:

            x_attack = attack(net=bnn, x_test=x_test, y_test=y_test, 
                              device=device, method=method, filename=bnn.name, 
                              n_samples=attack_samples, hyperparams={"epsilon":epsilon})

            for def_samples in defence_samples_list:
                test_acc, adv_acc, softmax_rob = attack_evaluation(net=bnn, x_test=x_test, 
                        n_samples=def_samples, x_attack=x_attack, y_test=y_test, device=device)
            
                for pointwise_rob in softmax_rob:
                    df_dict = {"epsilon":epsilon, "attack_method":method, 
                               "attack_samples":attack_samples, "defence_samples":def_samples,
                               "test_acc":test_acc, "adv_acc":adv_acc, 
                               "softmax_rob":pointwise_rob.item()}

                    df.loc[row_count] = pandas.Series(df_dict)
                    row_count += 1

    print("\nSaving:", df)
    os.makedirs(os.path.dirname(TESTS+bnn.name+"/"), exist_ok=True)
    df.to_csv(TESTS+bnn.name+"/increasing_eps_"+str(method)+"_attack.csv", 
              index = False, header=True)
    return df

def load_eps_attacks_df(bnn, method, load_dir):
    return pandas.read_csv(load_dir+"/"+str(bnn.name)+"/increasing_eps_"+str(method)+"_attack.csv")


def lineplot_increasing_eps(df, model_type, bnn, method):
    print(df.head())
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

    attack_samples_list = np.unique(df["attack_samples"])

    sns.set_style("darkgrid")
    matplotlib.rc('font', **{'size': 10})
    fig, ax = plt.subplots(nrows=2, ncols=len(attack_samples_list), 
             figsize=(10, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.suptitle(f"{model_type} {method} attack on {bnn.dataset_name}")

    for j, attack_samp in enumerate(attack_samples_list):

        ax[0,j].set_title(f"{attack_samp} attack samples")
        data = df[df["attack_samples"]==attack_samp]

        for (i,y) in [(0,"adv_acc"), (1,"softmax_rob")]:
            sns.lineplot(data=data, x="epsilon", y=y,  style="defence_samples", ax=ax[i,j],
                         palette=["darkorange","darkred","black"],
                         hue="defence_samples", legend="full" if i==0 and j==0 else False)
            
            ax[i,j].set(xlabel='', ylabel='')

        ax[1,0].set(xlabel='attack strength', ylabel='softmax robustness')
        ax[0,0].set(xlabel='', ylabel='adversarial accuracy')
        ax[1,j].set(xlabel='attack strength', ylabel='')

    filename = bnn.name+"/"+bnn.name+"_increasing_eps_"+str(method)+"_attack.png"
    os.makedirs(os.path.dirname(TESTS), exist_ok=True)
    plt.savefig(TESTS + filename)


def main(args):

    epsilon_list=[0.1, 0.15, 0.2, 0.25, 0.3]
    n_samples_list=[1, 100, 500]

    if args.device=="cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    load_dir = DATA if args.load_dir=="DATA" else TESTS
    (x_test, y_test), bnn = load_test_net(model_idx=args.model_idx, model_type=args.model_type, 
                        device=args.device, load_dir=load_dir, n_inputs=args.n_inputs,
                        return_data_loader=False)

    if args.load_df:
        df = load_eps_attacks_df(bnn=bnn, method=args.attack_method, load_dir=TESTS)

    else:
        df = build_eps_attacks_df(bnn=bnn, device=args.device, 
                                 x_test=x_test, y_test=y_test, method=args.attack_method, 
                                 attack_samples_list=n_samples_list, 
                                 defence_samples_list=n_samples_list, 
                                 epsilon_list=epsilon_list)

    lineplot_increasing_eps(df=df, model_type=args.model_type, bnn=bnn, method=args.attack_method)



if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_inputs", default=100, type=int, help="inputs to be attacked")
    parser.add_argument("--model_idx", default=0, type=int, help="choose idx from saved_NNs")
    parser.add_argument("--model_type", default="fullBNN", type=str, help="fullBNN, redBNN, laplBNN")
    parser.add_argument("--load_dir", default='DATA', type=str, help="DATA, TESTS")  
    parser.add_argument("--attack_method", default="fgsm", type=str, help="fgsm, pgd")
    parser.add_argument("--load_df", default=False, type=eval)
    parser.add_argument("--device", default='cuda', type=str, help="cpu, cuda")   
    main(args=parser.parse_args())