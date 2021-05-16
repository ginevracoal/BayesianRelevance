""" Architectures and parameters """

baseNN_settings = {"model_0":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                            "architecture":"conv", "epochs":5, "lr":0.001},
                   "model_1":{"dataset":"fashion_mnist", "hidden_size":1024, "activation":"leaky",
                            "architecture":"conv", "epochs":15, "lr":0.001},
                   "model_2":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                             "architecture":"fc2", "epochs":15, "lr":0.001},
                   "model_3":{"dataset":"fashion_mnist", "hidden_size":1024, "activation":"leaky",
                            "architecture":"fc2", "epochs":15, "lr":0.001},
                   "model_4":{"dataset":"cifar", "hidden_size":1024, "activation":"relu",
                            "architecture":"conv", "epochs":15, "lr":0.001},
                    }


fullBNN_settings = {"model_0":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                               "architecture":"conv", "inference":"svi", "epochs":5, 
                               "lr":0.01, "hmc_samples":None, "warmup":None},
                    "model_1":{"dataset":"fashion_mnist", "hidden_size":1024, "activation":"leaky",
                           "architecture":"conv", "inference":"svi", "epochs":15,
                           "lr":0.001, "hmc_samples":None, "warmup":None},
                    "model_2":{"dataset":"mnist", "hidden_size":512, "activation":"leaky",
                           "architecture":"fc2", "inference":"hmc", "epochs":None,
                           "lr":None, "hmc_samples":100, "warmup":100}, 
                    "model_3":{"dataset":"fashion_mnist", "hidden_size":1024, "activation":"leaky",
                           "architecture":"fc2", "inference":"hmc", "epochs":None,
                           "lr":None, "hmc_samples":100, "warmup":100},
                    "model_4":{"dataset":"cifar", "hidden_size":1024, "activation":"relu",
                           "architecture":"conv", "inference":"svi", "epochs":15,
                           "lr":0.001, "hmc_samples":None, "warmup":None},
                    }  
