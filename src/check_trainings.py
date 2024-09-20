import os
# import shutil

path = os.path.join('/mnt/c/Users/aless/Desktop/pykan/KAN-Continual_Learning', 'results', 'mnist', 'domainIL_comp_ultimi')
for fold in os.listdir(path):
    if ("ep7" in fold) and not "lr-3" in fold:
    # if "ep" in fold and not "lr-3" in fold and not "sched" in fold:
        missing = []
        trainings = os.listdir(os.path.join(path, fold))
        # if not "Efficient_KAN_Fix" in trainings:
        #     missing.append("Efficient_KAN_Fix")
        # if not "Efficient_KAN_Fixall" in trainings: 
        #     missing.append("Efficient_KAN_Fixall")
        # if not "MLPBig" in trainings: 
        #     missing.append("MLPBig")
        # if not "MLP" in trainings: 
        #     missing.append("MLP")
        if not "Py_KAN" in trainings: 
            missing.append("Py_KAN")
        else:
            epochs = os.listdir(os.path.join(path, fold, "Py_KAN"))
            if len(epochs) != 5 * int(fold[2:3]):
                print(fold, "\t", len(epochs))
        if not "Py_KAN_Fix" in trainings: 
            missing.append("Py_KAN_Fix")
        else:
            epochs = os.listdir(os.path.join(path, fold, "Py_KAN_Fix"))
            if len(epochs) != 5 * int(fold[2:3]):
                print(fold, "\t", len(epochs))
        if len(missing) != 0:
            print(fold, "\t", missing)


# path = '/mnt/c/Users/aless/Desktop/pykan/KAN-Continual_Learning/KAN_src/results/mnist/domainIL_comp_ultimi'
# trains = os.listdir(path)

# path_new = '/mnt/c/Users/aless/Desktop/pykan/KAN-Continual_Learning/results/mnist/domainIL_comp_ultimi'
# for fold in trains:
#     shutil.move(os.path.join(path, fold), os.path.join(path_new, fold))