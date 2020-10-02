import os
import time
import json
import argparse
import numpy as np
import foolbox
import scipy.io as io
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper')
from datetime import datetime

from utils import L1, L2, NumpyEncoder

SMALL_SIZE = 4.2
MEDIUM_SIZE = 5.8
BIGGER_SIZE = 6.0

TEXT_WIDTH = 4.8041

TICK_LABEL_TO_TICK_DISTANCE = -2  # the lower the closer

LINE_WIDTH = 0.6


def calc_fig_size(n_rows, n_cols, text_width=TEXT_WIDTH):
    ax_width = text_width / 3
    ax_height = text_width / 5
    extra_height = text_width / 4 * 2 - text_width / 5 * 2

    fig_width = n_cols * ax_width
    fig_height = n_rows * ax_height

    if fig_width > text_width:
        factor = text_width / fig_width
        fig_width *= factor
        fig_height *= factor

    fig_height += extra_height

    return fig_width, fig_height


def tex_rob(sub, sup, arg):
    return 'R_{{{}}}^{{{}}}({{{}}})'.format(sub, sup, arg)


X_EPS = r'perturbation size $\varepsilon$'
X_EPS_INF = r'$\ell_\infty$ perturbation size $\varepsilon$'
X_EPS_ONE = r'$\ell_1$ perturbation size $\varepsilon$'
X_EPS_TWO = r'$\ell_2$ perturbation size $\varepsilon$'
Y_ROB = '${}$'.format(tex_rob('', '', r'\varepsilon'))
Y_ROB_INF = '${}$'.format(tex_rob(r'\|\cdot\|_\infty', '', r'\varepsilon'))
Y_ROB_ONE = '${}$'.format(tex_rob(r'\|\cdot\|_1', '', r'\varepsilon'))
Y_ROB_TWO = '${}$'.format(tex_rob(r'\|\cdot\|_2', '', r'\varepsilon'))
Y_ROB_GEN = '${}$'.format(tex_rob(r'\|\cdot\|_p', '', r'\varepsilon'))

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)

colors = {
    "orange": sns.xkcd_rgb["yellowish orange"],
    "red": sns.xkcd_rgb["pale red"],
    "green": sns.xkcd_rgb["medium green"],
    "blue": sns.xkcd_rgb["denim blue"],
    "yellow": sns.xkcd_rgb["amber"],
    "purple": sns.xkcd_rgb["dusty purple"],
    "cyan": sns.xkcd_rgb["cyan"]
}

def time_this(original_function):
    
    """
    Wraps a timing function around a given function.
    """

    def new_function(*args, **kwargs):
        
        timestart = time.time()                  
        x = original_function(*args, **kwargs)               
        timeend = time.time()                     
        print("Took {0:.2f} seconds to run.\n".format(timeend-timestart))
            
        return x           
        
    return new_function

def save_to_json(dictionary, file_name):

    """
    Saves a given dictionary to a json file.
    """
        
    if not os.path.exists("res"):
        os.makedirs("res")

    with open("res/" + file_name + ".json", 'w') as fp:
        json.dump(dictionary, fp, cls = NumpyEncoder)

@time_this
def l_1_attack(f_model, x_test, y_test):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_1 distance to the original input. 
    """
         
    print("Starting l_1 attack.")
    attack = foolbox.attacks.EADAttack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = L1)
    
    adversarials = []
    for i, point in enumerate(x_test):

        adversarials.append(attack(point, np.array([y_test[i].argmax()]), binary_search_steps=10))
        
    adversarials = np.array(adversarials)
        
    return adversarials

@time_this
def l_2_attack(f_model, x_test, y_test):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_2 distance to the original input. 
    """
          
    print("Starting l_2 attack.")
    attack = foolbox.attacks.CarliniWagnerL2Attack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = L2)
    
    adversarials = []
    for i, point in enumerate(x_test):

        adversarials.append(attack(point, np.array([y_test[i].argmax()]), binary_search_steps=10))
        
    adversarials = np.array(adversarials)
        
    return adversarials

@time_this
def l_sup_attack(f_model, x_test, y_test):

    """
    Carries out an adversarial attack for a given model and dataset that optimizes 
    for closest l_infty distance to the original input. 
    """

    print("Starting l_sup attack.")    
    attack = foolbox.attacks.ProjectedGradientDescentAttack(model = f_model, criterion = foolbox.criteria.Misclassification(), distance = foolbox.distances.Linf)
    
    adversarials = []
    for i, point in enumerate(x_test):

        adversarials.append(attack(point, np.array([y_test[i].argmax()])))
        
    adversarials = np.array(adversarials)
        
    return adversarials

def generate_curve_data(args):

    """
    Calculates the robustness curve data for given parameters. 
    Calculates the data for a specific dataset (given by args.inputs and args.labels),
    for a specific model (given by args.f_model),
    with adversarials of minimal distance to the original data points measured in different norms (given by args.norms).
    Optional parameters are whether to save the data (given by args.save),
    and whether to plot the data (given by args.plot).
    You can find examples on how to use this method in 'Readme.md' or in some of the notebooks in the folder 'experiments'.
    """

    save_name = "approximated_robustness_curves_{}".format(str(datetime.now())[:-7])

    NORM_TO_ATTACK = {"inf": l_sup_attack, "2": l_2_attack, "1": l_1_attack}

    test_predictions = []
    for point in args.inputs:

        test_predictions.append(args.f_model.forward(point).argmax())
            
    test_predictions = np.array(test_predictions)

    robustness_curve_data = dict()

    for norm in args.norms:

        attack = NORM_TO_ATTACK[norm]
        adversarials = attack(args.f_model, args.inputs, args.labels)

        dists_r = np.array([np.linalg.norm(x = vector, ord = np.inf) for vector in np.subtract(adversarials.reshape(adversarials.shape[0],-1 ), args.inputs.reshape(args.inputs.shape[0], -1))])
        dists_r[test_predictions != args.labels.argmax(axis=1)] = 0
        dists_r.sort(axis=0)

        probs = 1/float(test_predictions.shape[0]) * np.arange(1, test_predictions.shape[0]+1)

        probs[np.isnan(dists_r)] = 1.0
        dists_r = np.nan_to_num(dists_r, nan = np.nanmax(dists_r))

        robustness_curve_data[norm] = {"x" : dists_r, "y": probs }

    if args.save == True:
        save_to_json(robustness_curve_data, save_name)

    if args.plot == True:
        plot_curve_data(robustness_curve_data)

    return robustness_curve_data

def plot_curve_data(robustness_curve_data):

    """
    Plots the robustness curve data.
    """

    save_name = "approximated_robustness_curves_{}".format(str(datetime.now())[:-7])

    fig, ax = plt.subplots(1,
                           1,
                           figsize=calc_fig_size(1, 1), dpi=250)

    norms = robustness_curve_data.keys()
    colors = sns.color_palette(n_colors=len(norms))
    norm_to_latex = {"inf":"\infty", "2":"2", "1": "1"}

    for i, norm in enumerate(norms):

        robustness_curve_data[norm]["x"] = np.insert(robustness_curve_data[norm]["x"], 0, 0.0, axis=0)
        robustness_curve_data[norm]["y"] = np.insert(robustness_curve_data[norm]["y"], 0, 0.0, axis=0)

        ax.plot(robustness_curve_data[norm]["x"], robustness_curve_data[norm]["y"], c = colors[i], label = "$\ell_{}$ robustness curve".format(norm_to_latex[norm]))

    ax.legend()
    ax.set_ylabel(Y_ROB_GEN)
    ax.set_xlabel(r'perturbation threshold')
    ax.set_title("robustness curves")
    ax.set_xlim(left=0.0)

    fig.tight_layout()

    if not os.path.exists("res"):
        os.makedirs("res")

    fig.savefig('res/{}.pdf'.format(save_name))

    plt.show()

