import uncertainpy as un
import chaospy as cp
import numpy as np

import statistics
from scipy.stats import qmc
import matplotlib.pyplot as plt


def linear_fie(x1, x2, x3):
    return 0, x1 + x2 + x1 * x3


def ishigami(x1, x2, x3):
    return 0, np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)


def morris_init():

    b0 = np.array(np.random.randn())
    b1 = np.random.randn(20)
    b2 = np.random.randn(20, 20)
    b3 = np.random.randn(20, 20, 20)
    b4 = np.random.randn(20, 20, 20, 20)

    # b0 = np.array(0)
    # b1 = np.zeros((20))
    # b2 = np.zeros((20, 20))
    # b3 = np.zeros((20, 20, 20))
    # b4 = np.zeros((20, 20, 20, 20))

    for i in range(10):
        b1[i] = 20.

    for i in range(6):
        for j in range(6):
            b2[i, j] = -15.

    for i in range(5):
        for j in range(5):
            for k in range(5):
                b3[i, j, k] = -10.

    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    b4[i, j, k, l] = 5.

    return b0, b1, b2, b3, b4


def morris_fie(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, **kwargs):

    x_list = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20])
    b0 = kwargs['coefficients_morris'][0]
    b1 = kwargs['coefficients_morris'][1]
    b2 = kwargs['coefficients_morris'][2]
    b3 = kwargs['coefficients_morris'][3]
    b4 = kwargs['coefficients_morris'][4]

    w_list = []
    for i in range(1, 21):
        if i == 3 or i == 5 or i == 7:
            w_list.append(2 * (1.1 * x_list[i-1] / (x_list[i-1] + 0.1) - 0.5))
        else:
            w_list.append(2*(x_list[i-1]-0.5))

    y1 = 0.
    y2 = 0.
    y3 = 0.
    y4 = 0.

    for i in range(1, 21):
        y1 += b1[i-1] * w_list[i-1]

        for j in range(1, 21):
            if i < j:
                y2 += b2[i-1, j-1] * w_list[i-1] * w_list[j-1]

            for k in range(1, 21):
                if i < j < k:
                    y3 += b3[i-1, j-1, k-1] * w_list[i-1] * w_list[j-1] * w_list[k-1]

                for l in range(1, 21):
                    if i < j < k < l:
                        y4 += b4[i-1, j-1, k-1, l-1] * w_list[i-1] * w_list[j-1] * w_list[k-1] * w_list[l-1]

    return 0, b0 + y1 + y2 + y3 + y4


def morris_fie_fortran(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, **kwargs):
    import sys
    sys.path.append(r'C:\Users\verd_he\tools\Morris-python')
    from morris import morris
    x_list = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20])
    return 0, morris(x_list)


def oat_screening(self, **kwargs):
    """
    Hübler:
    "The general concept is to vary one parameter while freezing all others. In most cases, only two values
    (maximum and minimum) are tested. Non-linear effects and interactions between inputs are neglected.
    A sensitivity index for the OAT method for the kth input factor can be defined as the (symmetric) derivative
    with respect to the kth input factor"
    """
    uncertain_params = self.convert_uncertain_parameters()
    nr_params = len(uncertain_params)
    nr_samples_per_param = 2  # lower and upper
    all_distributions = self.create_distribution()
    sample_means = [distr.mom([1])[0] for distr in all_distributions]

    nodes = np.tile(np.array(sample_means).reshape(len(sample_means), 1), nr_params*nr_samples_per_param)
    nodes_mean = np.copy(nodes)

    linear_disturbance = True
    for idx_param in range(nr_params):
        if linear_disturbance is True:
            print('The uncertain paramaters are disturbed by 1E-9 x Distribution.lower and 1E-9 x Distribution.upper. '
                  'This makes sure only the linear effect of the uncertain parameter is computed.')
            nodes[idx_param, idx_param*nr_samples_per_param] = all_distributions[idx_param].lower[0] * 1E-9
            nodes[idx_param, idx_param * nr_samples_per_param + 1] = all_distributions[idx_param].upper[0] * 1E-9
        else:
            print('The uncertain paramaters are disturbed by -Distribution.lower and +Distribution.upper. '
                  'This could give false results for nonlinear functions.')
            nodes[idx_param, idx_param*nr_samples_per_param] = all_distributions[idx_param].lower[0]
            nodes[idx_param, idx_param * nr_samples_per_param + 1] = all_distributions[idx_param].upper[0]

    deltas = np.sum(nodes - nodes_mean, axis=0)

    data = self.runmodel.run(nodes, uncertain_params)

    s_oat = [(data.data[data.model_name].evaluations[i] - data.data[data.model_name].evaluations[i+1]) / (deltas[i] - deltas[i+1])
             for i in range(0, nr_params*nr_samples_per_param, 2)]

    data.data[data.model_name].s_oat = s_oat

    return data


def oat(self, **kwargs):
    """
    OAT, but for a larger number of samples -> obtaining sensitivity from Monte-Carlo
    """
    uncertain_params = self.convert_uncertain_parameters()
    nr_params = len(uncertain_params)
    nr_samples_per_param = round(kwargs['nr_samples_oat'] / len(uncertain_params))
    all_distributions = self.create_distribution()
    sample_means = [distr.mom([1])[0] for distr in all_distributions]

    nodes = np.tile(np.array(sample_means).reshape(len(sample_means), 1), len(uncertain_params)*nr_samples_per_param)

    for idx_param in range(len(uncertain_params)):
        nodes[idx_param, idx_param*nr_samples_per_param:(idx_param+1)*nr_samples_per_param] = all_distributions[idx_param].sample(nr_samples_per_param)

    data = self.runmodel.run(nodes, uncertain_params)

    variances = []
    for idx_param in range(len(uncertain_params)):
        variances.append(statistics.variance(data.data[data.model_name].evaluations[idx_param*nr_samples_per_param:(idx_param+1)*nr_samples_per_param]))

    total_variance = np.sum(variances)

    print('First Order Sobol Indices:', np.array(variances)/total_variance)
    data.data[data.model_name].sobol_first = np.array(variances)/total_variance

    return data


def morris_screening(self, **kwargs):
    """
    http://www.andreasaltelli.eu/file/repository/Screening_CPC_2011.pdf

    Morris = Local derivative at multiple points
        -> If the local derivatives are large -> large mean value -> large influence of the parameter
        -> If the local derivatives change a lot -> large std -> non-linear fie or interaction with other parameters
    """
    nr_of_repetitions = kwargs['nr_of_morris_repetitions']  # = Number of starting conditions from which OAT is done
    all_distributions = self.create_distribution()
    sample_means = [distr.mom([1])[0] for distr in all_distributions]

    uncertain_params = self.convert_uncertain_parameters()
    nr_params = len(uncertain_params)
    sampler = qmc.Sobol(d=nr_params)
    sample = sampler.random(n=nr_of_repetitions*2)
    # -> array with :nr_of_repetitions of the reference coordinates and nr_of_repetitions: values as disturbance

    # if linear disturbances are preferred, the nr_of_repetitions: samples are modified
    if kwargs['linear_disturbance'] is True:
        sample[nr_of_repetitions:] = sample[:nr_of_repetitions] + \
                                     kwargs['linear_disturbance_factor'] * sample[nr_of_repetitions:]

    # build the normalized_nodes array, which will has following format:
    # row 1: unmodified reference coordinates 1
    # row 2: disturbed reference coordinate 1, unmodified reference coordinates 1 for all other parameters
    # row 3: disturbed reference coordinate 2, unmodified reference coordinates 1 for all other parameters
    # ...
    # row nr_parameters+1: unmodified reference coordinates 2
    # row nr_parameters+2: disturbed reference coordinate 1, unmodified reference coordinates 2 for all other parameters
    # ...
    normalized_nodes = np.repeat(sample[:nr_of_repetitions], nr_params+1, axis=0)
    for repetition_idx, b_values in enumerate(sample[nr_of_repetitions:, :]):
        np.fill_diagonal(normalized_nodes[repetition_idx * (nr_params+1) + 1:
                                          (repetition_idx+1) * (nr_params+1), :], b_values)

    # scale the nodes with the input distributions
    normalized_nodes = normalized_nodes.T
    nodes = np.zeros(normalized_nodes.shape)
    distr_ranges = [[distr.lower[0], distr.upper[0]] for distr in all_distributions]
    for idx, distr_range in enumerate(distr_ranges):
        nodes[idx, :] = distr_range[0] + (distr_range[1]-distr_range[0]) * normalized_nodes[idx, :]

    # run the simulations
    data = self.runmodel.run(nodes, uncertain_params)

    ee = np.zeros((nr_params, nr_of_repetitions))
    evaluations = np.array(data.data[data.model_name].evaluations)

    for repetition in range(nr_of_repetitions):
        # evaluation difference between disturbed and reference computation
        y_delta = evaluations[repetition * (nr_params+1) + 1: (repetition+1) * (nr_params+1)] - evaluations[repetition * (nr_params+1)]
        # normalized input disturbance
        x_delta = normalized_nodes[:, repetition * (nr_params+1) + 1: (repetition+1) * (nr_params+1)] - normalized_nodes[:, [repetition * (nr_params+1)]]
        ee[:, repetition] = y_delta / x_delta.sum(axis=0)

    data.data[data.model_name].ee = ee
    data.data[data.model_name].ee_mean = np.mean(np.abs(ee), axis=1)
    data.data[data.model_name].ee_std = np.std(np.abs(ee), axis=1)

    fig, ax = plt.subplots()

    for idx in range(nr_params):
        print('MEAN {}: {}'.format(idx, data.data[data.model_name].ee_mean[idx]))
        print('STD {}: {}'.format(idx, data.data[data.model_name].ee_std[idx]))

        if idx < 10:
            marker = 'o'
        else:
            marker = '+'
        ax.plot(data.data[data.model_name].ee_mean[idx], data.data[data.model_name].ee_std[idx],
                marker=marker, label='Uncertain param #{}'.format(idx))

    ax.set_xlabel('Mean')
    ax.set_ylabel('Standard Dev.')
    ax.grid()
    ax.legend()
    plt.show()

    return data


if __name__ == '__main__':

    case = 'linear_fie'  # linear_fie, ishigami, morris_fie/morris_fie_fortran
    analysis_type = 'morris_screening'  # oat, oat_screening, morris_screening

    ##############################################
    # MODEL SETUP
    ##############################################
    if case == 'ishigami':
        # create the model
        model = un.Model(run=ishigami, labels=["x1", "x2", "x3"])

        # Create the distributions
        x1_dist = cp.Uniform(-np.pi, np.pi)
        x2_dist = cp.Uniform(-np.pi, np.pi)
        x3_dist = cp.Uniform(-np.pi, np.pi)

        # Define the parameter dictionary
        parameters = {"x1": x1_dist, "x2": x2_dist, "x3": x3_dist}

    elif case == 'morris_fie':
        # create the model
        model = un.Model(run=morris_fie, coefficients_morris=morris_init())

        parameters = dict()
        for i in range(1, 21):
            parameters['x{}'.format(i)] = cp.Uniform()

    elif case == 'morris_fie_fortran':
        # create the model
        model = un.Model(run=morris_fie_fortran)

        parameters = dict()
        for i in range(1, 21):
            parameters['x{}'.format(i)] = cp.Uniform()

    elif case == 'linear_fie':
        # create the model
        model = un.Model(run=linear_fie)

        # Define the parameter dictionary
        parameters = {"x1": cp.Uniform(), "x2": cp.Uniform(), "x3": cp.Uniform()}

    ##############################################
    # UQ SETUP
    ##############################################
    # Set up the uncertainty quantification
    if analysis_type == 'oat':
        UQ = un.UncertaintyQuantification(model=model, parameters=parameters, CPUs=None,
                                          custom_uncertainty_quantification=oat)
    elif analysis_type == 'oat_screening':
        UQ = un.UncertaintyQuantification(model=model, parameters=parameters, CPUs=None,
                                          custom_uncertainty_quantification=oat_screening)
    elif analysis_type == 'morris_screening':
        UQ = un.UncertaintyQuantification(model=model, parameters=parameters, CPUs=None,
                                          custom_uncertainty_quantification=morris_screening)

    ##############################################
    # UQ ANALYSIS
    ##############################################
    data = UQ.quantify(method='custom', nr_samples_oat=1000, nr_of_morris_repetitions=200, linear_disturbance=True,
                       linear_disturbance_factor=1E-9, seed=10)



    # Comparison of PCE / Monte-Carlo / OAT
    """
    mc_results = []
    pc_results = []
    oat_results = []

    actual_nr_mc_samples = []
    actual_nr_pc_samples = []
    actual_nr_oat_samples = []
    nr_samples_range = np.arange(10, 200, 20)
    for nr_samples in nr_samples_range:
        print('\nSobol indices according to Monte-Carlo:')
        input_mc_samples = nr_samples / (len(parameters) + 2) * 2  # -> (nr_mc_samples / 2) * (nr_uncertain_parameters + 2)
        data = UQ.quantify(method='mc', nr_mc_samples=input_mc_samples)
        actual_nr_mc_samples.append(len(data[0].data[data[0].model_name].evaluations))
        mc_results.append(data[0].data[data[0].model_name].sobol_first)

        print('\nSobol indices according to PCE:')
        data = UQ.quantify(method='pc', nr_collocation_nodes=nr_samples)
        actual_nr_pc_samples.append(len(data[0].data[data[0].model_name].evaluations))
        pc_results.append(data[0].data[data[0].model_name].sobol_first)

        print('\nSobol indices according to OAT:')
        data = UQ.quantify(method='custom', nr_samples_oat=nr_samples)
        actual_nr_oat_samples.append(len(data[0].data[data[0].model_name].evaluations))
        oat_results.append(data[0].data[data[0].model_name].sobol_first)

    fig, ax = plt.subplots()
    ax.plot(actual_nr_mc_samples, np.stack(mc_results), label='Monte-Carlo', linestyle='-', marker='o')
    ax.plot(actual_nr_pc_samples, np.stack(pc_results), label='PCE', linestyle='--', marker='o')
    ax.plot(actual_nr_oat_samples, np.stack(oat_results), label='OAT', linestyle='-.', marker='o')
    ax.grid()
    ax.legend()
    plt.show()
    """
