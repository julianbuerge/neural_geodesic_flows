# This file is adapted from https://github.com/greydanus/hamiltonian-nn
# and is licensed under the Apache License, Version 2.0
# See the LICENSE file in this HNN/ directory.

# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski


def trained_HNN_prediction(initial_data_point, time_steps):

    import torch, time, sys
    import numpy as np
    import scipy.integrate
    solve_ivp = scipy.integrate.solve_ivp

    from applications.master_thesis.HNN.nn_models import (
        MLP
    )

    from applications.master_thesis.HNN.hnn import (
        HNN
    )

    from applications.master_thesis.HNN.utils import (
        L2_loss,
        to_pickle,
        from_pickle
    )

    from applications.master_thesis.HNN.data import (
        get_dataset,
        coords2state,
        get_orbit,
        random_config,
        potential_energy,
        kinetic_energy,
        total_energy
    )

    def get_args():
        return {'input_dim': 2*4, #two bodies, each with q_x, q_y, p_z, p_y
             'hidden_dim': 200,
             'learn_rate': 1e-3,
             'input_noise': 0.,
             'batch_size': 200,
             'nonlinearity': 'tanh',
             'total_steps': 1500,
             'field_type': 'solenoidal',
             'print_every': 200,
             'verbose': True,
             'name': '2body',
             'seed': 0,
             'save_dir': '',
             'fig_dir': ''}

    class ObjectView(object):
        def __init__(self, d): self.__dict__ = d

    def load_model(args, baseline=False):
        output_dim = args.input_dim if baseline else 2
        nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
        model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)

        case = 'baseline' if baseline else 'hnn'
        path = "applications/master_thesis/HNN/trained_hnn.tar".format(args.save_dir, args.name, case)
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)

        return model

    args = ObjectView(get_args())
    #base_model = load_model(args, baseline=True)
    hnn_model = load_model(args, baseline=False)


    def model_update(t, state, model):
        state = state.reshape(-1,5)

        deriv = np.zeros_like(state)
        np_x = state[:,1:] # drop mass
        np_x = np_x.T.flatten()[None, :]
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32)
        dx_hat = model.time_derivative(x)
        deriv[:,1:] = dx_hat.detach().data.numpy().reshape(4,2).T
        return deriv.reshape(-1)


    t_points = time_steps
    t_span = [0,2.3293172690763053]
    state =  np.zeros((2,5))
    #masses
    state[:,0] = 1
    #loctations
    state[0,1:3] = initial_data_point[0:2]
    state[1,1:3] = initial_data_point[2:4]
    #velocities
    state[0,3:5] = initial_data_point[4:6]
    state[1,3:5] = initial_data_point[6:8]

    orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span)

    update_fn = lambda t, y0: model_update(t, y0, hnn_model)
    hnn_orbit, settings = get_orbit(state, t_points=t_points, t_span=t_span, update_fn=update_fn)

    return hnn_orbit
