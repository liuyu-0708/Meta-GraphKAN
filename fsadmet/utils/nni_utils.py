

import nni


def update_cfg(cfg):
    # get trialID
    trial_id = nni.get_trial_id()
    # initialize the params
    optimized_params = nni.get_next_parameter()
    if optimized_params:
        # update the config before training
        for p in cfg.model.gnn:
            if p in optimized_params:
                cfg.model.gnn[p] = optimized_params[p]

        cfg.train.random_seed = optimized_params["random_seed"]
        cfg.train.meta_lr = optimized_params["meta_lr"]
        cfg.train.update_lr = optimized_params["update_lr"]
        cfg.train.update_step_test = optimized_params["update_step_test"]
        cfg.train.decay = optimized_params["decay"]
        cfg.meta.contrastive_weight = optimized_params["contrastive_weight"]
        cfg.logger.log_dir = "outputs_{}".format(trial_id)

    return cfg
