config = {
 'expname': 'cutbeef_explicit',
 'logdir': './logs/realdynamic',
 'device': 'cuda:0',

 # Run first for 1 step with data_downsample=4 to generate weights for ray importance sampling
 'data_downsample': 2,
 'data_dirs': ['data/dynerf/cut_roasted_beef'],
 'contract': False,
 'ndc': True,
 'ndc_far': 2.6,
 'near_scaling': 0.95,
 'isg': False,
 'isg_step': -1,
 'ist_step': 50000,  # 'ist_step': - 1 for linear decoder
 'keyframes': False,
 'scene_bbox': [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]],

 # Optimization settings
 'num_steps': 90005,
 'batch_size': 4096,
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0002,  # for linear encoder = .0001
 'plane_tv_weight_proposal_net': 0.0002,  # for linear encoder = .0001
 'time_smoothness_weight': 0.001,
 'time_smoothness_weight_proposal_net': 1e-05,
 # new added losses
 'L1_motion_loss': 0.00,
 "l1_time_planes_rev_grid": 0.0001,

 # Training settings
 'save_every': 30000,
 'valid_every': 90001,
 'save_outputs': True,
 'train_fp16': True,  # set it to False for disabling amp

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 150]},
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [256, 256, 256, 150]}
 ],

 # Model settings
 'disp_concat_features_across_scales': True,
 'model_concat_features_across_scales': True,
 'density_activation': 'trunc_exp',
 'linear_decoder': False,  # making it hybrid
 'linear_decoder_layers': 0,
 'multiscale_res': [1, 2, 4, 8],
 'disp_grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 32,
  'resolution': [64, 64, 64, 150]
 }],
 'model_grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 3,
  'output_coordinate_dim': 16,
  'resolution': [64, 64, 64]
 }],
}
