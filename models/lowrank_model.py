from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from plenoxels_with_GB_feature_length_in_disp32.models.reverse_disp_grid import Rev_Disp_Field
from plenoxels_with_GB_feature_length_in_disp32.models.density_fields import KPlaneDensityField
from plenoxels_with_GB_feature_length_in_disp32.models.kplane_field import KPlaneField, disp_KPlaneField
from plenoxels_with_GB_feature_length_in_disp32.ops.activations import init_density_activation
from plenoxels_with_GB_feature_length_in_disp32.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from plenoxels_with_GB_feature_length_in_disp32.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels_with_GB_feature_length_in_disp32.utils.timer import CudaTimer


class LowrankModel(nn.Module):
    def __init__(self,
                 disp_grid_config: Union[str, List[Dict]],
                 model_grid_config :Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 disp_concat_features_across_scales: bool = False,
                 model_concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()

        self.disp_grid_config: List[Dict] = disp_grid_config
        self.model_grid_config = model_grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.disp_concat_features_across_scales = disp_concat_features_across_scales
        self.model_concat_features_across_scales = model_concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)
        self.disp_field = disp_KPlaneField(aabb,
                                           grid_config=self.disp_grid_config,
                                           concat_features_across_scales=self.disp_concat_features_across_scales,
                                           multiscale_res=self.multiscale_res,
                                           spatial_distortion=self.spatial_distortion,
                                           density_activation=self.density_act,
                                           linear_decoder=self.linear_decoder,
                                           )

        self.rev_grid = Rev_Disp_Field(aabb,
                                       grid_config=self.disp_grid_config,
                                       concat_features_across_scales=self.disp_concat_features_across_scales,
                                       multiscale_res=self.multiscale_res,
                                       spatial_distortion=self.spatial_distortion,
                                       density_activation=self.density_act,
                                       linear_decoder=self.linear_decoder,
                                       )
        self.field = KPlaneField(
            aabb,
            grid_config=self.model_grid_config,
            concat_features_across_scales=self.model_concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        self.disp_grid_used = 1
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_point_alpha_decomp(pts: torch.Tensor, weights: torch.Tensor):
        comp_rgb = torch.sum(weights * pts, dim=-2)
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, w2c: torch.Tensor, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):

        # # ADD IMAGE PLANE POINTS AND POSES IN THE OUT FOR MOTION SUPERVISION

        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)
        pts = ray_samples.get_positions()


        if self.disp_grid_used is not None:
            disp_out = self.disp_field(pts, timestamps=timestamps)  # output with timestamps
            disp_out = disp_out.reshape(pts.shape)
            timestamps1 = timestamps[:, None, None].expand(disp_out.shape)  # [n_rays, n_samples,1]
            disp = torch.where(timestamps1 != -1, disp_out, 0)
            # feeding the displacement only for non-zero timestamps
            field_out = self.field(pts + disp, ray_bundle.directions, timestamps=None)
        else:  # REMOVING THE DISPLACEMENT PART
            field_out = self.field(pts, timestamps=timestamps)
            disp_out = None

        # rev_disp = self.rev_grid(pts+disp, timestamps=timestamps)
        # rev_disp = rev_disp.reshape(pts.shape)

        rgb, density = field_out["rgb"], field_out["density"]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # point3d_rev_grid = self.render_point_alpha_decomp(pts=rev_disp, weights=weights)

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            # "3d_points_rev_grid": point3d_rev_grid,
            "canonical_time_points": disp,
            "pts": pts,
            # "rev_disp": rev_disp,

        }

        """
        POSES,CANONICAL VOLUME POINTS AT T==-1 ,3D POINTS FROM REVERSE GRID
        (FROM OBTAINED POINTS AT T==-1 TO GIVEN T REVERSE MAPPING),
        PROJECTED POINTS TO DIFFERENT CAMERA POSES,
        (.CSV FILES GIVING X1,Y1,X2,Y2)
        """

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        disp_model_params = self.disp_field.get_disp_params()
        rev_disp_param = self.rev_grid.get_rev_disp_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + rev_disp_param["rev_field"] + disp_model_params['field'] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + rev_disp_param["rev_nn"] + disp_model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]
