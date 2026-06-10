#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import copy
import math
import time
from collections import deque
from typing import Callable, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)


class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)
        # diffusion = DiffusionModel(config)
        # torch._dynamo.reset()
        # torch.set_float32_matmul_precision("high")
        # self.diffusion.conditional_sample = torch.compile(self.diffusion.conditional_sample)

        self.expected_cam_image_keys = [k for k in config.image_features if "cam" in k]

        if config.num_of_tac_encoders > 0:
            self.expected_tac_image_keys = [k for k in config.image_features if "tac" in k]
            if len(self.expected_tac_image_keys) % config.num_of_tac_encoders != 0:
                raise ValueError(
                    f"Number of tac encoders {config.num_of_tac_encoders} does not divide the number of tac images {len(self.expected_tac_image_keys)}"
                )
        else:
            self.expected_tac_image_keys = []

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        # if self.config.image_features:
        #     self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if len(self.expected_cam_image_keys) > 0:
            if self.config.use_separate_rgb_encoder_per_camera:
                for k in self.expected_cam_image_keys:
                    self._queues[k] = deque(maxlen=self.config.n_obs_steps)
            else:
                self._queues["observation.cam_images"] = deque(maxlen=self.config.n_obs_steps)
        if len(self.expected_tac_image_keys) > 0:
            if self.config.num_of_tac_encoders > 1:
                for i in range(self.config.num_of_tac_encoders):
                    self._queues[f'observation.tac_images.{i}'] = deque(maxlen=self.config.n_obs_steps)
            else:
                self._queues["observation.tac_images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        # if self.config.image_features:
        #     batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        #     batch["observation.images"] = torch.stack(
        #         [batch[key] for key in self.config.image_features], dim=-4
            # )
        if len(self.expected_cam_image_keys) > 0:
            if not self.config.use_separate_rgb_encoder_per_camera:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.cam_images"] = torch.stack([batch[k] for k in self.expected_cam_image_keys], dim=-4)
        if len(self.expected_tac_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            if self.config.num_of_tac_encoders > 1:
                for i in range(self.config.num_of_tac_encoders):
                    num = len(self.expected_tac_image_keys) // self.config.num_of_tac_encoders
                    batch[f'observation.tac_images.{i}'] = torch.stack([batch[k] for k in self.expected_tac_image_keys[i:i+num]], dim=-4)
            else:
                batch["observation.tac_images"] = torch.stack([batch[k] for k in self.expected_tac_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action
    
    # def make_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    #     batch = self.normalize_inputs(batch)
    #     # if self.config.image_features:
    #     #     batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
    #     #     batch["observation.images"] = torch.stack(
    #     #         [batch[key] for key in self.config.image_features], dim=-4
    #     #     )
    #     if len(self.expected_cam_image_keys) > 0:
    #         if not self.config.use_separate_rgb_encoder_per_camera:
    #             batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
    #             batch["observation.cam_images"] = torch.stack([batch[k] for k in self.expected_cam_image_keys], dim=-4)
    #     if len(self.expected_tac_image_keys) > 0:
    #         batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
    #         if self.config.num_of_tac_encoders > 1:
    #             for i in range(self.config.num_of_tac_encoders):
    #                 num = len(self.expected_tac_image_keys) // self.config.num_of_tac_encoders
    #                 batch[f'observation.tac_images.{i}'] = torch.stack([batch[k] for k in self.expected_tac_image_keys[i:i+num]], dim=-4)
    #         else:
    #             batch["observation.tac_images"] = torch.stack([batch[k] for k in self.expected_tac_image_keys], dim=-4)
    #     batch = self.normalize_targets(batch)
    #     return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        # if self.config.image_features:
        #     batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        #     batch["observation.images"] = torch.stack(
        #         [batch[key] for key in self.config.image_features], dim=-4
        #     )
        if len(self.expected_cam_image_keys) > 0:
            if not self.config.use_separate_rgb_encoder_per_camera:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch["observation.cam_images"] = torch.stack([batch[k] for k in self.expected_cam_image_keys], dim=-4)
        if len(self.expected_tac_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            if self.config.num_of_tac_encoders > 1:
                for i in range(self.config.num_of_tac_encoders):
                    num = len(self.expected_tac_image_keys) // self.config.num_of_tac_encoders
                    batch[f'observation.tac_images.{i}'] = torch.stack([batch[k] for k in self.expected_tac_image_keys[i:i+num]], dim=-4)
            else:
                batch["observation.tac_images"] = torch.stack([batch[k] for k in self.expected_tac_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        global_cond_dim = {}

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim["state"] = self.config.robot_state_feature.shape[0]
        self.expected_cam_image_keys = [k for k in config.image_features if "cam" in k]
        if config.num_of_tac_encoders > 0:
            self.expected_tac_image_keys = [k for k in config.image_features if "tac" in k]
        else:
            self.expected_tac_image_keys = []
        num_cam_images = len(self.expected_cam_image_keys)
        num_tac_images = len(self.expected_tac_image_keys)
        self._use_tac_images = False
        self._use_cam_images = False
        self.rgb_encoder_cam = nn.ModuleDict()
        self.rgb_encoder_tac = nn.ModuleDict()
        if  num_cam_images > 0:
            self._use_cam_images = True
            cam_dim = 0
            if self.config.use_separate_rgb_encoder_per_camera:
                for k in self.expected_cam_image_keys: 
                    encoder_name = k.replace("observation.images.", "")
                    self.rgb_encoder_cam[encoder_name] = DiffusionRgbEncoder(config, "cam")
                    cam_dim += self.rgb_encoder_cam[encoder_name].feature_dim
            else:
                self.rgb_encoder_cam['cam_share'] = DiffusionRgbEncoder(config, "cam")
                cam_dim += self.rgb_encoder_cam['cam_share'].feature_dim * num_cam_images
            global_cond_dim["cam"] = cam_dim
        if num_tac_images > 0:
            self._use_tac_images = True
            tac_dim = 0
            if config.num_of_tac_encoders > 1:
                for i in range(config.num_of_tac_encoders):
                    num = len(self.expected_tac_image_keys) // self.config.num_of_tac_encoders
                    self.rgb_encoder_tac[f'tac{i}'] = DiffusionRgbEncoder(config, "tac")
                    tac_dim += self.rgb_encoder_tac[f'tac{i}'].feature_dim * num
            else:
                self.rgb_encoder_tac['tac_share'] = DiffusionRgbEncoder(config, "tac")
                tac_dim += self.rgb_encoder_tac['tac_share'].feature_dim * num_tac_images
            global_cond_dim["tac"] = tac_dim
        # if self.config.image_features:
        #     num_images = len(self.config.image_features)
        #     if self.config.use_separate_rgb_encoder_per_camera:
        #         encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
        #         self.rgb_encoder = nn.ModuleList(encoders)
        #         global_cond_dim += encoders[0].feature_dim * num_images
        #     else:
        #         self.rgb_encoder = DiffusionRgbEncoder(config)
        #         global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim["env_state"] = self.config.env_state_feature.shape[0]

        total_cond_dim = sum(global_cond_dim.values())
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=total_cond_dim * config.n_obs_steps)
        # unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        # torch._dynamo.reset()
        # torch.set_float32_matmul_precision("high")
        # self.unet = torch.compile(unet)
        if not config.add_joint_to_transformer:
            # If we don't add the joint state to the transformer, we don't need to encode it.
            global_cond_dim.pop("state", None)
        self.z_encoder = Z_Encoder(d_model=512, global_cond = global_cond_dim, obs_steps = config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        begin_time = time.time()
        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        end_time = time.time()
        print(f"Diffusion sampling time: {end_time - begin_time:.4f} seconds")

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = [batch[OBS_ROBOT]]
        transformer_feats = {}
        if self.config.add_joint_to_transformer:
            transformer_feats["state"] = batch[OBS_ROBOT]
        # Extract image features.
        # if self.config.image_features:
        #     if self.config.use_separate_rgb_encoder_per_camera:
        #         # Combine batch and sequence dims while rearranging to make the camera index dimension first.
        #         images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        #         img_features_list = torch.cat(
        #             [
        #                 encoder(images)
        #                 for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
        #             ]
        #         )
        #         # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
        #         # feature dim (effectively concatenating the camera features).
        #         img_features = einops.rearrange(
        #             img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        #         )
        #     else:
        #         # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
        #         img_features = self.rgb_encoder(
        #             einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
        #         )
        #         # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
        #         # feature dim (effectively concatenating the camera features).
        #         img_features = einops.rearrange(
        #             img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        #         )
        #     global_cond_feats.append(img_features)
        if self._use_cam_images:
            cam_features: list | Tensor = []
            if self.config.use_separate_rgb_encoder_per_camera:
                for k in self.expected_cam_image_keys:
                    encoder_name = k.replace("observation.images.", "")
                    # print(f"{batch[k].shape},{batch[k]}")
                    cam_img_features = self.rgb_encoder_cam[encoder_name](
                        einops.rearrange(batch[k], "b s ... -> (b s) ...")
                    )
                    # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                    # feature dim (effectively concatenating the camera features).
                    cam_img_features = einops.rearrange(
                        cam_img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps
                    )
                    # global_cond_feats.append(cam_img_features)
                    cam_features.append(cam_img_features)
                cam_features = torch.cat(cam_features, dim=-1)
            else:
                cam_img_features = self.rgb_encoder_cam['cam_share'](
                    einops.rearrange(batch["observation.cam_images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                cam_img_features = einops.rearrange(
                    cam_img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
                # global_cond_feats.append(cam_img_features)
                cam_features = cam_img_features
            transformer_feats["cam"] = cam_features

        if self._use_tac_images:
            tac_features: list | Tensor = []
            if self.config.num_of_tac_encoders > 1:
                for i in range(self.config.num_of_tac_encoders):
                    tac_img_features = self.rgb_encoder_tac[f'tac{i}'](
                        einops.rearrange(batch[f'observation.tac_images.{i}'], "b s n ... -> (b s n) ...")
                    )
                    # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                    # feature dim (effectively concatenating the camera features).
                    tac_img_features = einops.rearrange(
                        tac_img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                    )
                    # global_cond_feats.append(tac_img_features)
                    tac_features.append(tac_img_features)
                tac_features = torch.cat(tac_features, dim=-1)
            else:
                tac_img_features = self.rgb_encoder_tac['tac_share'](
                    einops.rearrange(batch["observation.tac_images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                tac_img_features = einops.rearrange(
                    tac_img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
                # global_cond_feats.append(tac_img_features)
                tac_features = tac_img_features
            transformer_feats["tac"] = tac_features

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])

        z = self.z_encoder.encode(transformer_feats)

        # Concatenate features then flatten to (B, global_cond_dim).
        # return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
        global_cond = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
        return torch.cat([global_cond, z], dim=-1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        # Check for required image/state features
        features_keys = {"observation.images.head_cam", "observation.tac_images.0", "observation.tac_images", "observation.cam_images", "observation.environment_state"}
        assert any(key in batch for key in features_keys), "Missing required image or environment state data"
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        # print(f"{horizon},{self.config.horizon}")
        # print(f"{batch['action'].shape},{batch['observation.state'].shape},{batch['observation.tac_images'].shape},{batch['observation.images.head_cam'].shape}")
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each differemt timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
        # return pred, target


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig, source: str | list[str]):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None and "cam" in source:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        # if type(source) is list:
        #     image_keys = source
        #     self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # elif source == "tac_share":
        #     image_keys = [k for k in config.image_features if "tac" in k]
        #     self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # elif source == "cam_share":
        #     image_keys = [k for k in config.image_features if "cam" in k]
        # elif "cam" in source:
        #     image_keys = [source]
        if source == "tac":
            image_keys = [k for k in config.image_features if "tac" in k]
            self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.feature_dim = int(config.spatial_softmax_num_keypoints / 2)
        elif source == "cam":
            image_keys = [k for k in config.image_features if "cam" in k]
            self.feature_dim = config.spatial_softmax_num_keypoints * 4
        else:
            raise ValueError("Invalid type of image")
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        images_shape = config.image_features[image_key].shape
        # images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if self.do_crop == True else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        # self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        # print(x.shape)
        # print(x)
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet. (batch, horizon, action_dim)
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim) (batch, state+cam+tac)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep) #(batch, diffusion_step_embed_dim)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    
def build_encoder(d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1,
                   num_encoder_layers=4, normalize_before=False, activation="relu"):

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class Z_Encoder(nn.Module):
    def __init__(self, d_model: int = 256, global_cond: dict | None = None, obs_steps: int = 1):
        super().__init__()
        self.encoder = build_encoder(d_model)
        self.cls_embed = nn.Embedding(1, d_model)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1+len(global_cond)*obs_steps, d_model))
        if "state" in global_cond:
            self.joint_proj = nn.Linear(global_cond["state"], d_model)
            del global_cond["state"]
        if "cam" in global_cond:
            self.cam_proj = nn.Linear(global_cond["cam"], d_model)
        if "tac" in global_cond:
            self.tac_proj = nn.Linear(global_cond["tac"], d_model)
        total_dim = sum(global_cond.values())
        self.latent_z_dim = obs_steps * (total_dim)
        self.latent_proj = nn.Linear(d_model, self.latent_z_dim)

    def encode(self, features: dict[str, Tensor]) -> Tensor:
        bs = features["cam"].shape[0]
        features_embed = []
        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        features_embed.append(cls_embed)
        if "state" in features:
            features_embed.append(self.joint_proj(features["state"]))
        if "cam" in features:
            features_embed.append(self.cam_proj(features["cam"]))
        if "tac" in features:
            features_embed.append(self.tac_proj(features["tac"]))
        encoder_input = torch.cat(features_embed, dim=1)
        encoder_input = encoder_input.permute(1, 0, 2)
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)
        encoder_output = self.encoder(encoder_input, pos=pos_embed)
        z = self.latent_proj(encoder_output[0]) # (B, latent_z_dim)
        return z