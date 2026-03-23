import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check


class R_MAPPO_LwF(R_MAPPO):
    """
    MAPPO trainer with Learning without Forgetting (LwF) distillation.

    Extends the standard R_MAPPO trainer by adding a distillation loss term
    that penalizes divergence from a frozen teacher model's action distribution.

    Total actor loss = PPO_loss + alpha * distillation_loss - entropy_coef * entropy

    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update (student).
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param lwf_alpha: (float) weight of the distillation loss term.
    :param lwf_loss_type: (str) 'l2' for logit matching or 'kl' for KL divergence.
    """

    def __init__(self, args, policy, device=torch.device("cpu"),
                 lwf_alpha=0.01, lwf_loss_type="l2", lwf_temperature=1.0,
                 track_logits=False):
        super().__init__(args, policy, device=device)
        self.lwf_alpha = lwf_alpha
        self.lwf_loss_type = lwf_loss_type
        self.lwf_temperature = lwf_temperature
        self.teacher_trainer = None
        self.track_logits = track_logits

        # Track distillation-specific metrics
        self.last_ppo_loss = 0.0
        self.last_distillation_loss = 0.0
        self.last_total_policy_loss = 0.0

        # Per-action logit tracking (populated when track_logits=True)
        self.last_teacher_logit_mean = None  # shape: (n_actions,)
        self.last_teacher_logit_std = None
        self.last_student_logit_mean = None
        self.last_student_logit_std = None

    def set_teacher(self, teacher_trainer):
        """
        Set the frozen teacher trainer for distillation.

        :param teacher_trainer: (R_MAPPO) trainer with frozen teacher policy.
        """
        self.teacher_trainer = teacher_trainer
        # Ensure teacher is in eval mode and frozen
        self.teacher_trainer.policy.actor.eval()
        self.teacher_trainer.policy.critic.eval()
        for p in self.teacher_trainer.policy.actor.parameters():
            p.requires_grad = False
        for p in self.teacher_trainer.policy.critic.parameters():
            p.requires_grad = False

    def _compute_distillation_loss(self, obs_batch, rnn_states_batch,
                                   masks_batch, available_actions_batch):
        """
        Compute distillation loss between student and teacher action distributions.

        Returns:
            distillation_loss: (torch.Tensor) scalar loss value.
        """
        # Student logits
        student_dist = self.policy.actor.get_logit_forward(
            obs_batch, rnn_states_batch, masks_batch, available_actions_batch
        )
        student_logits = student_dist.logits

        # Teacher logits (no gradient)
        with torch.no_grad():
            teacher_dist = self.teacher_trainer.policy.actor.get_logit_forward(
                obs_batch, rnn_states_batch, masks_batch, available_actions_batch
            )
            teacher_logits = teacher_dist.logits

        # Action availability mask: (batch, n_actions), True = available
        if available_actions_batch is not None:
            avail = check(available_actions_batch).to(**self.tpdv).bool()
        else:
            avail = torch.ones_like(teacher_logits, dtype=torch.bool)

        if self.track_logits:
            with torch.no_grad():
                # Per-action mean and std across batch, only over available observations
                counts = avail.sum(dim=0).float().clamp(min=1)
                t_masked = teacher_logits * avail
                s_masked = student_logits * avail
                t_mean = t_masked.sum(dim=0) / counts
                s_mean = s_masked.sum(dim=0) / counts
                t_var = ((teacher_logits - t_mean.unsqueeze(0)) ** 2 * avail).sum(dim=0) / counts
                s_var = ((student_logits - s_mean.unsqueeze(0)) ** 2 * avail).sum(dim=0) / counts
                self.last_teacher_logit_mean = t_mean.cpu()
                self.last_teacher_logit_std = t_var.sqrt().cpu()
                self.last_student_logit_mean = s_mean.cpu()
                self.last_student_logit_std = s_var.sqrt().cpu()

        if self.lwf_loss_type == "kl":
            # KL divergence: KL(teacher || student)
            # Masked actions have softmax ≈ 0, so they contribute nothing.
            T = self.lwf_temperature
            teacher_log_probs = torch.log_softmax(teacher_logits / T, dim=-1)
            student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
            teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
            kl = torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs),
                dim=-1, keepdim=True
            )
            # Scale by T² to maintain gradient magnitude
            return (T * T) * kl.mean()
        else:
            # L2 loss on mean-centered logits (default)
            # Center and normalize only over available actions so that L2
            # matches the large-T limit of T²·KL (which naturally ignores
            # masked actions via zero softmax probability).
            avail_f = avail.float()
            n_avail = avail_f.sum(dim=-1, keepdim=True).clamp(min=1)
            # Mean over available actions only
            teacher_mean = (teacher_logits * avail_f).sum(dim=-1, keepdim=True) / n_avail
            student_mean = (student_logits * avail_f).sum(dim=-1, keepdim=True) / n_avail
            teacher_centered = (teacher_logits - teacher_mean) * avail_f
            student_centered = (student_logits - student_mean) * avail_f
            diff = teacher_centered - student_centered
            return (0.5 / n_avail * torch.sum(diff ** 2, dim=-1, keepdim=True)).mean()

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks with optional distillation loss.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Forward pass for PPO
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch
        )

        # PPO actor loss
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # Distillation loss
        distillation_loss = torch.tensor(0.0, device=self.device)
        if update_actor and self.teacher_trainer is not None:
            distillation_loss = self._compute_distillation_loss(
                obs_batch, rnn_states_batch, masks_batch, available_actions_batch
            )

        # Combined policy loss
        policy_loss = policy_action_loss + self.lwf_alpha * distillation_loss

        # Track metrics
        self.last_ppo_loss = policy_action_loss.item()
        self.last_distillation_loss = distillation_loss.item()
        self.last_total_policy_loss = policy_loss.item()

        # Actor update
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            from onpolicy.utils.util import get_gard_norm
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # Critic update (unchanged from base)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            from onpolicy.utils.util import get_gard_norm
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD, with distillation metrics.

        Overrides base class to accumulate distillation-specific loss metrics
        across all minibatches and epochs.
        """
        import numpy as np
        from onpolicy.algorithms.utils.util import check

        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['clip_frac'] = 0
        train_info['ppo_loss'] = 0
        train_info['distillation_loss'] = 0
        train_info['total_policy_loss'] = 0

        # Accumulators for per-action logit tracking
        if self.track_logits:
            logit_accum = {
                'teacher_logit_mean': None,
                'teacher_logit_std': None,
                'student_logit_mean': None,
                'student_logit_std': None,
            }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm

                active_masks_batch = check(sample[8]).to(**self.tpdv)
                denom = active_masks_batch.sum() + 1e-8
                ratio_active = (imp_weights * active_masks_batch).sum() / denom
                train_info['ratio'] += ratio_active.item()

                clipped = (torch.abs(imp_weights - 1.0) > self.clip_param).float()
                clip_frac = (clipped * active_masks_batch).sum() / denom
                train_info['clip_frac'] += clip_frac.item()

                # Accumulate distillation metrics
                train_info['ppo_loss'] += self.last_ppo_loss
                train_info['distillation_loss'] += self.last_distillation_loss
                train_info['total_policy_loss'] += self.last_total_policy_loss

                # Accumulate per-action logit stats
                if self.track_logits and self.last_teacher_logit_mean is not None:
                    for key, attr in [
                        ('teacher_logit_mean', self.last_teacher_logit_mean),
                        ('teacher_logit_std', self.last_teacher_logit_std),
                        ('student_logit_mean', self.last_student_logit_mean),
                        ('student_logit_std', self.last_student_logit_std),
                    ]:
                        if logit_accum[key] is None:
                            logit_accum[key] = attr.clone()
                        else:
                            logit_accum[key] += attr

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        # Average and unpack per-action logit stats into train_info
        if self.track_logits and logit_accum['teacher_logit_mean'] is not None:
            n_actions = logit_accum['teacher_logit_mean'].shape[0]
            for key in logit_accum:
                logit_accum[key] /= num_updates
            for a in range(n_actions):
                train_info[f'teacher_logit_mean_action_{a}'] = logit_accum['teacher_logit_mean'][a].item()
                train_info[f'teacher_logit_std_action_{a}'] = logit_accum['teacher_logit_std'][a].item()
                train_info[f'student_logit_mean_action_{a}'] = logit_accum['student_logit_mean'][a].item()
                train_info[f'student_logit_std_action_{a}'] = logit_accum['student_logit_std'][a].item()

        return train_info
