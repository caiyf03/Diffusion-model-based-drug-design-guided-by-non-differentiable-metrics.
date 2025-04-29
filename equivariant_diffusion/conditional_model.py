import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

import utils
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from analysis.molecule_builder import build_molecule, process_molecule
from analysis.metrics import MoleculeProperties
import torch.nn as nn
import torch.optim as optim

    

class ConditionalDDPM(EnVariationalDiffusion):
    """
    Conditional Diffusion Module.
    """
    print("we are in CDDPM")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.dynamics.update_pocket_coords
        # 实例化了调整网络，并定义了专用的优化器。
        self.adjust_net = self.AdjustNet(input_dim=13, hidden_dim=128)
        self.adjust_optimizer = optim.Adam(self.adjust_net.parameters(), lr=1e-3) #这行代码使用 Adam 优化器来管理和更新 self.adjust_net 的参数，学习率设为 1e-3。利用这个优化器根据梯度更新调整网络的参数。

    class AdjustNet(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, input_dim)
        
        def forward(self, zt, t, ligand_mask, pocket_mask):
            # 此处可以把 t 以及 mask 信息拼接到 zt 中，具体设计依实际需求调整
            # 简单示例：仅基于 zt 调整
            h = self.relu(self.fc1(zt))
            adjustment = self.fc2(h)
            return adjustment
        

    def kl_prior(self, xh_lig, mask_lig, num_nodes):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice
        negligible in the loss. However, you compute it so that you see it when
        you've made a mistake in your noise schedule.
        """
        batch_size = len(num_nodes)

        # Compute the last alpha value, alpha_T.
        ones = torch.ones((batch_size, 1), device=xh_lig.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh_lig)

        # Compute means.
        mu_T_lig = alpha_T[mask_lig] * xh_lig
        mu_T_lig_x, mu_T_lig_h = \
            mu_T_lig[:, :self.n_dims], mu_T_lig[:, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_lig_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_lig_h).squeeze()

        # Compute KL for h-part.
        zeros = torch.zeros_like(mu_T_lig_h)
        ones = torch.ones_like(sigma_T_h)
        mu_norm2 = self.sum_except_batch((mu_T_lig_h - zeros) ** 2, mask_lig)
        kl_distance_h = self.gaussian_KL(mu_norm2, sigma_T_h, ones, d=1)

        # Compute KL for x-part.
        zeros = torch.zeros_like(mu_T_lig_x)
        ones = torch.ones_like(sigma_T_x)
        mu_norm2 = self.sum_except_batch((mu_T_lig_x - zeros) ** 2, mask_lig)
        subspace_d = self.subspace_dimensionality(num_nodes)
        kl_distance_x = self.gaussian_KL(mu_norm2, sigma_T_x, ones, subspace_d)

        return kl_distance_x + kl_distance_h

    def log_pxh_given_z0_without_constants(self, ligand, z_0_lig, eps_lig,
                                           net_out_lig, gamma_0, epsilon=1e-10):

        # Discrete properties are predicted directly from z_t.
        z_h_lig = z_0_lig[:, self.n_dims:]

        # Take only part over x.
        eps_lig_x = eps_lig[:, :self.n_dims]
        net_lig_x = net_out_lig[:, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0_lig)
        sigma_0_cat = sigma_0 * self.norm_values[1]

        # Computes the error for the distribution
        # N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        squared_error = (eps_lig_x - net_lig_x) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            self.sum_except_batch(squared_error, ligand['mask'])
        )

        # Compute delta indicator masks.
        # un-normalize
        ligand_onehot = ligand['one_hot'] * self.norm_values[1] + self.norm_biases[1]

        estimated_ligand_onehot = z_h_lig * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded.
        centered_ligand_onehot = estimated_ligand_onehot - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0_cat[ligand['mask']])
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0_cat[ligand['mask']])
            + epsilon
        )

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1,
                                keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        # Select the log_prob of the current category using the onehot
        # representation.
        log_ph_given_z0_ligand = self.sum_except_batch(
            log_probabilities_ligand * ligand_onehot, ligand['mask'])

        return log_p_x_given_z0_without_constants_ligand, log_ph_given_z0_ligand
    @torch.no_grad()
    def sample_p_xh_given_z0(self, z0_lig, xh0_pocket, lig_mask, pocket_mask,
                             batch_size, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=z0_lig.device)
        gamma_0 = self.gamma(t_zeros) #计算时间步为 0 时的 gamma 值
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0)
        net_out_lig, _ = self.dynamics(
            z0_lig, xh0_pocket, t_zeros, lig_mask, pocket_mask)
        print(f"z0_lg = {z0_lig.shape}")
        # Compute mu for p(zs | zt).
        mu_x_lig = self.compute_x_pred(net_out_lig, z0_lig, gamma_0, lig_mask)
        xh_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_x_lig, xh0_pocket, sigma_x, lig_mask, pocket_mask, fix_noise)

        x_lig, h_lig = self.unnormalize(
            xh_lig[:, :self.n_dims], z0_lig[:, self.n_dims:])
        x_pocket, h_pocket = self.unnormalize(
            xh0_pocket[:, :self.n_dims], xh0_pocket[:, self.n_dims:])
        print(f"x_lig = {x_lig.shape}, h_lig = {h_lig.shape}")
        #################################################################################
        h_lig = F.one_hot(torch.argmax(h_lig, dim=1), self.atom_nf)#转换为 one-hot 编码形式，最大值赋值为一其他为0
        # h_pocket = F.one_hot(torch.argmax(h_pocket, dim=1), self.residue_nf)

        return x_lig, h_lig, x_pocket, h_pocket

    def sample_normal(self, *args):
        raise NotImplementedError("Has been replaced by sample_normal_zero_com()")

    def sample_normal_zero_com(self, mu_lig, xh0_pocket, sigma, lig_mask,
                               pocket_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        if fix_noise:
            # bs = 1 if fix_noise else mu.size(0)
            raise NotImplementedError("fix_noise option isn't implemented yet")

        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)

        out_lig = mu_lig + sigma[lig_mask] * eps_lig #生成配体样本

        # project to COM-free subspace 投影到质心为零的子空间
        #print(f"original is:{xh0_pocket}")
        xh_pocket = xh0_pocket.detach().clone()
        out_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(out_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)
        #print(f"now is:{xh_pocket}")
        return out_lig, xh_pocket

    def noised_representation(self, xh_lig, xh0_pocket, lig_mask, pocket_mask,
                              gamma_t):
        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, xh_lig)
        sigma_t = self.sigma(gamma_t, xh_lig)
        
        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps_lig = self.sample_gaussian(
            size=(len(lig_mask), self.n_dims + self.atom_nf),
            device=lig_mask.device)
        # print(f"[INFO]conditional_model.py_noised_representation: alpha_t shape: {alpha_t.shape}")
        # print(f"[INFO]conditional_model.py_noised_representation: lig_mask shape: {lig_mask.shape}")
        # print(f"[INFO]conditional_model.py_noised_representation: xh_lig shape: {xh_lig.shape}")
        # print(f"[INFO]conditional_model.py_noised_representation: sigma_t shape: {sigma_t.shape}")
        # print(f"[INFO]conditional_model.py_noised_representation: eps_lig shape: {eps_lig.shape}")
        '''
        lig_mask shape: torch.Size([376])
        xh_lig shape: torch.Size([376, 14])
        sigma_t shape: torch.Size([16, 1])
        eps_lig shape: torch.Size([376, 13])
        '''
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t_lig = alpha_t[lig_mask] * xh_lig + sigma_t[lig_mask] * eps_lig
        
        # project to COM-free subspace
        xh_pocket = xh0_pocket.detach().clone()
        z_t_lig[:, :self.n_dims], xh_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(z_t_lig[:, :self.n_dims],
                                   xh_pocket[:, :self.n_dims],
                                   lig_mask, pocket_mask)
        
        return z_t_lig, xh_pocket, eps_lig

    def log_pN(self, N_lig, N_pocket):
        """
        Prior on the sample size for computing
        log p(x,h,N) = log p(x,h|N) + log p(N), where log p(x,h|N) is the
        model's output
        Args:
            N: array of sample sizes
        Returns:
            log p(N)
        """
        log_pN = self.size_distribution.log_prob_n1_given_n2(N_lig, N_pocket)
        return log_pN

    def delta_log_px(self, num_nodes):
        return -self.subspace_dimensionality(num_nodes) * \
               np.log(self.norm_values[0])

    def forward(self, ligand, pocket, return_info=False):
        """
        Computes the loss and NLL terms
        """
        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)
        print(f"[INFO]: conditional_modules.py: This is for training, forwarding ...")
        # Likelihood change due to normalization
        # if self.vnode_idx is not None:
        #     delta_log_px = self.delta_log_px(ligand['size'] - ligand['num_virtual_atoms'] + pocket['size'])
        # else:
        delta_log_px = self.delta_log_px(ligand['size'])

        # Sample a timestep t for each example in batch
        # At evaluation time, loss_0 will be computed separately to decrease
        # variance in the estimator (costs two forward passes)
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float()
        s_int = t_int - 1  # previous timestep

        # Masks: important to compute log p(x | z0).
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), ligand['x'])
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation 
        # 根据 $\gamma_t$ 对 xh0_lig 和 xh0_pocket 进行加噪处理，得到加噪后的表示 z_t_lig 和噪声 eps_t_lig
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)

        # Neural net prediction. 神经网络预测噪声 epsilon_t
        net_out_lig, _ = self.dynamics(
            z_t_lig, xh_pocket, t, ligand['mask'], pocket['mask'])

        # For LJ loss term
        # xh_lig_hat does not need to be zero-centered as it is only used for
        # computing relative distances
        xh_lig_hat = self.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                                  ligand['mask'])

        # Compute the L2 error. 计算噪声的L2误差
        squared_error = (eps_t_lig - net_out_lig) ** 2
        if self.vnode_idx is not None:
            # coordinates of virtual atoms should not contribute to the error
            squared_error[ligand['one_hot'][:, self.vnode_idx].bool(), :self.n_dims] = 0
        error_t_lig = self.sum_except_batch(squared_error, ligand['mask']) # 对每个样本内部求和，得到每个样本的误差

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization
        SNR_weight = (1 - self.SNR(gamma_s - gamma_t)).squeeze(1)
        assert error_t_lig.size() == SNR_weight.size()

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(
            n_nodes=ligand['size'], device=error_t_lig.device)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1).
        # Should be close to zero.
        kl_prior = self.kl_prior(xh0_lig, ligand['mask'], ligand['size'])

        if self.training:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_t_lig, eps_t_lig, net_out_lig, gamma_t)

            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand * \
                              t_is_zero.squeeze()
            loss_0_h = -log_ph_given_z0 * t_is_zero.squeeze()

            # apply t_is_zero mask
            error_t_lig = error_t_lig * t_is_not_zero.squeeze()

        else:
            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), ligand['x'])

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            z_0_lig, xh_pocket, eps_0_lig = \
                self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                           pocket['mask'], gamma_0)

            net_out_0_lig, _ = self.dynamics(
                z_0_lig, xh_pocket, t_zeros, ligand['mask'], pocket['mask'])

            log_p_x_given_z0_without_constants_ligand, log_ph_given_z0 = \
                self.log_pxh_given_z0_without_constants(
                    ligand, z_0_lig, eps_0_lig, net_out_0_lig, gamma_0)
            loss_0_x_ligand = -log_p_x_given_z0_without_constants_ligand
            loss_0_h = -log_ph_given_z0

        # sample size prior
        log_pN = self.log_pN(ligand['size'], pocket['size'])

        info = {
            'eps_hat_lig_x': scatter_mean(
                net_out_lig[:, :self.n_dims].abs().mean(1), ligand['mask'],
                dim=0).mean(),
            'eps_hat_lig_h': scatter_mean(
                net_out_lig[:, self.n_dims:].abs().mean(1), ligand['mask'],
                dim=0).mean(),
        }
        loss_terms = (delta_log_px, error_t_lig, torch.tensor(0.0), SNR_weight,
                      loss_0_x_ligand, torch.tensor(0.0), loss_0_h,
                      neg_log_constants, kl_prior, log_pN,
                      t_int.squeeze(), xh_lig_hat)
        return (*loss_terms, info) if return_info else loss_terms
    
    def partially_noised_ligand(self, ligand, pocket, noising_steps):
        """
        Partially noises a ligand to be later denoised.
        """

        # Inflate timestep into an array
        t_int = torch.ones(size=(ligand['size'].size(0), 1),
            device=ligand['x'].device).float() * noising_steps

        # Normalize t to [0, 1].
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_t = self.inflate_batch_array(self.gamma(t), ligand['x'])

        # Concatenate x, and h[categorical].
        xh0_lig = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        # Center the input nodes
        xh0_lig[:, :self.n_dims], xh0_pocket[:, :self.n_dims] = \
            self.remove_mean_batch(xh0_lig[:, :self.n_dims],
                                   xh0_pocket[:, :self.n_dims],
                                   ligand['mask'], pocket['mask'])

        # Find noised representation
        z_t_lig, xh_pocket, eps_t_lig = \
            self.noised_representation(xh0_lig, xh0_pocket, ligand['mask'],
                                       pocket['mask'], gamma_t)
            
        return z_t_lig, xh_pocket, eps_t_lig

    def diversify(self, ligand, pocket, noising_steps):
        """
        Diversifies a set of ligands via noise-denoising
        """

        # Normalize data, take into account volume change in x.
        ligand, pocket = self.normalize(ligand, pocket)

        z_lig, xh_pocket, _ = self.partially_noised_ligand(ligand, pocket, noising_steps)

        timesteps = self.T
        n_samples = len(pocket['size'])
        device = pocket['x'].device

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = ligand['mask']

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.

        for s in reversed(range(0, noising_steps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            z_lig, xh_pocket, log_prob_adjust = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig.detach(), xh_pocket.detach(), lig_mask, pocket['mask'])

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Overwrite last frame with the resulting x and h.
        out_lig = torch.cat([x_lig, h_lig], dim=1)
        out_pocket = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig, out_pocket, lig_mask, pocket['mask']


    def xh_given_zt_and_epsilon(self, z_t, epsilon, gamma_t, batch_mask):
        """ Equation (7) in the EDM paper """
        alpha_t = self.alpha(gamma_t, z_t)
        sigma_t = self.sigma(gamma_t, z_t)
        xh = z_t / alpha_t[batch_mask] - epsilon * sigma_t[batch_mask] / \
             alpha_t[batch_mask]
        return xh
    @torch.no_grad()
    def my_to_x0(self, t, zt_lig,xh0_pocket,ligand_mask,pocket_mask,n_samples):
        eps_t_lig, _ = self.dynamics(
                zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)
        gamma_t = self.gamma(t)
        log_alpha2_t = F.logsigmoid(-gamma_t)
        alpha_t = torch.exp(0.5 * log_alpha2_t)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)
        z0_lig = (zt_lig - sigma_t[ligand_mask]*eps_t_lig) / alpha_t[ligand_mask]
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z0_lig, xh0_pocket, ligand_mask, pocket_mask, n_samples
        )
        return x_lig, h_lig, x_pocket, h_pocket
    
    def sample_p_zt_given_zs(self, zs_lig, xh0_pocket, ligand_mask, pocket_mask,
                             gamma_t, gamma_s, fix_noise=False):
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zs_lig)

        mu_lig = alpha_t_given_s[ligand_mask] * zs_lig
        # 投影到质心为零的空间
        zt_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma_t_given_s, ligand_mask, pocket_mask,
            fix_noise)

        return zt_lig, xh0_pocket

    def sample_p_zs_given_zt(self, s, t, zt_lig, xh0_pocket, ligand_mask,
                             pocket_mask,optimize, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_lig)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_lig)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_lig)

        # Neural net prediction. 输出预测的噪声 eps_t_lig
        # 调用 dynamics 预测噪声，但因为 dynamics 参数冻结，可以使用 no_grad
        '''
        PyTorch 会记录下整个前向传播的计算图和所有中间激活值，以便后续反向传播计算梯度。这些中间信息包括：
        每一层的输入、输出以及中间激活值；
        各层的操作（如矩阵乘法、激活函数等）的梯度信息；
        计算过程中用于链式法则反向传播的所有缓存数据。'
        '''
        with torch.no_grad():
            eps_t_lig, _ = self.dynamics(
                zt_lig, xh0_pocket, t, ligand_mask, pocket_mask)
        # print(eps_t_lig.shape) torch.Size([388, 13])
        #############################################################################
        # 调用调整网络 f₍φ₎ 来计算噪声调整项（例如，可以基于 zt_lig 和 t 作为输入）
        adjustment = self.adjust_net(zt_lig, t, ligand_mask, pocket_mask)
        kl_div = torch.mean((adjustment - eps_t_lig) ** 2)
        log_prob_adjust = -0.5 * kl_div
        # print(f"[INFO]conditional_model.py: we add the adjustment to the noise: {adjustment}")
        # 生成调整后的噪声：即原始噪声加上调整项
        if (optimize == 1):
            eps_t_lig = eps_t_lig + adjustment
        # 记录 log 概率（例如：可以近似使用 -0.5 * ||adjustment||^2 作为 log 概率）
        #log_prob_adjust = -0.5 * torch.sum(adjustment ** 2, dim=-1)
        # 你可以将 log_prob_adjust 保存下来，用于后续的策略梯度更新
        #############################################################################

        # Compute mu for p(zs | zt).
        # Note: mu_{t->s} = 1 / alpha_{t|s} z_t - sigma_{t|s}^2 / sigma_t / alpha_{t|s} epsilon
        # follows from the definition of mu_{t->s} and Equ. (7) in the EDM paper
        mu_lig = zt_lig / alpha_t_given_s[ligand_mask] - \
                 (sigma2_t_given_s / alpha_t_given_s / sigma_t)[ligand_mask] * \
                 eps_t_lig

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        zs_lig, xh0_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand_mask, pocket_mask, fix_noise)
        # 检查是否质心漂移
        self.assert_mean_zero_with_mask(zt_lig[:, :self.n_dims], ligand_mask)
        # 及时删除中间变量，帮助释放内存
        del gamma_s, gamma_t, sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s, sigma_s, sigma_t, eps_t_lig, adjustment
        torch.cuda.empty_cache()

        return zs_lig, xh0_pocket, log_prob_adjust

    def sample_combined_position_feature_noise(self, lig_indices, xh0_pocket,
                                               pocket_indices):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise
        for z_h.
        """
        raise NotImplementedError("Use sample_normal_zero_com() instead.")

    def sample(self, *args):
        raise NotImplementedError("Conditional model does not support sampling "
                                  "without given pocket.")
    
    #此函数可以在step中间强制生成配体分子，并存放分子文件于try
    def my_in_test(self,index,xh_lig, xh_pocket, lig_mask, pocket_mask, pocket_com_before,dataset_info,sanitize,relax_iter,largest_frag,pdb_id):
        x_dims = 3
        #print(f"[INFO] con_mol.py: we are in the try.func with {index}")
        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :x_dims], pocket_mask, dim=0)

        xh_pocket[:, :x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        x = xh_lig[:, :x_dims].detach().cpu() # 提取xh_lig 中的原子坐标部分。 表示选取 xh_lig 中所有行（第一个 :），以及从第 0 列到第 self.x_dims - 1 列（第二个 :self.x_dims）的元素。
        atom_type = xh_lig[:, x_dims:].argmax(1).detach().cpu() # atom_type 提取了 xh_lig 中的原子类型信息，通过 argmax(1) 找到每个原子类型的独热编码中最大值的索引，即原子类型的编号。
        lig_mask = lig_mask.cpu()

        molecules = []
        #  将批量的原子坐标和原子类型数据根据 lig_mask 分割成单个分子的数据。组合成一个元组 mol_pc
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)
            ###########################################
            if pdb_id == None:
                print("No name, something wrong!")
            folder_path = f'DiffSBDD/example/try/{pdb_id}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = f'{folder_path}/try_{index}.sdf'
            utils.write_sdf_file(file_path, molecules)
        print(f"[INFO] con_mol.py: success write t={index} mol to {file_path}")

    def my_reward_function(self, molecules):
        """
        计算生成的分子的奖励。
        x_lig: 配体的坐标
        h_lig: 配体的特征
        x_pocket: 口袋的坐标
        h_pocket: 口袋的特征
        lig_mask: 配体的掩码
        pocket_mask: 口袋的掩码
        """
        # 示例：奖励为配体与口袋之间的负距离（越小越好）
        mol_metrics = MoleculeProperties()
        print(f"[INFO]con_mo.py: we are in the my_reward_func")
        all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = mol_metrics.evaluate(molecules)
        qed_flattened = [x for px in all_qed for x in px]
        sa_flattened = [x for px in all_sa for x in px]
        logp_flattened = [x for px in all_logp for x in px]
        lipinski_flattened = [x for px in all_lipinski for x in px]
        print(f"[INFO]con_mo.py: length is {len(qed_flattened)}")
        reward = (
            sum(qed_flattened)*6/len(qed_flattened) +  # QED 值越高越好
            sum(sa_flattened)*2/len(qed_flattened) +   # SA 值越高越好
            #sum(logp_flattened) - # LogP 值需要适中，这里假设越高越好
            sum(lipinski_flattened)/5/len(qed_flattened)  # 符合 Lipinski 规则的分子越多越好
        )
        print(f"[INFO]con_mo.py: the reward is {reward}")
        return reward  # 负距离作为奖励
    def my_reward_for_SVDD(self, molecules):
        """
        计算生成的分子的奖励。
        x_lig: 配体的坐标
        h_lig: 配体的特征
        x_pocket: 口袋的坐标
        h_pocket: 口袋的特征
        lig_mask: 配体的掩码
        pocket_mask: 口袋的掩码
        """
        # 示例：奖励为配体与口袋之间的负距离（越小越好）
        mol_metrics = MoleculeProperties()
        all_qed, all_sa, all_logp, all_lipinski = mol_metrics.evaluate_new(molecules)
        print(f"[INFO]con_mo.py: we are in the my_reward_SVDD:{len(all_qed[0])}")
        qed_flattened = [x for px in all_qed for x in px]
        sa_flattened = [x for px in all_sa for x in px]
        logp_flattened = [x for px in all_logp for x in px]
        lipinski_flattened = [x for px in all_lipinski for x in px]
        reward = []
        def sigmoid(z: float) -> float:
            return 1.0 / (1.0 + math.exp(-z))
        k = 20
        for i in range(len(qed_flattened)):
            the_reward = (
                qed_flattened[i]*2 +  # QED 值越高越好
                sa_flattened[i]*2 +   # SA 值越高越好
                sigmoid(k * (logp_flattened[i] + 1)) * sigmoid(-k * (logp_flattened[i] - 5)) + # LogP 值需要适中，这里假设越高越好
                lipinski_flattened[i]/5  # 符合 Lipinski 规则的分子越多越好
            )
            reward.append(the_reward)
        #print(f"[INFO]con_mo.py: the reward is {reward}")
        return reward  # 负距离作为奖励
    def compute_policy_gradient(self, trajectories, reward):
        """
        计算策略梯度。
        trajectories: 轨迹列表，每个元素是 (state, action) 对。
        reward: 最终奖励。
        """
        log_probs = []
        for state, action in trajectories:
            z_lig, xh_pocket, s_array, t_array, lig_mask, pocket_mask = state
            # 计算 log p_theta(x_{t-1} | x_t, c)
            log_prob = -torch.norm(z_lig - action, dim=-1).mean()  # 假设噪声服从高斯分布
            log_probs.append(log_prob)
        # 将 log_probs 转换为张量
        log_probs = torch.stack(log_probs)  # 形状为 [轨迹长度, 样本数]
        
        # 将 reward 扩展为与 log_probs 相同的形状
        reward_expanded = torch.full_like(log_probs, reward)  # 将 reward 扩展为与 log_probs 相同的形状
        
        # 计算策略梯度
        policy_gradient = (reward_expanded * log_probs).mean()  # 逐点相乘后取均值
        
        return policy_gradient
    
    def update_model(self, policy_gradient):
        """
        使用策略梯度更新模型参数。
        policy_gradient: 计算得到的策略梯度。
        """
        self.optimizer.zero_grad()  # 清除梯度
        policy_gradient.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
        self.optimizer.step()  # 更新参数

    def save_checkpoint(self, optimizer, filename):
        import os
        import torch
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 1) 获取整个 ConditionalDDPM 的 state_dict()
        full_state = self.state_dict()  
        # 其中 "adjust_net.fc1.weight", "adjust_net.fc1.bias", ... 都在 full_state 里

        # 2) 只筛选出 adjust_net 的部分参数
        new_adjust_state = {}
        for key, value in full_state.items():
            # 如果你只想保存 adjust_net
            if key.startswith("adjust_net."):
                # 不再加任何 'ddpm.' 前缀，直接使用原键名
                new_adjust_state[key] = value

        # 3) 整理出要保存的字典
        flat_checkpoint = {}
        flat_checkpoint.update(new_adjust_state)
        flat_checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # 4) 最后用 torch.save 保存到文件
        torch.save(flat_checkpoint, filename)
        print(f"Checkpoint saved to {filename}")


    def load_checkpoint(self, optimizer, filename):
        import torch
        #print(f"===> [DEBUG] Entering load_checkpoint with file = {filename}")
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        #print("===> [DEBUG] Checkpoint keys:", list(checkpoint.keys()))
        self.load_state_dict(checkpoint, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ############################  SPSA_begin  ########################################
    def my_perturbation_for_molecule(self,coords_part, zeta,t,zt_lig, ligand_mask, pocket_mask,):
        # 从标准正态分布采样，确保噪声在同一设备和数据类型下
        noise = torch.randn(coords_part.shape, device=coords_part.device, dtype=coords_part.dtype)
        if True == False:
            adjustment = self.adjust_net(zt_lig, t, ligand_mask, pocket_mask)
            noise = adjustment[:, :noise.shape[1]]
        # 计算每个分子中所有原子的扰动均值 (形状为 [1, 3])
        noise_mean = torch.mean(noise, dim=0, keepdim=True)
        # 对扰动进行中心化，确保均值为0
        noise_centered = noise - noise_mean
        # 缩放扰动
        perturbation = zeta * noise_centered
        return perturbation

    def my_gradient_for_molecule(self,the_perturbations,z_lig_p_0,z_lig_m_0, xh_pocket_p_0,xh_pocket_m_0, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag, zeta):
        mol_plus = self.handle_to_mol(z_lig_p_0, xh_pocket_p_0, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag)
        mol_minus = self.handle_to_mol(z_lig_m_0, xh_pocket_m_0, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag)
        # 分别计算扰动后 reward
        f_plus = self.my_reward_for_SPSA(mol_plus)
        f_minus = self.my_reward_for_SPSA(mol_minus)
        if isinstance(f_plus, list):
            f_plus = torch.tensor(f_plus, device=the_perturbations.device, dtype=the_perturbations.dtype)
        if isinstance(f_minus, list):
            f_minus = torch.tensor(f_minus, device=the_perturbations.device, dtype=the_perturbations.dtype)
        # 计算方向导数（标量），即沿扰动方向的变化率
        directional_derivative = (f_plus - f_minus) / (2 * zeta)
        # 利用 lig_mask 将每个分子的标量复制到该分子所有原子上
        # 假设 lig_mask 是一个长为 N (361) 的张量，值为分子ID（0~19）
        directional_derivative = directional_derivative[lig_mask]  # 形状 (361,)
        directional_derivative = directional_derivative.unsqueeze(1)  # 形状 (361, 1)
        # 得到每个原子扰动的方向（单位向量）
        norm = torch.norm(the_perturbations, dim=1, keepdim=True) + 1e-8
        unit_vector = the_perturbations / norm
        # 最终梯度估计为方向导数乘以单位扰动向量
        grad_est = directional_derivative * the_perturbations
        return grad_est
    def my_update_z_lig(self, z_lig, xh_pocket, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag,t_array,n_samples, zeta, guidance_scale=1e-2):
        z_lig_updated_plus = z_lig.clone()
        z_lig_updated_minus = z_lig.clone()
        z_lig_final = z_lig.clone()
        k = 10
        the_grads = []
        for i in range (k):
            # 初始化一个张量，用于存储全批次的扰动（仅坐标部分），形状 (N, 3)
            all_perturbations = torch.empty_like(z_lig[:, :3])
            unique_ids = torch.unique(lig_mask)
            #对每个样本进行加扰动
            for uid in unique_ids:
                # 找到属于该分子的所有原子索引
                indices = (lig_mask == uid).nonzero(as_tuple=True)[0]
                # 提取该分子坐标部分 (n,3)
                coords_part = z_lig[indices, :3]
                h_part = z_lig[indices, 3:]
                # 生成扰动
                perturbation = self.my_perturbation_for_molecule(coords_part, zeta,t_array,z_lig,lig_mask, pocket_mask)
                # 将当前分子的扰动写入全批次扰动张量
                all_perturbations[indices] = perturbation
                coords_plus = coords_part + perturbation
                coords_minus = coords_part - perturbation
                # 生成分子
                z_plus = torch.cat((coords_plus, h_part), dim=1)
                z_minus = torch.cat((coords_minus, h_part), dim=1)
                # 赋值
                z_lig_updated_plus[indices] = z_plus
                z_lig_updated_minus[indices] = z_minus

            z_lig_p_0, h_lig_p_0, x_pocket_p_0, h_pocket_p_0 = self.my_to_x0(t_array, z_lig_updated_plus, xh_pocket, lig_mask,pocket_mask,n_samples)
            xh_pocket_p_0 = torch.cat([x_pocket_p_0, h_pocket_p_0], dim=1)
            z_lig_p_0 = torch.cat([z_lig_p_0, h_lig_p_0], dim=1)

            z_lig_m_0, h_lig_m_0, x_pocket_m_0, h_pocket_m_0 = self.my_to_x0(t_array, z_lig_updated_minus, xh_pocket, lig_mask,pocket_mask,n_samples)
            xh_pocket_m_0 = torch.cat([x_pocket_m_0, h_pocket_m_0], dim=1)
            z_lig_m_0 = torch.cat([z_lig_m_0, h_lig_m_0], dim=1)

            # 计算该分子的梯度估计（仅对坐标部分）
            grad_est = self.my_gradient_for_molecule(all_perturbations,z_lig_p_0,z_lig_m_0, xh_pocket_p_0,xh_pocket_m_0, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag, zeta=1e-4)
            the_grads.append(grad_est)
        avg_grad = sum(the_grads) / len(the_grads)
        # 根据梯度上升更新坐标，使 reward 增加
        # 注意：如果目标是最大化 reward，则直接加上 guidance_scale * grad_est
        new_coords = z_lig[:, :3] + guidance_scale * avg_grad
        z_lig_updated_final = z_lig.clone()
        z_lig_updated_final[:, :3] = new_coords
        # 为新lig生成对应的pocket
        xh_pocket_new = xh_pocket.detach().clone()
        z_lig_updated_final[:, :3], xh_pocket_new[:, :3] = \
            self.remove_mean_batch(z_lig_updated_final[:, :3],
                                   xh_pocket_new[:, :3],
                                   lig_mask, pocket_mask)
        return z_lig_updated_final,xh_pocket_new
        
    ############################# SPSA_end  ##########################################
    def my_reward_for_SPSA(self, molecules):
        """
        计算生成的分子的奖励。
        x_lig: 配体的坐标
        h_lig: 配体的特征
        x_pocket: 口袋的坐标
        h_pocket: 口袋的特征
        lig_mask: 配体的掩码
        pocket_mask: 口袋的掩码
        """
        # 示例：奖励为配体与口袋之间的负距离（越小越好）
        mol_metrics = MoleculeProperties()
        all_qed, all_sa, all_logp, all_lipinski = mol_metrics.evaluate_new(molecules)
        print(f"[INFO]con_mo.py: we are in the my_reward_SVDD:{len(all_qed[0])}")
        qed_flattened = [x for px in all_qed for x in px]
        sa_flattened = [x for px in all_sa for x in px]
        logp_flattened = [x for px in all_logp for x in px]
        lipinski_flattened = [x for px in all_lipinski for x in px]
        reward = []
        for i in range(len(qed_flattened)):
            the_reward = (
                qed_flattened[i]*2 +  # QED 值越高越好
                sa_flattened[i]*3 +   # SA 值越高越好
                #logp_flattened[i] - # LogP 值需要适中，这里假设越高越好
                lipinski_flattened[i]/5  # 符合 Lipinski 规则的分子越多越好
            )
            reward.append(the_reward)
        return reward 
    
    def handle_to_mol(self,xh_lig, xh_pocket, lig_mask, pocket_mask, pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag):
        x_dims = 3
         # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, :x_dims], pocket_mask, dim=0)
        print(f"pocket_mask is :{pocket_mask.shape}")
        print(f"pocket_com_before is :{pocket_com_before}")
        print(f"pocket_com_after is :{pocket_com_after.shape}")

        #print(f"(pocket_com_before - pocket_com_after):{(pocket_com_before - pocket_com_after)}")
        #print(f"(pocket_com_before - pocket_com_after)[pocket_mask]:{(pocket_com_before - pocket_com_after)[pocket_mask].shape}")
        
        print(f"xh_pocket[:, :x_dims] :{xh_pocket[:, :x_dims].shape}")
        print(f"xh_lig[:, :x_dims] is :{xh_lig[:, :x_dims].shape}")
        xh_pocket[:, :x_dims] += \
            (pocket_com_before - pocket_com_after)[pocket_mask]
        xh_lig[:, :x_dims] += \
            (pocket_com_before - pocket_com_after)[lig_mask]

        # Build mol objects
        x = xh_lig[:, :x_dims].detach().cpu() # 提取xh_lig 中的原子坐标部分。 表示选取 xh_lig 中所有行（第一个 :），以及从第 0 列到第 self.x_dims - 1 列（第二个 :self.x_dims）的元素。
        atom_type = xh_lig[:, x_dims:].argmax(1).detach().cpu() # atom_type 提取了 xh_lig 中的原子类型信息，通过 argmax(1) 找到每个原子类型的独热编码中最大值的索引，即原子类型的编号。
        lig_mask = lig_mask.cpu()

        molecules = []
        #  将批量的原子坐标和原子类型数据根据 lig_mask 分割成单个分子的数据。组合成一个元组 mol_pc
        for mol_pc in zip(utils.batch_to_list(x, lig_mask),
                          utils.batch_to_list(atom_type, lig_mask)):

            mol = build_molecule(*mol_pc, dataset_info, add_coords=True)
            mol = process_molecule(mol,
                                   add_hydrogens=False,
                                   sanitize=sanitize,
                                   relax_iter=relax_iter,
                                   largest_frag=largest_frag)
            if mol is not None:
                molecules.append(mol)
        return [molecules]
    ####################################################################

    # @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, 
                            pocket_com_before,dataset_info,sanitize,relax_iter,largest_frag,pdb_id,device,optimize,path,path_save,svdd,spsa,return_frames=1,timesteps=None):
        """
        Draw samples from the generative model. Optionally, return intermediate
        states for visualization purposes.
        """
        #print(torch.cuda.memory_summary(device=device, abbreviated=False))
        if optimize == 1:
            print("[INFO] con_mol.py: we will optimize the noise")
        else:
            print(f"[INFO] con_mol.py: we will not optimize the noise")
        if svdd == 1:
            print("[INFO] we will use the SVDD optimizer")
        else:
            print(f"[INFO] we will not use the SVDD optimizer")
        if spsa == 1:
            print("[INFO] we will use the SPSA optimizer")
        else:
            print(f"[INFO] we will not use the SPSA optimizer")
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        print(f"[INFO] con_mol.py: the timesteps is {timesteps}")
        # print(f"[INFO] con_mol.py: the return_frame is {return_frames}, so it has middle data?")
        # is 1
        n_samples = len(pocket['size'])
        device = pocket['x'].device
        #print(pocket)
        _, pocket = self.normalize(pocket=pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        # lig_mask（一维，0...19）
        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)
        # Sample from Normal distribution in the pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0) #计算口袋的质心作为配体坐标的均值。
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)#初始化配体原子类型的均值为零。
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]#将坐标和原子类型的均值拼接在一起。
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)#初始化配体的标准差为 1。
        # xh_pocket 是给定蛋白质口袋坐标变换后的值，z_lig从均值为 mu_lig、标准差为 sigma 的正态分布中采样
        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask'])
        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        #初始化用于存储中间结果的输出张量
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)
        ############################################################
        # 用于存储每一步调整网络计算的 log 概率（用于 RL 更新）
        total_log_prob_adjust = 0.0
        count = 0
        ############################################################
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        #with torch.no_grad():
        for s in reversed(range(0, timesteps)):
            #with torch.no_grad():
            with torch.no_grad():
                s_array = torch.full((n_samples, 1), fill_value=s,
                                    device=z_lig.device) #创建一个新的张量，其所有元素都被填充为指定的值
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps
            if (s%100== 0):
                # print(f"z_lig 的形状: {z_lig.shape}")  # torch.Size([404, 13])
                # print(f"xh_pocket 的形状: {xh_pocket.shape}")  # torch.Size([5720, 13])
                print("looping with s = ",s)
            z_lig, xh_pocket,log_prob_adjust = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'],optimize)
            #print(z_lig[0])tensor([-0.4570, -1.5915,  2.0042, -0.0874, -0.4227, -1.4779, -2.2353, -0.0502...])
            # 累加 log_prob_adjust：这里假设 log_prob_adjust 是标量或者其 sum 后是标量
            total_log_prob_adjust += log_prob_adjust.sum()
            count += log_prob_adjust.numel()
            # # save frame 记录特定时间步的中间状态
            # with torch.no_grad():
            #     if (s * return_frames) % timesteps == 0:
            #         idx = (s * return_frames) // timesteps
            #         out_lig[idx], out_pocket[idx] = \
            #             self.unnormalize_z(z_lig, xh_pocket)
            ##########################################################
            # # 生成一次完整的配体分子
            # #if (s % 100 == 0 or s == (timesteps-1)):
            #     if (s % 100 == 0):
            #         x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)
            #         out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
            #         out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)
            #         self.my_in_test(s,out_lig.squeeze(0),out_pocket.squeeze(0),lig_mask,pocket['mask'],pocket_com_before,dataset_info,sanitize,relax_iter,largest_frag,pdb_id)
            # ##########################################################
            # 删除不再需要的变量并清理缓存
            # --------------SVDD:  核心改动：每隔10步就生成5组，然后根据reward挑选前20 ----------------
            if s <= 250 and s % 30 == 0 and svdd == 1 and False:
                print(f"[DEBUG] Doing multi-sample at step s={s}")
                # 1) 把当前 (z_lig, xh_pocket) 视作第1组
                multi_z_lig_list = [z_lig]
                multi_xh_pocket_list = [xh_pocket]
                multi_mask_list = [lig_mask]  # mask 也要对应复制
                multi_pocket_mask_list = [pocket['mask'].clone()]  # 原本 pocket_mask
                # 2) 再多采4组(总计5组)
                num_extra = 4
                for i in range(num_extra):
                    # 可以先复制 z_lig, xh_pocket，然后再采一次
                    z_tmp, xh_tmp, _ = self.sample_p_zs_given_zt(
                        s_array, t_array,
                        z_lig.clone(), xh_pocket.clone(),
                        lig_mask.clone(), pocket['mask'].clone(),
                        optimize
                    )
                    multi_z_lig_list.append(z_tmp)
                    multi_xh_pocket_list.append(xh_tmp)
                    multi_mask_list.append(lig_mask.clone())
                    multi_pocket_mask_list.append(pocket['mask'].clone())
                # 3) 合并这 5 组 (z_lig, xh_pocket)，共5 * 20 = 100分子
                #   如果 z_lig shape = [N_lig, 13], xh_pocket shape=[N_pocket,13],
                #   则 5份合并后 shape = [5*N_lig, 13] / [5*N_pocket, 13]
                big_z_lig = torch.cat(multi_z_lig_list, dim=0)
                big_xh_pocket = torch.cat(multi_xh_pocket_list, dim=0)
                big_mask = torch.cat(multi_mask_list, dim=0)
                big_pmask = torch.cat(multi_pocket_mask_list, dim=0)    # shape ~ [5*N_pocket]
                print(f"big_z_lig: {big_z_lig.shape}")
                print(f"big_xh_pocket: {big_xh_pocket.shape}")
                print(f"big_mask: {big_mask.shape}")
                print(f"big_pmask: {big_pmask.shape}")
                # (4) 对 pocket_mask 做 offset, 让 scatter_mean 分出 5 组 batch
                chunk_size = xh_pocket.shape[0]  # 原单份 pocket 节点数
                print(f"chunk_size: {chunk_size}")
                for i in range(1, 5):
                    start_idx = i * chunk_size
                    end_idx = (i+1) * chunk_size
                    # 每份 pocket_mask 全加 i => batch区分
                    big_pmask[start_idx:end_idx] += i * 20
                #  此时 big_mask 的取值范围还都是 [0..19], 并没区分 5 组
                #  我们需要把“前面N_lig个节点归属0..19, 中间N_lig个节点归属20..39”... 
                offset_size = z_lig.shape[0]  # N_lig
                for i in range(1, len(multi_mask_list)):
                    start_idx = i * offset_size
                    end_idx = (i+1) * offset_size
                    # 给 big_mask[start_idx:end_idx] + i*20
                    big_mask[start_idx:end_idx] += i * 20
                big_pocket_com_before = pocket_com_before.repeat(5, 1)
                pocket_com_before_saved = pocket_com_before  # 先存原
                pocket_com_before = big_pocket_com_before    # 临时替换
                print(f"pocket_com_before: {pocket_com_before.shape}")
                # 3) 对 big_z_lig, big_xh_pocket 做 handle_to_mol => 5*20个分子
                #    并计算 reward
                with torch.no_grad():
                    molecules_100 = self.handle_to_mol(
                        big_z_lig, big_xh_pocket,
                        big_mask,  # 这里已区分 0..99
                        big_pmask, pocket_com_before,
                        dataset_info, sanitize, relax_iter, largest_frag
                    )
                    pocket_com_before = pocket_com_before_saved

                    rewards_100 = self.my_reward_for_SVDD(molecules_100)
                    rewards_100 = torch.tensor(rewards_100, device=z_lig.device)
                # 4) 按rewards选 top20
                _, top_idx = rewards_100.topk(k=20, largest=True)
                # 5) 根据 top_idx 重建 z_lig, xh_pocket, lig_mask
                # 只保留 top20 分子的节点
                #   => 先构造 new_z_lig_list / new_mask_list
                new_z_lig_list = []
                new_xh_pocket_list = []
                new_mask_list = []
                for rank, idx_f in enumerate(top_idx):
                    # idx_f.item() in [0..99]
                    # 找到 big_mask==idx_f
                    node_mask = (big_mask == idx_f.item()) # 配子分子的掩码
                    nodep_mask = (big_pmask == idx_f.item()) # 蛋白质口袋的掩码
                    # 取出 z_lig, xh_pocket
                    z_sub = big_z_lig[node_mask]
                    xh_sub = big_xh_pocket[nodep_mask]
                    # 但要把 mask 改到 rank
                    # 让 new_mask=rank
                    sub_mask = torch.full_like(node_mask, fill_value=rank, dtype=big_mask.dtype)
                    sub_mask = sub_mask[node_mask]  # 只剩那些 True
                    # 分别存起来
                    new_z_lig_list.append(z_sub)
                    new_xh_pocket_list.append(xh_sub)
                    new_mask_list.append(sub_mask)
                
                # 拼接 => 得到新的 [N_lig', 13], 其中 N_lig' = 所有 top20 分子节点之和
                z_lig = torch.cat(new_z_lig_list, dim=0)
                xh_pocket = torch.cat(new_xh_pocket_list, dim=0)
                lig_mask = torch.cat(new_mask_list, dim=0)
                print(f"[DEBUG] selected top20 => new z_lig.shape={z_lig.shape}, new lig_mask range={lig_mask.shape}, s={s}")
                # 新 z_lig 保证零质心
                #z_lig, xh_pocket = self.remove_mean_batch(z_lig, xh_pocket, lig_mask, pocket['mask'])

                x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                print("complete SPSA")
                xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                z_lig = torch.cat([z_p_lig, h_p_lig], dim=1)
                self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

            if s <=50 and s % 10 == 0 and svdd == 1 and True:
                if s > 200:
                    # 短程 Lookahead 方法
                    for i in range (0,(s-100)):
                        origion_lig = z_lig
                        origion_pocket = xh_pocket
                        z_lig_ahead, xh_pocket_ahead,log_prob_adjust = self.sample_p_zs_given_zt(
                        s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'],optimize)
                        
                print(f"[DEBUG] Doing multi-sample at step s={s}")
                z_lig_0, h_lig_0, x_pocket_0, h_pocket_0 = self.my_to_x0(t_array, z_lig, xh_pocket,lig_mask,pocket['mask'],n_samples)
                xh_pocket_0 = torch.cat([x_pocket_0, h_pocket_0], dim=1)
                z_lig_0 = torch.cat([z_lig_0, h_lig_0], dim=1)
                # 1) 把当前 (z_lig, xh_pocket) 视作第1组
                multi_z_lig_list_0 = [z_lig_0]
                multi_xh_pocket_list_0 = [xh_pocket_0]
                multi_mask_list_0 = [lig_mask]  # mask 也要对应复制
                multi_pocket_mask_list_0 = [pocket['mask'].clone()]  # 原本 pocket_mask

                multi_z_lig_list = [z_lig]
                multi_xh_pocket_list = [xh_pocket]
                multi_mask_list = [lig_mask]  # mask 也要对应复制
                multi_pocket_mask_list = [pocket['mask'].clone()]  # 原本 pocket_mask
                # 2) 再多采4组(总计5组)
                num_extra = 4
                for i in range(num_extra):
                    # 可以先复制 z_lig, xh_pocket，然后再采一次
                    z_tmp, xh_tmp, _ = self.sample_p_zs_given_zt(
                        s_array, t_array,
                        z_lig.clone(), xh_pocket.clone(),
                        lig_mask.clone(), pocket['mask'].clone(),
                        optimize
                    )
                    z_temp_0, h_temp_0, x_pocket_temp_0, h_pocket_temp_0 = self.my_to_x0(t_array, z_tmp, xh_tmp,lig_mask,pocket['mask'],n_samples)
                    xh_pocket_0 = torch.cat([x_pocket_temp_0, h_pocket_temp_0], dim=1)
                    z_temp_0 = torch.cat([z_temp_0, h_temp_0], dim=1)
                    multi_z_lig_list_0.append(z_temp_0)
                    multi_xh_pocket_list_0.append(xh_pocket_0)
                    multi_mask_list_0.append(lig_mask.clone())
                    multi_pocket_mask_list_0.append(pocket['mask'].clone())

                    multi_z_lig_list.append(z_tmp)
                    multi_xh_pocket_list.append(xh_tmp)
                    multi_mask_list.append(lig_mask.clone())
                    multi_pocket_mask_list.append(pocket['mask'].clone())
                # 3) 合并这 5 组 (z_lig, xh_pocket)，共5 * 20 = 100分子
                #   如果 z_lig shape = [N_lig, 13], xh_pocket shape=[N_pocket,13],
                #   则 5份合并后 shape = [5*N_lig, 13] / [5*N_pocket, 13]
                big_z_lig_0 = torch.cat(multi_z_lig_list_0, dim=0)
                big_xh_pocket_0 = torch.cat(multi_xh_pocket_list_0, dim=0)
                big_mask_0 = torch.cat(multi_mask_list_0, dim=0)
                big_pmask_0 = torch.cat(multi_pocket_mask_list_0, dim=0)    # shape ~ [5*N_pocket]

                big_z_lig = torch.cat(multi_z_lig_list, dim=0)
                big_xh_pocket = torch.cat(multi_xh_pocket_list, dim=0)
                big_mask = torch.cat(multi_mask_list, dim=0)
                big_pmask = torch.cat(multi_pocket_mask_list, dim=0)    # shape ~ [5*N_pocket]

                # print(f"big_z_lig: {big_z_lig.shape}")
                # print(f"big_xh_pocket: {big_xh_pocket.shape}")
                # print(f"big_mask: {big_mask.shape}")
                # print(f"big_pmask: {big_pmask.shape}")
                # (4) 对 pocket_mask 做 offset, 让 scatter_mean 分出 5 组 batch
                chunk_size = xh_pocket.shape[0]  # 原单份 pocket 节点数
                print(f"chunk_size: {chunk_size}")
                for i in range(1, 5):
                    start_idx = i * chunk_size
                    end_idx = (i+1) * chunk_size
                    # 每份 pocket_mask 全加 i => batch区分
                    big_pmask_0[start_idx:end_idx] += i * 20
                for i in range(1, 5):
                    start_idx = i * chunk_size
                    end_idx = (i+1) * chunk_size
                    # 每份 pocket_mask 全加 i => batch区分
                    big_pmask[start_idx:end_idx] += i * 20
                #  此时 big_mask 的取值范围还都是 [0..19], 并没区分 5 组
                #  我们需要把“前面N_lig个节点归属0..19, 中间N_lig个节点归属20..39”... 
                offset_size = z_lig.shape[0]  # N_lig
                for i in range(1, len(multi_mask_list)):
                    start_idx = i * offset_size
                    end_idx = (i+1) * offset_size
                    # 给 big_mask[start_idx:end_idx] + i*20
                    big_mask_0[start_idx:end_idx] += i * 20
                for i in range(1, len(multi_mask_list)):
                    start_idx = i * offset_size
                    end_idx = (i+1) * offset_size
                    # 给 big_mask[start_idx:end_idx] + i*20
                    big_mask[start_idx:end_idx] += i * 20
                big_pocket_com_before = pocket_com_before.repeat(5, 1)
                pocket_com_before_saved = pocket_com_before  # 先存原
                pocket_com_before = big_pocket_com_before    # 临时替换
                print(f"pocket_com_before: {pocket_com_before.shape}")
                # 3) 对 big_z_lig_0, big_xh_pocket_0 做 handle_to_mol => 5*20个分子
                #    并计算 reward
                with torch.no_grad():
                    molecules_100_0 = self.handle_to_mol(
                        big_z_lig_0, big_xh_pocket_0,
                        big_mask_0,  # 这里已区分 0..99
                        big_pmask_0, pocket_com_before,
                        dataset_info, sanitize, relax_iter, largest_frag
                    )
                    rewards_100_0 = self.my_reward_for_SVDD(molecules_100_0)
                    rewards_100_0 = torch.tensor(rewards_100_0, device=z_lig.device)
                
                # 3) 对 big_z_lig, big_xh_pocket 做 handle_to_mol => 5*20个分子
                #    并计算 reward
                with torch.no_grad():
                    molecules_100 = self.handle_to_mol(
                        big_z_lig, big_xh_pocket,
                        big_mask,  # 这里已区分 0..99
                        big_pmask, pocket_com_before,
                        dataset_info, sanitize, relax_iter, largest_frag
                    )
                    pocket_com_before = pocket_com_before_saved

                    rewards_100 = self.my_reward_for_SVDD(molecules_100)
                    rewards_100 = torch.tensor(rewards_100, device=z_lig.device)
                # 4) 将两种reward混合
                mixed_reward = rewards_100_0*(s/250) + rewards_100*(250-s/250)
                # 4) 按rewards选 top20
                _, top_idx = mixed_reward.topk(k=20, largest=True)
                # 5) 根据 top_idx 重建 z_lig, xh_pocket, lig_mask
                # 只保留 top20 分子的节点
                #   => 先构造 new_z_lig_list / new_mask_list
                new_z_lig_list = []
                new_xh_pocket_list = []
                new_mask_list = []
                for rank, idx_f in enumerate(top_idx):
                    # idx_f.item() in [0..99]
                    # 找到 big_mask==idx_f
                    node_mask = (big_mask == idx_f.item()) # 配子分子的掩码
                    nodep_mask = (big_pmask == idx_f.item()) # 蛋白质口袋的掩码
                    # 取出 z_lig, xh_pocket
                    z_sub = big_z_lig[node_mask]
                    xh_sub = big_xh_pocket[nodep_mask]
                    # 但要把 mask 改到 rank
                    # 让 new_mask=rank
                    sub_mask = torch.full_like(node_mask, fill_value=rank, dtype=big_mask.dtype)
                    sub_mask = sub_mask[node_mask]  # 只剩那些 True
                    # 分别存起来
                    new_z_lig_list.append(z_sub)
                    new_xh_pocket_list.append(xh_sub)
                    new_mask_list.append(sub_mask)
                
                # 拼接 => 得到新的 [N_lig', 13], 其中 N_lig' = 所有 top20 分子节点之和
                z_lig = torch.cat(new_z_lig_list, dim=0)
                xh_pocket = torch.cat(new_xh_pocket_list, dim=0)
                lig_mask = torch.cat(new_mask_list, dim=0)
                print(f"[DEBUG] selected top20 => new z_lig.shape={z_lig.shape}, new lig_mask range={lig_mask.shape}, s={s}")
                # 新 z_lig 保证零质心
                x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                print("complete SPSA")
                xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                z_lig = torch.cat([z_p_lig, h_p_lig], dim=1)
                self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)
            # --------------SPSA -------------------------------------------------------------------
            if s <= 30 and s % 2 == 0 and spsa == 1:
                zeta_0 = 1e-3
                zeta = zeta_0 * (s/500)
                print(f'zeta is',zeta)
                guidance_scale = 1e-3
                # z_lig_0, h_lig_0, x_pocket_0, h_pocket_0 = self.my_to_x0(t_array, z_lig, xh0_pocket,lig_mask,pocket['mask'],n_samples)
                # xh_pocket_0 = torch.cat([x_pocket_0, h_pocket_0], dim=1)
                # z_lig_0 = torch.cat([z_lig_0, h_lig_0], dim=1)
                # 得到扰动后的z_lig
                z_lig, xh_pocket = self.my_update_z_lig(z_lig,xh_pocket, lig_mask,pocket['mask'], pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag,t_array,n_samples, zeta, guidance_scale=guidance_scale)
                x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                print("complete SPSA")
                xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                z_lig = torch.cat([z_p_lig, h_p_lig], dim=1)
                self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

                if s == 30:
                    z_lig_0, h_lig_0, x_pocket_0, h_pocket_0 = self.my_to_x0(t_array, z_lig, xh_pocket,lig_mask,pocket['mask'],n_samples)
                    xh_pocket_0 = torch.cat([x_pocket_0, h_pocket_0], dim=1)
                    z_lig_0 = torch.cat([z_lig_0, h_lig_0], dim=1)
                    # 1) 把当前 (z_lig, xh_pocket) 视作第1组
                    multi_z_lig_list_0 = [z_lig_0]
                    multi_xh_pocket_list_0 = [xh_pocket_0]
                    multi_mask_list_0 = [lig_mask]  # mask 也要对应复制
                    multi_pocket_mask_list_0 = [pocket['mask'].clone()]  # 原本 pocket_mask

                    multi_z_lig_list = [z_lig]
                    multi_xh_pocket_list = [xh_pocket]
                    multi_mask_list = [lig_mask]  # mask 也要对应复制
                    multi_pocket_mask_list = [pocket['mask'].clone()]  # 原本 pocket_mask
                    # 2) 再多采4组(总计5组)
                    num_extra = 4
                    for i in range(num_extra):
                        # 可以先复制 z_lig, xh_pocket，然后再采一次
                        z_tmp, xh_tmp, _ = self.sample_p_zs_given_zt(
                            s_array, t_array,
                            z_lig.clone(), xh_pocket.clone(),
                            lig_mask.clone(), pocket['mask'].clone(),
                            optimize
                        )
                        if i>=2:
                            zeta = 1e-3
                        z_lig, xh_pocket = self.my_update_z_lig(z_tmp,xh_tmp, lig_mask,pocket['mask'], pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag,t_array,n_samples, zeta, guidance_scale=guidance_scale)
                        x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                        z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                        z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                        print("complete SPSA")
                        xh_tmp = torch.cat([x_pocket, h_pocket], dim=1)
                        z_tmp = torch.cat([z_p_lig, h_p_lig], dim=1)

                        z_temp_0, h_temp_0, x_pocket_temp_0, h_pocket_temp_0 = self.my_to_x0(t_array, z_tmp, xh_tmp,lig_mask,pocket['mask'],n_samples)
                        xh_pocket_0 = torch.cat([x_pocket_temp_0, h_pocket_temp_0], dim=1)
                        z_temp_0 = torch.cat([z_temp_0, h_temp_0], dim=1)
                        multi_z_lig_list_0.append(z_temp_0)
                        multi_xh_pocket_list_0.append(xh_pocket_0)
                        multi_mask_list_0.append(lig_mask.clone())
                        multi_pocket_mask_list_0.append(pocket['mask'].clone())

                        multi_z_lig_list.append(z_tmp)
                        multi_xh_pocket_list.append(xh_tmp)
                        multi_mask_list.append(lig_mask.clone())
                        multi_pocket_mask_list.append(pocket['mask'].clone())
                    # 3) 合并这 5 组 (z_lig, xh_pocket)，共5 * 20 = 100分子
                    #   如果 z_lig shape = [N_lig, 13], xh_pocket shape=[N_pocket,13],
                    #   则 5份合并后 shape = [5*N_lig, 13] / [5*N_pocket, 13]
                    big_z_lig_0 = torch.cat(multi_z_lig_list_0, dim=0)
                    big_xh_pocket_0 = torch.cat(multi_xh_pocket_list_0, dim=0)
                    big_mask_0 = torch.cat(multi_mask_list_0, dim=0)
                    big_pmask_0 = torch.cat(multi_pocket_mask_list_0, dim=0)    # shape ~ [5*N_pocket]

                    big_z_lig = torch.cat(multi_z_lig_list, dim=0)
                    big_xh_pocket = torch.cat(multi_xh_pocket_list, dim=0)
                    big_mask = torch.cat(multi_mask_list, dim=0)
                    big_pmask = torch.cat(multi_pocket_mask_list, dim=0)    # shape ~ [5*N_pocket]

                    # print(f"big_z_lig: {big_z_lig.shape}")
                    # print(f"big_xh_pocket: {big_xh_pocket.shape}")
                    # print(f"big_mask: {big_mask.shape}")
                    # print(f"big_pmask: {big_pmask.shape}")
                    # (4) 对 pocket_mask 做 offset, 让 scatter_mean 分出 5 组 batch
                    chunk_size = xh_pocket.shape[0]  # 原单份 pocket 节点数
                    print(f"chunk_size: {chunk_size}")
                    for i in range(1, 5):
                        start_idx = i * chunk_size
                        end_idx = (i+1) * chunk_size
                        # 每份 pocket_mask 全加 i => batch区分
                        big_pmask_0[start_idx:end_idx] += i * 20
                    for i in range(1, 5):
                        start_idx = i * chunk_size
                        end_idx = (i+1) * chunk_size
                        # 每份 pocket_mask 全加 i => batch区分
                        big_pmask[start_idx:end_idx] += i * 20
                    #  此时 big_mask 的取值范围还都是 [0..19], 并没区分 5 组
                    #  我们需要把“前面N_lig个节点归属0..19, 中间N_lig个节点归属20..39”... 
                    offset_size = z_lig.shape[0]  # N_lig
                    for i in range(1, len(multi_mask_list)):
                        start_idx = i * offset_size
                        end_idx = (i+1) * offset_size
                        # 给 big_mask[start_idx:end_idx] + i*20
                        big_mask_0[start_idx:end_idx] += i * 20
                    for i in range(1, len(multi_mask_list)):
                        start_idx = i * offset_size
                        end_idx = (i+1) * offset_size
                        # 给 big_mask[start_idx:end_idx] + i*20
                        big_mask[start_idx:end_idx] += i * 20
                    big_pocket_com_before = pocket_com_before.repeat(5, 1)
                    pocket_com_before_saved = pocket_com_before  # 先存原
                    pocket_com_before = big_pocket_com_before    # 临时替换
                    print(f"pocket_com_before: {pocket_com_before.shape}")
                    # 3) 对 big_z_lig_0, big_xh_pocket_0 做 handle_to_mol => 5*20个分子
                    #    并计算 reward
                    with torch.no_grad():
                        molecules_100_0 = self.handle_to_mol(
                            big_z_lig_0, big_xh_pocket_0,
                            big_mask_0,  # 这里已区分 0..99
                            big_pmask_0, pocket_com_before,
                            dataset_info, sanitize, relax_iter, largest_frag
                        )
                        rewards_100_0 = self.my_reward_for_SVDD(molecules_100_0)
                        rewards_100_0 = torch.tensor(rewards_100_0, device=z_lig.device)
                    
                    # 3) 对 big_z_lig, big_xh_pocket 做 handle_to_mol => 5*20个分子
                    #    并计算 reward
                    with torch.no_grad():
                        molecules_100 = self.handle_to_mol(
                            big_z_lig, big_xh_pocket,
                            big_mask,  # 这里已区分 0..99
                            big_pmask, pocket_com_before,
                            dataset_info, sanitize, relax_iter, largest_frag
                        )
                        pocket_com_before = pocket_com_before_saved

                        rewards_100 = self.my_reward_for_SVDD(molecules_100)
                        rewards_100 = torch.tensor(rewards_100, device=z_lig.device)
                    # 4) 将两种reward混合
                    mixed_reward = rewards_100_0*(s/250) + rewards_100*(250-s/250)
                    # 4) 按rewards选 top20
                    _, top_idx = mixed_reward.topk(k=20, largest=True)
                    # 5) 根据 top_idx 重建 z_lig, xh_pocket, lig_mask
                    # 只保留 top20 分子的节点
                    #   => 先构造 new_z_lig_list / new_mask_list
                    new_z_lig_list = []
                    new_xh_pocket_list = []
                    new_mask_list = []
                    for rank, idx_f in enumerate(top_idx):
                        # idx_f.item() in [0..99]
                        # 找到 big_mask==idx_f
                        node_mask = (big_mask == idx_f.item()) # 配子分子的掩码
                        nodep_mask = (big_pmask == idx_f.item()) # 蛋白质口袋的掩码
                        # 取出 z_lig, xh_pocket
                        z_sub = big_z_lig[node_mask]
                        xh_sub = big_xh_pocket[nodep_mask]
                        # 但要把 mask 改到 rank
                        # 让 new_mask=rank
                        sub_mask = torch.full_like(node_mask, fill_value=rank, dtype=big_mask.dtype)
                        sub_mask = sub_mask[node_mask]  # 只剩那些 True
                        # 分别存起来
                        new_z_lig_list.append(z_sub)
                        new_xh_pocket_list.append(xh_sub)
                        new_mask_list.append(sub_mask)
                    
                    # 拼接 => 得到新的 [N_lig', 13], 其中 N_lig' = 所有 top20 分子节点之和
                    z_lig = torch.cat(new_z_lig_list, dim=0)
                    xh_pocket = torch.cat(new_xh_pocket_list, dim=0)
                    lig_mask = torch.cat(new_mask_list, dim=0)
                    print(f"[DEBUG] selected top20 => new z_lig.shape={z_lig.shape}, new lig_mask range={lig_mask.shape}, s={s}")
                    # 新 z_lig 保证零质心
                    x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                    z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                    z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                    print("complete SPSA")
                    xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                    z_lig = torch.cat([z_p_lig, h_p_lig], dim=1)
                    self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

            # -------------- 继续原逻辑 --------------
            del s_array, t_array, log_prob_adjust
            torch.cuda.empty_cache()
        # 计算平均 log_prob_adjust（标量）
        # print(f"z_lig 的形状: {z_lig.shape}")#torch.Size([445, 3])
        avg_log_prob_adjust = total_log_prob_adjust / count
        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)
        #print(f"x_lig 的形状是: {x_lig.shape}而h_lig的形状是:{h_lig.shape}")#torch.Size([445, 3])
        self.assert_mean_zero_with_mask(x_lig, lig_mask)
        # Correct CoM drift for examples without intermediate states 校正质心漂移
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                print(f'Warning CoG drift with error {max_cog:.3f}. Projecting '
                    f'the positions down.')
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket['mask'])
        # Overwrite last frame with the resulting x and h.
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)
        ################################################################## 
        with torch.no_grad():
            my_molecules = self.handle_to_mol(out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, pocket['mask'], pocket_com_before,dataset_info,sanitize,relax_iter,largest_frag)
            my_reward = self.my_reward_function(my_molecules)# 对所有时间步中记录的 log_prob_adjust 取平均
        def print_adjust_net_gradients():
            total_norm = 0.0
            print("[INFO] AdjustNet gradients:")
            for name, param in self.adjust_net.named_parameters():
                if param.grad is not None:
                    norm_val = param.grad.data.norm(2)
                    total_norm += norm_val.item() ** 2
                    print(f"  {name}: {norm_val.item():.4f}")
                else:
                    print(f"  {name}: no grad")
            total_norm = total_norm ** 0.5
            #print(f"Total AdjustNet gradient norm: {total_norm:.4f}")
            return total_norm
        with torch.enable_grad():
            # REINFORCE 损失：负 reward 乘以总的 log 概率
            if (path_save and optimize == 1):
                print(f"the reward is {my_reward}")
                scale_factor = 0.1
                loss_rl =  my_reward * avg_log_prob_adjust * scale_factor 
                self.adjust_optimizer.zero_grad()
                loss_rl.backward()
                #########################监控################################
                grad_norm = print_adjust_net_gradients()
                print(f"norm is {grad_norm}")
                # 如果梯度范数异常大，可能需要梯度裁剪
                if grad_norm > 10:
                    print("[WARNING] AdjustNet gradient norm is high!")
                #############################################################
                self.adjust_optimizer.step()
                print("RL update loss:", loss_rl.item())
                #filename = "DiffSBDD/RL_check_point/adjust_checkpoint.pth"
                filename = os.path.join("DiffSBDD/RL_check_point/", path_save)
                self.save_checkpoint(self.adjust_optimizer, filename)
                print(f"[INFO]con_mo.py: The checkpoint for optimizing the noise is saved to {filename}")
            else:
                print("No RL update")
        #######################################################################
        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask']
    
    @torch.no_grad()
    def inpaint(self, ligand, pocket, lig_fixed, svdd,pocket_com_before,dataset_info,sanitize,relax_iter,largest_frag, resamplings=1, return_frames=1,
                timesteps=None, center='ligand'):
        """
        Draw samples from the generative model while fixing parts of the input.
        Optionally, return intermediate states for visualization purposes.
        Inspired by Algorithm 1 in:
        Lugmayr, Andreas, et al.
        "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition. 2022.
        """
        print("[INFO]cond.py: we are now using the inpainting func")
        optimize = 0
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        if len(lig_fixed.size()) == 1:
            lig_fixed = lig_fixed.unsqueeze(1)

        n_samples = len(ligand['size'])
        device = pocket['x'].device

        # Normalize
        ligand, pocket = self.normalize(ligand, pocket)

        # xh0_pocket is the original pocket while xh_pocket might be a
        # translated version of it
        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)
        com_pocket_0 = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        xh0_ligand = torch.cat([ligand['x'], ligand['one_hot']], dim=1)
        xh_ligand = xh0_ligand.clone()

        # Center initial system, subtract COM of known parts
        if center == 'ligand':
            mean_known = scatter_mean(ligand['x'][lig_fixed.bool().view(-1)],
                                      ligand['mask'][lig_fixed.bool().view(-1)],
                                      dim=0)
        elif center == 'pocket':
            mean_known = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        else:
            raise NotImplementedError(
                f"Centering option {center} not implemented")

        # Sample from Normal distribution in the ligand center
        mu_lig_x = mean_known
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[ligand['mask']]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, ligand['mask'], pocket['mask'])

        # Output tensors
        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Iteratively sample with resampling iterations
        # resamplings = resamplings+1
        lig_mask = ligand['mask']
        for s in reversed(range(0, timesteps)):
            print(f"[INFO]cond.py: we are now in the s={s} with resamplings={resamplings}")
            # resampling iterations
            for u in range(resamplings):
                # Denoise one time step: t -> s
                s_array = torch.full((n_samples, 1), fill_value=s,
                                     device=device)
                t_array = s_array + 1
                s_array = s_array / timesteps
                t_array = t_array / timesteps

                gamma_t = self.gamma(t_array)
                gamma_s = self.gamma(s_array)

                # sample inpainted part
                z_lig_unknown, xh_pocket, log_prob_adjust = self.sample_p_zs_given_zt(
                    s_array, t_array, z_lig, xh_pocket, ligand['mask'],
                    pocket['mask'],optimize)
                ########################################################
                if s <= 16 and s>=12 and u < 1 and True:
                    zeta_0 = 1e-3
                    zeta = zeta_0 * (s / 1200)
                    print(f'zeta is',zeta)
                    guidance_scale = 1e-3
                    # 得到扰动后的z_lig
                    z_lig, xh_pocket = self.my_update_z_lig(z_lig,xh_pocket, lig_mask,pocket['mask'], pocket_com_before, dataset_info,sanitize,relax_iter,largest_frag,t_array,n_samples, zeta, guidance_scale=guidance_scale)
                    x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                    z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                    z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                    print("complete SPSA")
                    xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                    z_lig_unknown = torch.cat([z_p_lig, h_p_lig], dim=1)
                    self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)
                ########################################################
                # sample known nodes from the input
                com_pocket = scatter_mean(xh_pocket[:, :self.n_dims],
                                          pocket['mask'], dim=0)
                xh_ligand[:, :self.n_dims] = \
                    ligand['x'] + (com_pocket - com_pocket_0)[ligand['mask']]
                z_lig_known, xh_pocket, _ = self.noised_representation(
                    xh_ligand, xh_pocket, ligand['mask'], pocket['mask'],
                    gamma_s)

                # move center of mass of the noised part to the center of mass
                # of the corresponding denoised part before combining them
                # -> the resulting system should be COM-free
                com_noised = scatter_mean(
                    z_lig_known[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                com_denoised = scatter_mean(
                    z_lig_unknown[lig_fixed.bool().view(-1)][:, :self.n_dims],
                    ligand['mask'][lig_fixed.bool().view(-1)], dim=0)
                dx = com_denoised - com_noised
                z_lig_known[:, :self.n_dims] = z_lig_known[:, :self.n_dims] + dx[ligand['mask']]
                xh_pocket[:, :self.n_dims] = xh_pocket[:, :self.n_dims] + dx[pocket['mask']]

                # combine
                z_lig = z_lig_known * lig_fixed + z_lig_unknown * (
                            1 - lig_fixed)

                if u < resamplings - 1:
                    # Noise the sample
                    z_lig, xh_pocket = self.sample_p_zt_given_zs(
                        z_lig, xh_pocket, ligand['mask'], pocket['mask'],
                        gamma_t, gamma_s)

                # save frame at the end of a resampling cycle
                if u == resamplings - 1:
                    if (s * return_frames) % timesteps == 0:
                        idx = (s * return_frames) // timesteps

                        out_lig[idx], out_pocket[idx] = \
                            self.unnormalize_z(z_lig, xh_pocket)
                        
            lig_mask = ligand['mask']
            if s <= 10 and s % 2 == 0 and svdd == 1:
                print(f"[DEBUG] Doing multi-sample at step s={s}")
                z_lig_0, h_lig_0, x_pocket_0, h_pocket_0 = self.my_to_x0(t_array, z_lig, xh0_pocket,lig_mask,pocket['mask'],n_samples)
                xh_pocket_0 = torch.cat([x_pocket_0, h_pocket_0], dim=1)
                z_lig_0 = torch.cat([z_lig_0, h_lig_0], dim=1)
                # 1) 把当前 (z_lig, xh_pocket) 视作第1组
                multi_z_lig_list_0 = [z_lig_0]
                multi_xh_pocket_list_0 = [xh_pocket_0]
                multi_mask_list_0 = [lig_mask]  # mask 也要对应复制
                multi_pocket_mask_list_0 = [pocket['mask'].clone()]  # 原本 pocket_mask

                multi_z_lig_list = [z_lig]
                multi_xh_pocket_list = [xh_pocket]
                multi_mask_list = [lig_mask]  # mask 也要对应复制
                multi_pocket_mask_list = [pocket['mask'].clone()]  # 原本 pocket_mask
                # 2) 再多采4组(总计5组)
                num_extra = 4
                for i in range(num_extra):
                    # 可以先复制 z_lig, xh_pocket，然后再采一次
                    z_tmp, xh_tmp, _ = self.sample_p_zs_given_zt(
                        s_array, t_array,
                        z_lig.clone(), xh_pocket.clone(),
                        lig_mask.clone(), pocket['mask'].clone(),
                        optimize
                    )
                    z_temp_0, h_temp_0, x_pocket_temp_0, h_pocket_temp_0 = self.my_to_x0(t_array, z_tmp, xh_tmp,lig_mask,pocket['mask'],n_samples)
                    xh_pocket_0 = torch.cat([x_pocket_temp_0, h_pocket_temp_0], dim=1)
                    z_temp_0 = torch.cat([z_temp_0, h_temp_0], dim=1)
                    multi_z_lig_list_0.append(z_temp_0)
                    multi_xh_pocket_list_0.append(xh_pocket_0)
                    multi_mask_list_0.append(lig_mask.clone())
                    multi_pocket_mask_list_0.append(pocket['mask'].clone())

                    multi_z_lig_list.append(z_tmp)
                    multi_xh_pocket_list.append(xh_tmp)
                    multi_mask_list.append(lig_mask.clone())
                    multi_pocket_mask_list.append(pocket['mask'].clone())
                # 3) 合并这 5 组 (z_lig, xh_pocket)，共5 * 20 = 100分子
                #   如果 z_lig shape = [N_lig, 13], xh_pocket shape=[N_pocket,13],
                #   则 5份合并后 shape = [5*N_lig, 13] / [5*N_pocket, 13]
                big_z_lig_0 = torch.cat(multi_z_lig_list_0, dim=0)
                big_xh_pocket_0 = torch.cat(multi_xh_pocket_list_0, dim=0)
                big_mask_0 = torch.cat(multi_mask_list_0, dim=0)
                big_pmask_0 = torch.cat(multi_pocket_mask_list_0, dim=0)    # shape ~ [5*N_pocket]

                big_z_lig = torch.cat(multi_z_lig_list, dim=0)
                big_xh_pocket = torch.cat(multi_xh_pocket_list, dim=0)
                big_mask = torch.cat(multi_mask_list, dim=0)
                big_pmask = torch.cat(multi_pocket_mask_list, dim=0)    # shape ~ [5*N_pocket]

                print(f"big_z_lig: {big_z_lig.shape}")
                print(f"big_xh_pocket: {big_xh_pocket.shape}")
                print(f"big_mask: {big_mask.shape}")
                print(f"big_pmask: {big_pmask.shape}")
                # (4) 对 pocket_mask 做 offset, 让 scatter_mean 分出 5 组 batch
                chunk_size = xh_pocket.shape[0]  # 原单份 pocket 节点数
                print(f"chunk_size: {chunk_size}")
                for i in range(1, 5):
                    start_idx = i * chunk_size
                    end_idx = (i+1) * chunk_size
                    # 每份 pocket_mask 全加 i => batch区分
                    big_pmask_0[start_idx:end_idx] += i * 20
                for i in range(1, 5):
                    start_idx = i * chunk_size
                    end_idx = (i+1) * chunk_size
                    # 每份 pocket_mask 全加 i => batch区分
                    big_pmask[start_idx:end_idx] += i * 20
                #  此时 big_mask 的取值范围还都是 [0..19], 并没区分 5 组
                #  我们需要把“前面N_lig个节点归属0..19, 中间N_lig个节点归属20..39”... 
                offset_size = z_lig.shape[0]  # N_lig
                for i in range(1, len(multi_mask_list)):
                    start_idx = i * offset_size
                    end_idx = (i+1) * offset_size
                    # 给 big_mask[start_idx:end_idx] + i*20
                    big_mask_0[start_idx:end_idx] += i * 20
                for i in range(1, len(multi_mask_list)):
                    start_idx = i * offset_size
                    end_idx = (i+1) * offset_size
                    # 给 big_mask[start_idx:end_idx] + i*20
                    big_mask[start_idx:end_idx] += i * 20
                print(f"the pocket_com_before: {pocket_com_before.shape}")
                big_pocket_com_before = pocket_com_before.repeat(5, 1)
                pocket_com_before_saved = pocket_com_before  # 先存原
                pocket_com_before = big_pocket_com_before    # 临时替换
                print(f"pocket_com_before: {pocket_com_before.shape}")
                # 3) 对 big_z_lig_0, big_xh_pocket_0 做 handle_to_mol => 5*20个分子
                #    并计算 reward
                with torch.no_grad():
                    molecules_100_0 = self.handle_to_mol(
                        big_z_lig_0, big_xh_pocket_0,
                        big_mask_0,  # 这里已区分 0..99
                        big_pmask_0, pocket_com_before,
                        dataset_info, sanitize, relax_iter, largest_frag
                    )
                    rewards_100_0 = self.my_reward_for_SVDD(molecules_100_0)
                    rewards_100_0 = torch.tensor(rewards_100_0, device=z_lig.device)
                
                # 3) 对 big_z_lig, big_xh_pocket 做 handle_to_mol => 5*20个分子
                #    并计算 reward
                with torch.no_grad():
                    molecules_100 = self.handle_to_mol(
                        big_z_lig, big_xh_pocket,
                        big_mask,  # 这里已区分 0..99
                        big_pmask, pocket_com_before,
                        dataset_info, sanitize, relax_iter, largest_frag
                    )
                    pocket_com_before = pocket_com_before_saved

                    rewards_100 = self.my_reward_for_SVDD(molecules_100)
                    rewards_100 = torch.tensor(rewards_100, device=z_lig.device)
                # 4) 将两种reward混合
                mixed_reward = rewards_100_0*(s/250) + rewards_100*(250-s/250)
                # 4) 按rewards选 top20
                _, top_idx = mixed_reward.topk(k=20, largest=True)
                # 5) 根据 top_idx 重建 z_lig, xh_pocket, lig_mask
                # 只保留 top20 分子的节点
                #   => 先构造 new_z_lig_list / new_mask_list
                new_z_lig_list = []
                new_xh_pocket_list = []
                new_mask_list = []
                for rank, idx_f in enumerate(top_idx):
                    # idx_f.item() in [0..99]
                    # 找到 big_mask==idx_f
                    node_mask = (big_mask == idx_f.item()) # 配子分子的掩码
                    nodep_mask = (big_pmask == idx_f.item()) # 蛋白质口袋的掩码
                    # 取出 z_lig, xh_pocket
                    z_sub = big_z_lig[node_mask]
                    xh_sub = big_xh_pocket[nodep_mask]
                    # 但要把 mask 改到 rank
                    # 让 new_mask=rank
                    sub_mask = torch.full_like(node_mask, fill_value=rank, dtype=big_mask.dtype)
                    sub_mask = sub_mask[node_mask]  # 只剩那些 True
                    # 分别存起来
                    new_z_lig_list.append(z_sub)
                    new_xh_pocket_list.append(xh_sub)
                    new_mask_list.append(sub_mask)
                
                # 拼接 => 得到新的 [N_lig', 13], 其中 N_lig' = 所有 top20 分子节点之和
                z_lig = torch.cat(new_z_lig_list, dim=0)
                xh_pocket = torch.cat(new_xh_pocket_list, dim=0)
                lig_mask = torch.cat(new_mask_list, dim=0)
                print(f"[DEBUG] selected top20 => new z_lig.shape={z_lig.shape}, new lig_mask range={lig_mask.shape}, s={s}")
                # 新 z_lig 保证零质心
                x_pocket, h_pocket = self.unnormalize(xh_pocket[:, :self.n_dims], xh_pocket[:, self.n_dims:])
                z_p_lig, h_p_lig = self.unnormalize(z_lig[:, :self.n_dims], z_lig[:, self.n_dims:])
                z_p_lig, x_pocket = self.remove_mean_batch(z_p_lig, x_pocket, lig_mask, pocket['mask'])
                print("complete SPSA")
                xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)
                z_lig = torch.cat([z_p_lig, h_p_lig], dim=1)
                self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)
            

        # Finally sample p(x, h | z_0).
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, ligand['mask'], pocket['mask'], n_samples)

        # Overwrite last frame with the resulting x and h.
        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # remove frame dimension if only the final molecule is returned
        return out_lig.squeeze(0), out_pocket.squeeze(0), ligand['mask'], \
               pocket['mask']

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):

        # Just subtract the center of mass of the sampled part
        mean = scatter_mean(x_lig, lig_indices, dim=0)

        x_lig = x_lig - mean[lig_indices]
        x_pocket = x_pocket - mean[pocket_indices]
        return x_lig, x_pocket


# ------------------------------------------------------------------------------
# The same model without subspace-trick
# ------------------------------------------------------------------------------
class SimpleConditionalDDPM(ConditionalDDPM):
    """
    Simpler conditional diffusion module without subspace-trick.
    - rotational equivariance is guaranteed by construction
    - translationally equivariant likelihood is achieved by first mapping
      samples to a space where the context is COM-free and evaluating the
      likelihood there
    - molecule generation is equivariant because we can first sample in the
      space where the context is COM-free and translate the whole system back to
      the original position of the context later
    """
    def subspace_dimensionality(self, input_size):
        """ Override because we don't use the linear subspace anymore. """
        return input_size * self.n_dims

    @classmethod
    def remove_mean_batch(cls, x_lig, x_pocket, lig_indices, pocket_indices):
        """ Hacky way of removing the centering steps without changing too much
        code. """
        return x_lig, x_pocket

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
        return

    def forward(self, ligand, pocket, return_info=False):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        ligand['x'] = ligand['x'] - pocket_com[ligand['mask']]
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).forward(
            ligand, pocket, return_info)

    @torch.no_grad()
    def sample_given_pocket(self, pocket, num_nodes_lig, return_frames=1,
                            timesteps=None):

        # Subtract pocket center of mass
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        pocket['x'] = pocket['x'] - pocket_com[pocket['mask']]

        return super(SimpleConditionalDDPM, self).sample_given_pocket(
            pocket, num_nodes_lig, return_frames, timesteps) 