

import math
import numpy as np


NEG_INF = -1e300


def _safe_log_weight(w):
    """
    当混合权重为 0 时返回近似负无穷，避免 log(0) 数值错误。
    """
    if w <= 0.0:
        return NEG_INF
    return math.log(w)


class StepwiseMixGaussianPLDAccountant:
    """
    Step-wise MixGaussian PLD accountant.

    适用机制：
        每个 DP update 只抽一次 sigma_t。
        以 1-p_large 的概率使用 sigma_small。
        以 p_large 的概率使用 sigma_large。
        当前 step 的所有参数共享同一个 sigma_t。

    单步机制的一维投影分布：

        Q(z) =
            (1-p) N(0, sigma_small^2 C^2)
          + p     N(0, sigma_large^2 C^2)

        P(z) =
            (1-q) Q(z)
          + q [(1-p) N(C, sigma_small^2 C^2)
               + p     N(C, sigma_large^2 C^2)]

    说明：
        - 该 accountant 与 stepwise_gmm 加噪机制匹配。
        - 该 accountant 不用于 elementwise_gmm 的严格高维 product-mixture 核算。
    """

    def __init__(
        self,
        p_large,
        sigma_small,
        sigma_large,
        C,
        q,
        check_reverse=True,
        mixpld_mode='coin_aware',
    ):
        """
        初始化 Step-wise MixGaussian PLD 的机制参数和数值核算选项。
        """

        if sigma_small <= 0:
            raise ValueError(f"sigma_small must be positive, got {sigma_small}")

        if sigma_large <= 0:
            raise ValueError(f"sigma_large must be positive, got {sigma_large}")

        if sigma_large < sigma_small:
            raise ValueError(
                f"sigma_large should be >= sigma_small. "
                f"Got sigma_large={sigma_large}, sigma_small={sigma_small}."
            )

        if not (0.0 <= p_large <= 1.0):
            raise ValueError(f"p_large must be in [0, 1], got {p_large}")

        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")

        if not (0.0 < q <= 1.0):
            raise ValueError(f"q must be in (0, 1], got {q}")

        if mixpld_mode not in ['hidden', 'coin_aware']:
            raise ValueError(
                f"Unknown mixpld_mode: {mixpld_mode}. "
                f"Expected one of ['hidden', 'coin_aware']."
            )

        self.p_large = float(p_large)
        self.sigma_small = float(sigma_small)
        self.sigma_large = float(sigma_large)
        self.C = float(C)
        self.q = float(q)
        self.check_reverse = bool(check_reverse)
        self.mixpld_mode = mixpld_mode

        self.std_small = self.sigma_small * self.C
        self.std_large = self.sigma_large * self.C

        self.var_small = self.std_small ** 2
        self.var_large = self.std_large ** 2

        self.log_1_minus_p = _safe_log_weight(1.0 - self.p_large)
        self.log_p = _safe_log_weight(self.p_large)

        self.log_1_minus_q = _safe_log_weight(1.0 - self.q)
        self.log_q = _safe_log_weight(self.q)

    def _log_normal_pdf(self, z, mean, var):
        """
        计算一维正态分布 log density，并支持 numpy array 输入。
        """
        return (
            -0.5 * np.log(2.0 * np.pi * var)
            - ((z - mean) ** 2) / (2.0 * var)
        )

    def _log_mix_pdf(self, z, mean):
        """
        使用 logaddexp 稳定计算两分量 MixGaussian 的 log density。
        """
        log_small = (
            self.log_1_minus_p
            + self._log_normal_pdf(z, mean=mean, var=self.var_small)
        )
        log_large = (
            self.log_p
            + self._log_normal_pdf(z, mean=mean, var=self.var_large)
        )

        return np.logaddexp(log_small, log_large)

    def _log_Q_z(self, z):
        """
        计算邻接差异不存在时的基准分布 log Q(z)。
        """
        return self._log_mix_pdf(z, mean=0.0)

    def _log_shifted_mix_z(self, z):
        """
        计算被采样个体贡献 C 后的 shifted mixture log density。
        """
        return self._log_mix_pdf(z, mean=self.C)

    def _log_P_z(self, z):
        """
        计算 Poisson subsampling 后邻接分布 log P(z)。
        """
        log_Q = self._log_Q_z(z)
        log_shift = self._log_shifted_mix_z(z)

        part_not_sampled = self.log_1_minus_q + log_Q
        part_sampled = self.log_q + log_shift

        return np.logaddexp(part_not_sampled, part_sampled)

    def _log_gaussian_Q_z(self, z, sigma):
        """
        计算单一 Gaussian 分支的 log Q_sigma(z)。
        """
        var = (float(sigma) * self.C) ** 2
        return self._log_normal_pdf(z, mean=0.0, var=var)

    def _log_gaussian_shifted_z(self, z, sigma):
        """
        计算单一 Gaussian 分支的 shifted log density。
        """
        var = (float(sigma) * self.C) ** 2
        return self._log_normal_pdf(z, mean=self.C, var=var)

    def _log_gaussian_P_z(self, z, sigma):
        """
        计算 Poisson subsampled Gaussian 分支的 log P_sigma(z)。
        """
        log_Q = self._log_gaussian_Q_z(z, sigma=sigma)
        log_shift = self._log_gaussian_shifted_z(z, sigma=sigma)

        part_not_sampled = self.log_1_minus_q + log_Q
        part_sampled = self.log_q + log_shift

        return np.logaddexp(part_not_sampled, part_sampled)

    def _effective_grid_sigma(self):
        """
        根据 mixture 权重选择积分网格尺度。
        """
        if self.p_large <= 0.0:
            return self.sigma_small
        if self.p_large >= 1.0:
            return self.sigma_large
        return self.sigma_large

    def _make_z_grid(self, num_z_points, tail_multiplier):
        """
        根据当前有效混合分量构造积分网格，避免退化情形下无效宽尾降低分辨率。
        """
        grid_sigma = self._effective_grid_sigma()
        limit = tail_multiplier * grid_sigma * self.C
        z_grid = np.linspace(-limit, limit, num_z_points)
        dz = z_grid[1] - z_grid[0]
        return z_grid, dz

    def _single_step_hidden_pld_histogram(
        self,
        target_epsilon,
        num_z_points=50000,
        num_bins=8192,
        tail_multiplier=15.0,
        direction="forward",
    ):
        """
        构造单步 PLD 离散直方图，并支持 forward/reverse 双向 hockey-stick 检查。
        """

        if direction not in ["forward", "reverse"]:
            raise ValueError(f"Unknown direction: {direction}")

        z_grid, dz = self._make_z_grid(
            num_z_points=num_z_points,
            tail_multiplier=tail_multiplier,
        )

        log_P = self._log_P_z(z_grid)
        log_Q = self._log_Q_z(z_grid)

        if direction == "forward":
            log_base = log_P
            L_vals = log_P - log_Q
        else:
            log_base = log_Q
            L_vals = log_Q - log_P

        base_vals = np.exp(log_base)
        prob_mass = base_vals * dz

        valid = np.isfinite(L_vals) & np.isfinite(prob_mass) & (prob_mass >= 0.0)

        L_vals = L_vals[valid]
        prob_mass = prob_mass[valid]

        mass_before_norm = float(np.sum(prob_mass))
        if not np.isfinite(mass_before_norm) or mass_before_norm <= 0.0:
            raise RuntimeError(
                f"Invalid probability mass for direction={direction}. "
                f"mass={mass_before_norm}"
            )

        prob_mass = prob_mass / mass_before_norm

        L_min = float(np.min(L_vals))
        L_max = float(np.max(L_vals))

        if not np.isfinite(L_min) or not np.isfinite(L_max) or L_max <= L_min:
            raise RuntimeError(
                f"Invalid privacy loss range: L_min={L_min}, L_max={L_max}"
            )

        hist, edges = np.histogram(
            L_vals,
            bins=num_bins,
            range=(L_min, L_max),
            weights=prob_mass,
        )

        hist_sum = float(np.sum(hist))
        if not np.isfinite(hist_sum) or hist_sum <= 0.0:
            raise RuntimeError(
                f"Invalid PLD histogram for direction={direction}. "
                f"hist_sum={hist_sum}"
            )

        hist = hist / hist_sum

        dL = float(edges[1] - edges[0])
        centers = 0.5 * (edges[:-1] + edges[1:])

        diagnostics = {
            "direction": direction,
            "mass_before_norm": mass_before_norm,
            "hist_sum_after_norm": float(np.sum(hist)),
            "L_min": L_min,
            "L_max": L_max,
            "dL": dL,
            "num_z_points": num_z_points,
            "num_bins": num_bins,
            "target_epsilon": target_epsilon,
        }

        return hist, centers, dL, diagnostics

    def _single_step_coin_aware_pld_histogram(
        self,
        target_epsilon,
        num_z_points=50000,
        num_bins=8192,
        tail_multiplier=15.0,
        direction="forward",
    ):
        """
        Coin-aware 模式下先计算分支 PLD，再按硬币概率混合 privacy loss 分布。
        """

        if direction not in ["forward", "reverse"]:
            raise ValueError(f"Unknown direction: {direction}")

        z_grid, dz = self._make_z_grid(
            num_z_points=num_z_points,
            tail_multiplier=tail_multiplier,
        )

        branch_items = []
        branch_specs = [
            (1.0 - self.p_large, self.sigma_small, "small"),
            (self.p_large, self.sigma_large, "large"),
        ]
        all_L_vals = []

        for weight, sigma, name in branch_specs:
            if weight <= 0.0:
                continue

            log_P = self._log_gaussian_P_z(z_grid, sigma=sigma)
            log_Q = self._log_gaussian_Q_z(z_grid, sigma=sigma)

            if direction == "forward":
                log_base = log_P
                L_vals = log_P - log_Q
            else:
                log_base = log_Q
                L_vals = log_Q - log_P

            base_vals = np.exp(log_base)
            prob_mass = base_vals * dz

            valid = (
                np.isfinite(L_vals)
                & np.isfinite(prob_mass)
                & (prob_mass >= 0.0)
            )

            L_vals = L_vals[valid]
            prob_mass = prob_mass[valid]

            mass_before_norm = float(np.sum(prob_mass))
            if not np.isfinite(mass_before_norm) or mass_before_norm <= 0.0:
                raise RuntimeError(
                    f"Invalid branch probability mass. "
                    f"direction={direction}, branch={name}, mass={mass_before_norm}"
                )

            prob_mass = prob_mass / mass_before_norm

            branch_items.append(
                {
                    "name": name,
                    "weight": float(weight),
                    "sigma": float(sigma),
                    "L_vals": L_vals,
                    "prob_mass": prob_mass,
                    "mass_before_norm": mass_before_norm,
                }
            )
            all_L_vals.append(L_vals)

        if len(branch_items) == 0:
            raise RuntimeError("No valid branch in coin-aware PLD histogram.")

        all_L_concat = np.concatenate(all_L_vals)
        L_min = float(np.min(all_L_concat))
        L_max = float(np.max(all_L_concat))

        if not np.isfinite(L_min) or not np.isfinite(L_max) or L_max <= L_min:
            raise RuntimeError(
                f"Invalid coin-aware privacy loss range: "
                f"L_min={L_min}, L_max={L_max}"
            )

        hist_total = np.zeros(num_bins, dtype=np.float64)
        branch_diagnostics = []
        edges = None

        for item in branch_items:
            hist_branch, edges = np.histogram(
                item["L_vals"],
                bins=num_bins,
                range=(L_min, L_max),
                weights=item["prob_mass"],
            )

            hist_branch_sum = float(np.sum(hist_branch))
            if not np.isfinite(hist_branch_sum) or hist_branch_sum <= 0.0:
                raise RuntimeError(
                    f"Invalid branch histogram. "
                    f"direction={direction}, branch={item['name']}, "
                    f"hist_sum={hist_branch_sum}"
                )

            hist_branch = hist_branch / hist_branch_sum
            hist_total += item["weight"] * hist_branch

            branch_diagnostics.append(
                {
                    "name": item["name"],
                    "weight": item["weight"],
                    "sigma": item["sigma"],
                    "mass_before_norm": item["mass_before_norm"],
                    "hist_sum_after_norm": float(np.sum(hist_branch)),
                }
            )

        hist_sum = float(np.sum(hist_total))
        if not np.isfinite(hist_sum) or hist_sum <= 0.0:
            raise RuntimeError(
                f"Invalid coin-aware total histogram. hist_sum={hist_sum}"
            )

        hist_total = hist_total / hist_sum

        dL = float(edges[1] - edges[0])
        centers = 0.5 * (edges[:-1] + edges[1:])

        diagnostics = {
            "mode": "coin_aware",
            "direction": direction,
            "hist_sum_after_norm": float(np.sum(hist_total)),
            "L_min": L_min,
            "L_max": L_max,
            "dL": dL,
            "num_z_points": num_z_points,
            "num_bins": num_bins,
            "target_epsilon": target_epsilon,
            "branches": branch_diagnostics,
        }

        return hist_total, centers, dL, diagnostics

    def _single_step_pld_histogram(
        self,
        target_epsilon,
        num_z_points=50000,
        num_bins=8192,
        tail_multiplier=15.0,
        direction="forward",
    ):
        """
        根据 mixpld_mode 选择 hidden 或 coin-aware PLD。
        """
        if self.mixpld_mode == "hidden":
            return self._single_step_hidden_pld_histogram(
                target_epsilon=target_epsilon,
                num_z_points=num_z_points,
                num_bins=num_bins,
                tail_multiplier=tail_multiplier,
                direction=direction,
            )

        if self.mixpld_mode == "coin_aware":
            return self._single_step_coin_aware_pld_histogram(
                target_epsilon=target_epsilon,
                num_z_points=num_z_points,
                num_bins=num_bins,
                tail_multiplier=tail_multiplier,
                direction=direction,
            )

        raise ValueError(f"Unknown mixpld_mode: {self.mixpld_mode}")

    def _compose_pld_histogram(self, hist, centers, dL, T):
        """
        使用 FFT 对单步 PLD 做 T 次线性自卷积。
        """

        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        num_bins = len(hist)

        conv_len = T * (num_bins - 1) + 1
        pad_length = int(2 ** np.ceil(np.log2(conv_len)))

        padded_hist = np.pad(hist, (0, pad_length - num_bins))

        fft_hist = np.fft.rfft(padded_hist)
        fft_conv = fft_hist ** T
        final_hist = np.fft.irfft(fft_conv, n=pad_length)

        final_hist = np.maximum(final_hist, 0.0)

        final_sum = float(np.sum(final_hist))
        if not np.isfinite(final_sum) or final_sum <= 0.0:
            raise RuntimeError(f"Invalid composed histogram sum: {final_sum}")

        final_hist = final_hist / final_sum

        total_centers = T * centers[0] + np.arange(pad_length) * dL

        diagnostics = {
            "T": T,
            "single_num_bins": num_bins,
            "conv_len": conv_len,
            "pad_length": pad_length,
            "final_hist_sum_after_norm": float(np.sum(final_hist)),
            "total_L_min": float(total_centers[0]),
            "total_L_max": float(total_centers[-1]),
        }

        return final_hist, total_centers, diagnostics

    def _delta_from_pld_histogram(self, hist, centers, target_epsilon):
        """
        根据离散 PLD 计算给定 epsilon 下的 hockey-stick delta。
        """
        tail = centers > target_epsilon

        if not np.any(tail):
            return 0.0

        values = hist[tail] * (
            1.0 - np.exp(target_epsilon - centers[tail])
        )

        delta = float(np.sum(values))
        return max(delta, 0.0)

    def compute_delta_for_T_steps(
        self,
        T,
        target_epsilon,
        num_z_points=50000,
        num_bins=8192,
        tail_multiplier=15.0,
        return_diagnostics=False,
    ):
        """
        计算 T 步 Step-wise MixGaussian 机制在给定 epsilon 下的 delta。
        """

        results = {}

        hist_f, centers_f, dL_f, diag_single_f = self._single_step_pld_histogram(
            target_epsilon=target_epsilon,
            num_z_points=num_z_points,
            num_bins=num_bins,
            tail_multiplier=tail_multiplier,
            direction="forward",
        )

        final_hist_f, total_centers_f, diag_comp_f = self._compose_pld_histogram(
            hist=hist_f,
            centers=centers_f,
            dL=dL_f,
            T=T,
        )

        delta_forward = self._delta_from_pld_histogram(
            hist=final_hist_f,
            centers=total_centers_f,
            target_epsilon=target_epsilon,
        )

        results["forward"] = {
            "delta": delta_forward,
            "single": diag_single_f,
            "composition": diag_comp_f,
        }

        if self.check_reverse:
            hist_r, centers_r, dL_r, diag_single_r = self._single_step_pld_histogram(
                target_epsilon=target_epsilon,
                num_z_points=num_z_points,
                num_bins=num_bins,
                tail_multiplier=tail_multiplier,
                direction="reverse",
            )

            final_hist_r, total_centers_r, diag_comp_r = self._compose_pld_histogram(
                hist=hist_r,
                centers=centers_r,
                dL=dL_r,
                T=T,
            )

            delta_reverse = self._delta_from_pld_histogram(
                hist=final_hist_r,
                centers=total_centers_r,
                target_epsilon=target_epsilon,
            )

            results["reverse"] = {
                "delta": delta_reverse,
                "single": diag_single_r,
                "composition": diag_comp_r,
            }

            delta = max(delta_forward, delta_reverse)
        else:
            delta = delta_forward

        results["selected_delta"] = delta
        results["params"] = {
            "p_large": self.p_large,
            "sigma_small": self.sigma_small,
            "sigma_large": self.sigma_large,
            "C": self.C,
            "q": self.q,
            "T": T,
            "target_epsilon": target_epsilon,
            "check_reverse": self.check_reverse,
            "mixpld_mode": self.mixpld_mode,
        }

        if return_diagnostics:
            return delta, results

        return delta

    def compute_delta_T1_direct(
        self,
        target_epsilon,
        num_z_points=200000,
        tail_multiplier=15.0,
        direction="forward",
    ):
        """
        T=1 时直接做 hockey-stick 积分，用于验证 PLD 离散化。
        """

        z_grid, dz = self._make_z_grid(
            num_z_points=num_z_points,
            tail_multiplier=tail_multiplier,
        )

        P_vals = np.exp(self._log_P_z(z_grid))
        Q_vals = np.exp(self._log_Q_z(z_grid))

        if direction == "forward":
            integrand = np.maximum(P_vals - np.exp(target_epsilon) * Q_vals, 0.0)
        elif direction == "reverse":
            integrand = np.maximum(Q_vals - np.exp(target_epsilon) * P_vals, 0.0)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return float(np.sum(integrand) * dz)


def find_sigma_small_stepwise_mixpld(
    target_eps,
    target_delta,
    T,
    q,
    C,
    p_large=0.05,
    sigma_large=15.0,
    sigma_min=0.10,
    sigma_max=2.00,
    tol=1e-3,
    num_z_points=50000,
    num_bins=8192,
    tail_multiplier=15.0,
    check_reverse=True,
    mixpld_mode='coin_aware',
):
    """
    二分搜索满足 Step-wise MixGaussian PLD 预算的最小 sigma_small。
    """

    print(
        f"[Stepwise-MixGaussian-PLD] search sigma_small | "
        f"mode={mixpld_mode}, "
        f"eps={target_eps}, delta={target_delta:.3e}, "
        f"T={T}, q={q:.8f}, C={C}, "
        f"p_large={p_large}, sigma_large={sigma_large}, "
        f"sigma_range=[{sigma_min}, {sigma_max}]"
    )

    def compute_delta(sigma_small):
        accountant = StepwiseMixGaussianPLDAccountant(
            p_large=p_large,
            sigma_small=sigma_small,
            sigma_large=sigma_large,
            C=C,
            q=q,
            check_reverse=check_reverse,
            mixpld_mode=mixpld_mode,
        )
        return accountant.compute_delta_for_T_steps(
            T=T,
            target_epsilon=target_eps,
            num_z_points=num_z_points,
            num_bins=num_bins,
            tail_multiplier=tail_multiplier,
        )

    delta_at_max = compute_delta(sigma_max)
    if delta_at_max > target_delta:
        raise ValueError(
            f"sigma_max={sigma_max} still violates target delta. "
            f"delta_at_max={delta_at_max:.3e}, "
            f"target_delta={target_delta:.3e}. "
            f"Please increase --sigma_search_max or reduce target_steps."
        )

    delta_at_min = compute_delta(sigma_min)
    if delta_at_min <= target_delta:
        print(
            f"[Stepwise-MixGaussian-PLD] sigma_min={sigma_min:.6f} already feasible, "
            f"actual_delta={delta_at_min:.3e}"
        )
        return sigma_min, delta_at_min

    low = sigma_min
    high = sigma_max
    best_sigma = sigma_max
    best_delta = delta_at_max

    while high - low > tol:
        mid = (low + high) / 2.0
        delta_mid = compute_delta(mid)

        if delta_mid <= target_delta:
            best_sigma = mid
            best_delta = delta_mid
            high = mid
        else:
            low = mid

    print(
        f"[Stepwise-MixGaussian-PLD] found sigma_small={best_sigma:.6f}, "
        f"actual_delta={best_delta:.3e}"
    )

    return best_sigma, best_delta


def compute_delta_stepwise_mixpld(
    target_eps,
    T,
    q,
    C,
    sigma_small,
    sigma_large=15.0,
    p_large=0.05,
    num_z_points=50000,
    num_bins=8192,
    tail_multiplier=15.0,
    check_reverse=True,
    return_diagnostics=False,
    mixpld_mode='coin_aware',
):
    """
    给定 sigma_small，计算 Step-wise MixGaussian PLD 的 delta。
    """
    accountant = StepwiseMixGaussianPLDAccountant(
        p_large=p_large,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        C=C,
        q=q,
        check_reverse=check_reverse,
        mixpld_mode=mixpld_mode,
    )

    return accountant.compute_delta_for_T_steps(
        T=T,
        target_epsilon=target_eps,
        num_z_points=num_z_points,
        num_bins=num_bins,
        tail_multiplier=tail_multiplier,
        return_diagnostics=return_diagnostics,
    )



ProjectedGMMPLDAccountant = StepwiseMixGaussianPLDAccountant


def find_sigma_small_projected_gmm_pld(
    target_eps,
    target_delta,
    T,
    q,
    C,
    p_large=0.05,
    sigma_large=15.0,
    sigma_min=0.10,
    sigma_max=2.00,
    tol=1e-3,
    num_z_points=50000,
    num_bins=8192,
    mixpld_mode='coin_aware',
):
    """
    兼容旧接口，内部调用 Step-wise MixGaussian PLD 搜索器。
    """
    return find_sigma_small_stepwise_mixpld(
        target_eps=target_eps,
        target_delta=target_delta,
        T=T,
        q=q,
        C=C,
        p_large=p_large,
        sigma_large=sigma_large,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        tol=tol,
        num_z_points=num_z_points,
        num_bins=num_bins,
        mixpld_mode=mixpld_mode,
    )


def compute_delta_projected_gmm_pld(
    target_eps,
    T,
    q,
    C,
    sigma_small,
    sigma_large=15.0,
    p_large=0.05,
    num_z_points=50000,
    num_bins=8192,
    mixpld_mode='coin_aware',
):
    """
    兼容旧接口，内部调用 Step-wise MixGaussian PLD delta 计算器。
    """
    return compute_delta_stepwise_mixpld(
        target_eps=target_eps,
        T=T,
        q=q,
        C=C,
        sigma_small=sigma_small,
        sigma_large=sigma_large,
        p_large=p_large,
        num_z_points=num_z_points,
        num_bins=num_bins,
        mixpld_mode=mixpld_mode,
    )
