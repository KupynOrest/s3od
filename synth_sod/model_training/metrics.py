import torch
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt as bwdist


_EPS = np.spacing(1)
_TYPE = np.float64


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    return min(2 * matrix.mean(), max_value)


class EMeasure(object):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []
        self.metrics = {
            'adaptive_ems': [],
            'changeable_ems': []
        }

    def step(self, pred: np.ndarray, gt: np.ndarray):
        gt = gt > 0
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)
        self.metrics['changeable_ems'].append(changeable_ems)
        self.metrics['adaptive_ems'].append(adaptive_em)

    def reset(self):
        """Reset accumulated metrics to prevent memory leaks."""
        self.adaptive_ems.clear()
        self.changeable_ems.clear()
        self.metrics['adaptive_ems'].clear()
        self.metrics['changeable_ems'].clear()

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel, fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel, pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs, fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs, pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = 2 * (combination[0] * combination[1]) / \
                                     (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value)
        ]
        return parts_numel, combinations

    def get_metrics(self) -> dict:
        return {
            'Em': np.mean(np.array(self.metrics['changeable_ems'], dtype=_TYPE), axis=0).mean()
        }


class WeightedFMeasure(object):
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.metrics = {
            'weighted_fms': []
        }

    def step(self, pred: np.ndarray, gt: np.ndarray):
        gt = gt > 0
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.metrics['weighted_fms'].append(wfm)

    def reset(self):
        """Reset accumulated metrics to prevent memory leaks."""
        self.metrics['weighted_fms'].clear()

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])


        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m: m + 1, -n: n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_metrics(self) -> dict:
        return {
            'wF': np.mean(np.array(self.metrics['weighted_fms']))
        }


class EvaluationMetrics:
    def __init__(self, device, sm_only=False):
        self.device = device
        self.sm_only = sm_only
        self.metrics = {
            'mae': [],
            'max_f': [],
            'avg_f': [],
            's_score': []
        }
        if not sm_only:
            self.emeasure = EMeasure()
            self.weighted_fmeasure = WeightedFMeasure()

    def step(self, pred, mask):
        if self.sm_only:
            # Only compute S measure
            alpha = 0.5
            y = mask.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                Q = alpha * self._S_object(pred, mask) + (1 - alpha) * self._S_region(pred, mask)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_score = Q.item()
            self.metrics['s_score'].append(s_score)
        else:
            # MAE
            mae = torch.mean(torch.abs(pred - mask)).item()
            # MaxF measure
            beta2 = 0.3
            prec, recall = self._eval_pr(pred, mask, 255)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            max_f = f_score.max().item()
            # AvgF measure
            avg_f = f_score.mean().item()
            # S measure
            alpha = 0.5
            y = mask.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                Q = alpha * self._S_object(pred, mask) + (1 - alpha) * self._S_region(pred, mask)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_score = Q.item()

            self.metrics['mae'].append(mae)
            self.metrics['max_f'].append(max_f)
            self.metrics['avg_f'].append(avg_f)
            self.metrics['s_score'].append(s_score)

            # Calculate EMeasure and WeightedFMeasure
            pred_np = pred.float().cpu().numpy()
            mask_np = mask.float().cpu().numpy()
            self.emeasure.step(pred_np, mask_np)
            self.weighted_fmeasure.step(pred_np, mask_np)

    def compute_metrics(self) -> dict:
        if self.sm_only:
            return {
                'Sm': np.mean(self.metrics['s_score'])
            }
        else:
            base_metrics = {
                'MAE': np.mean(self.metrics['mae']),
                'MaxF': np.mean(self.metrics['max_f']),
                'AvgF': np.mean(self.metrics['avg_f']),
                'Sm': np.mean(self.metrics['s_score'])
            }

            return {
                **base_metrics,
                **self.emeasure.get_metrics(),
                **self.weighted_fmeasure.get_metrics()
            }

    def reset(self):
        """Reset all accumulated metrics to prevent memory leaks."""
        if self.sm_only:
            self.metrics['s_score'].clear()
        else:
            self.metrics['mae'].clear()
            self.metrics['max_f'].clear()
            self.metrics['avg_f'].clear()
            self.metrics['s_score'].clear()
            self.emeasure.reset()
            self.weighted_fmeasure.reset()

    def _eval_pr(self, y_pred, y, num):
        if self.device:
            prec, recall = torch.zeros(num).to(self.device), torch.zeros(num).to(self.device)
            thlist = torch.linspace(0, 1 - 1e-10, num).to(self.device)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, mask):
        fg = torch.where(mask == 0, torch.zeros_like(pred), pred)
        bg = torch.where(mask == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, mask)
        o_bg = self._object(bg, 1 - mask)
        u = mask.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, mask):
        temp = pred[mask == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, mask):
        X, Y = self._centroid(mask)
        mask1, mask2, mask3, mask4, w1, w2, w3, w4 = self._divideGT(mask, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, mask1)
        Q2 = self._ssim(p2, mask2)
        Q3 = self._ssim(p3, mask3)
        Q4 = self._ssim(p4, mask4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, mask):
        rows, cols = mask.size()[-2:]
        mask = mask.view(rows, cols)
        if mask.sum() == 0:
            if self.device:
                X = torch.eye(1).to(self.device) * round(cols / 2)
                Y = torch.eye(1).to(self.device) * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = mask.sum()
            if self.device:
                i = torch.from_numpy(np.arange(0, cols)).to(self.device).float()
                j = torch.from_numpy(np.arange(0, rows)).to(self.device).float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((mask.sum(dim=0) * i).sum() / total)
            Y = torch.round((mask.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, mask, X, Y):
        h, w = mask.size()[-2:]
        area = h * w
        mask = mask.view(h, w)
        LT = mask[:Y, :X]
        RT = mask[:Y, X:w]
        LB = mask[Y:h, :X]
        RB = mask[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, mask):
        mask = mask.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = mask.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((mask - y) * (mask - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (mask - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q
