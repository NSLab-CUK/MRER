import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import f1_score
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class MRER():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.lambda_balance = getattr(self.args, "lambda_balance", 0.005)
        self.use_er_dca = getattr(self.args, "use_er_dca", True)
        self.lambda_rw_recon = getattr(self.args, "lambda_rw_recon", 0.01)
        self.lambda_consistency = getattr(self.args, "lambda_consistency", 0.005)
        self.use_binary_aux = getattr(self.args, "use_binary_aux", False)
        self.lambda_binary = getattr(self.args, "lambda_binary", 0.005)
        self.use_boundary_aux = getattr(self.args, "use_boundary_aux", False)
        self.lambda_boundary = getattr(self.args, "lambda_boundary", 0.001)
        self.boundary_margin = getattr(self.args, "boundary_margin", 0.1)
        self.use_output_calibration = getattr(self.args, "use_output_calibration", False)
        self.calibration_scale = getattr(self.args, "calibration_scale", 1.0)
        self.calibration_bias = getattr(self.args, "calibration_bias", 0.0)
        self.use_acc7_aux = getattr(self.args, "use_acc7_aux", False)
        self.lambda_acc7 = getattr(self.args, "lambda_acc7", 0.025)
        self.use_ordinal_aux = getattr(self.args, "use_ordinal_aux", False)
        self.lambda_ordinal = getattr(self.args, "lambda_ordinal", 0.005)

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            model.train()

            train_loss = 0.0
            train_balance_loss = 0.0
            train_mean_weight = torch.zeros(3, device=self.args.device)
            train_batches = 0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)

                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_ma = self.criterion(output['last_h_l'], labels)  
                    loss_task_v_ma = self.criterion(output['last_h_v'], labels)
                    loss_task_a_ma = self.criterion(output['last_h_a'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = loss_task_all + loss_task_l_ma + loss_task_v_ma + loss_task_a_ma + loss_task_c
                    
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    cosine_similarity_s_c_l = self.cosine(output['s_l'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_l'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_v'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].transpose(0,1).contiguous().view(labels.size(0),-1), output['c_a'].transpose(0,1).contiguous().view(labels.size(0),-1),
                                                          torch.tensor([-1]).cuda()).mean(0)
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    reliability_weights = output['reliability_weights']
                    mean_weight = reliability_weights.mean(dim=0)
                    target = torch.ones_like(mean_weight) / 3
                    loss_balance = F.mse_loss(mean_weight, target)

                    if self.use_er_dca and self.lambda_consistency > 0:
                        loss_consistency = F.mse_loss(output['logits_c'], output['output_logit'].detach())
                    else:
                        loss_consistency = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)

                    combined_loss = loss_task + (loss_s_sr + loss_recon + (loss_sim+loss_ort) * 0.1) * 0.1
                    combined_loss = combined_loss + self.lambda_balance * loss_balance
                    if self.use_er_dca and self.lambda_consistency > 0:
                        combined_loss = combined_loss + self.lambda_consistency * loss_consistency

                    if self.use_binary_aux:
                        labels_flat = labels.squeeze(-1)
                        mask = labels_flat != 0
                        if mask.sum().item() > 0:
                            logit_bin = output['binary_logit'].squeeze(-1)[mask]
                            y_bin = (labels_flat[mask] > 0).float()
                            loss_binary = F.binary_cross_entropy_with_logits(logit_bin, y_bin)
                        else:
                            loss_binary = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)
                    else:
                        loss_binary = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)

                    if self.use_acc7_aux:
                        labels_7 = torch.clamp(torch.round(labels.squeeze(-1)), min=-3, max=3).long() + 3
                        loss_acc7 = F.cross_entropy(output['logits_7'], labels_7)
                    else:
                        loss_acc7 = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)

                    if self.use_ordinal_aux:
                        pred_flat = output['output_logit'].squeeze(-1)
                        labels_flat = labels.squeeze(-1)

                        thresholds = torch.tensor(
                            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
                            device=labels.device,
                            dtype=labels.dtype
                        )

                        ordinal_targets = (labels_flat.unsqueeze(1) > thresholds.unsqueeze(0)).float()
                        ordinal_logits = pred_flat.unsqueeze(1) - thresholds.unsqueeze(0)

                        loss_ordinal = F.binary_cross_entropy_with_logits(
                            ordinal_logits,
                            ordinal_targets
                        )
                    else:
                        loss_ordinal = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)

                    if self.use_boundary_aux:
                        labels_flat = labels.squeeze(-1)
                        pred_flat = output['output_logit'].squeeze(-1)

                        mask_boundary = labels_flat.abs() >= 0.5

                        if mask_boundary.sum().item() > 0:
                            y_sign = torch.where(
                                labels_flat[mask_boundary] > 0,
                                torch.ones_like(labels_flat[mask_boundary]),
                                -torch.ones_like(labels_flat[mask_boundary])
                            )

                            margin_value = self.boundary_margin
                            loss_boundary = F.relu(margin_value - y_sign * pred_flat[mask_boundary]).mean()
                        else:
                            loss_boundary = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)
                    else:
                        loss_boundary = torch.zeros(1, device=labels.device, dtype=labels.dtype).squeeze(0)

                    if self.use_binary_aux:
                        combined_loss = combined_loss + self.lambda_binary * loss_binary
                    if self.use_acc7_aux:
                        combined_loss = combined_loss + self.lambda_acc7 * loss_acc7
                    if self.use_ordinal_aux:
                        combined_loss = combined_loss + self.lambda_ordinal * loss_ordinal
                    if self.use_boundary_aux:
                        combined_loss = combined_loss + self.lambda_boundary * loss_boundary

                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)

                    train_loss += combined_loss.item()
                    train_balance_loss += loss_balance.item()
                    train_mean_weight += mean_weight.detach()
                    train_batches += 1

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            avg_balance_loss = train_balance_loss / len(dataloader['train'])
            avg_mean_weight = (train_mean_weight / max(train_batches, 1)).detach().cpu()
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"balance_loss: {round(avg_balance_loss, 6)} "
                f"mean_w: [{avg_mean_weight[0].item():.4f}, {avg_mean_weight[1].item():.4f}, {avg_mean_weight[2].item():.4f}] "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/mrer' + str(self.args.dataset_name) + '.pth'
                torch.save(model.state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        y_bin_pred = []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if 'binary_logit' in output:
                        y_bin_pred.append(output['binary_logit'].cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        if self.use_output_calibration:
            pred_for_metrics = pred * self.calibration_scale + self.calibration_bias
        else:
            pred_for_metrics = pred

        eval_results = self.metrics(pred_for_metrics, true)
        eval_results["Loss"] = round(eval_loss, 4)
        log_results = dict(eval_results)

        if mode in ["VAL", "TEST"] and getattr(self.args, "save_test_predictions", False):
            import os
            import pandas as pd

            save_dir = "result/predictions"
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{self.args.dataset_name}_{mode.lower()}_predictions.csv"
            )

            df = pd.DataFrame({
                "idx": list(range(len(true))),
                "true": true.squeeze(-1).cpu().numpy(),
                "pred_output": pred.squeeze(-1).cpu().numpy(),
                "pred_output_calibrated": pred_for_metrics.squeeze(-1).cpu().numpy(),
            })

            if len(y_bin_pred) > 0:
                bin_pred = torch.cat(y_bin_pred).squeeze(-1)
                df["pred_binary_logit"] = bin_pred.cpu().numpy()
                df["pred_binary_prob"] = torch.sigmoid(bin_pred).cpu().numpy()

            df.to_csv(save_path, index=False)
            logger.info(f"Saved {mode} predictions to {save_path}")

        if len(y_bin_pred) > 0:
            bin_pred = torch.cat(y_bin_pred).squeeze(-1)
            true_flat = true.squeeze(-1)
            mask = true_flat != 0
            if mask.sum().item() > 0:
                bin_pred_label = (bin_pred[mask] > 0).long()
                bin_true_label = (true_flat[mask] > 0).long()
                bin_acc_2 = (bin_pred_label == bin_true_label).float().mean().item()
                bin_f1 = f1_score(
                    bin_true_label.cpu().numpy(),
                    bin_pred_label.cpu().numpy(),
                    average='weighted'
                )
                log_results["Bin_acc_2"] = round(bin_acc_2, 4)
                log_results["Bin_F1"] = round(bin_f1, 4)

        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(log_results)}")

        return eval_results
