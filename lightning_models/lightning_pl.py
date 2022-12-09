import pandas as pd
import numpy as np
import platform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import copy


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.optimization.optimizer.learning_rate
        if cfg.cpu_thread.limit_cpu_thread:
            torch.set_num_threads(cfg.cpu_thread.num_cpu)
        self.loss = nn.MSELoss()
        self.output_dict = {}

        self.output_dict["prior_to_model_building"] = torch.cuda.memory_allocated()
        self.model = self.build_model(cfg)
        self.output_dict["after_model_building"] = torch.cuda.memory_allocated()

        self.output_dict["pytorch_total_params"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.last_training_epoch = -1
        self.train_batch_time = []

    def training_step(self, batch, batch_idx):
        # collect memory allocation at the 0, 5th epoch and first batch
        tic = time.time()
        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"before_first_batch_{self.current_epoch}"] = torch.cuda.memory_allocated()

        batch_x, batch_y, batch_x_mark, batch_y_mark = batch["seq_x"], batch["seq_y"], batch["seq_x_mark"], batch["seq_y_mark"]

        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"after_first_batch_{self.current_epoch}"] = torch.cuda.memory_allocated()
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)
        # concatenation of encoder - decoder inputs into one tensor
        enc_input = torch.cat((batch_x, batch_x_mark), dim=2)
        dec_input = torch.cat((dec_inp, batch_y_mark), dim=2)
        input_cat = torch.cat((enc_input, dec_input), dim=1)

        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"before_model_forward_{self.current_epoch}"] = torch.cuda.memory_allocated()
        # torch.cuda.synchronize()
        if self.cfg.model.model_name in ["Dlinear"]:
            outputs = self.model(batch_x)
        else:
            outputs = self.model(input_cat)
        # torch.cuda.synchronize()

        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"after_model_forward_{self.current_epoch}"] = torch.cuda.memory_allocated()

        f_dim = -1 if self.cfg.features == "MS" else 0
        outputs = outputs[:, -self.cfg.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.cfg.pred_len :, f_dim:]

        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"before_loss_{self.current_epoch}"] = torch.cuda.memory_allocated()
        loss = self.loss(outputs, batch_y)

        if self.current_epoch in [0, 3] and batch_idx == 0:
            self.output_dict[f"after_loss_{self.current_epoch}"] = torch.cuda.memory_allocated()
        toc = time.time()
        train_time = toc - tic
        self.train_batch_time.append(train_time)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch["seq_x"], batch["seq_y"], batch["seq_x_mark"], batch["seq_y_mark"]

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)
        # concatenation of encoder - decoder inputs into one tensor
        enc_input = torch.cat((batch_x, batch_x_mark), dim=2)
        dec_input = torch.cat((dec_inp, batch_y_mark), dim=2)
        input_cat = torch.cat((enc_input, dec_input), dim=1)

        if self.cfg.model.model_name in ["Dlinear"]:
            outputs = self.model(batch_x)
        else:
            outputs = self.model(input_cat)
        f_dim = -1 if self.cfg.features == "MS" else 0
        outputs = outputs[:, -self.cfg.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.cfg.pred_len :, f_dim:]

        loss = self.loss(outputs, batch_y)
        return {"val_loss": loss, "y_pred": outputs, "label": batch_y}

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu()
        y_true = torch.cat([x["label"] for x in outputs]).cpu()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # reshape to 2D
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        self.log_dict(
            {
                "val_loss": val_loss,
                "mae": mae,
                "mse": mse,
            }
        )
        self.last_training_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch["seq_x"], batch["seq_y"], batch["seq_x_mark"], batch["seq_y_mark"]

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.cfg.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : self.cfg.label_len, :], dec_inp], dim=1).float().to(self.device)
        # concatenation of encoder - decoder inputs into one tensor
        enc_input = torch.cat((batch_x, batch_x_mark), dim=2)
        dec_input = torch.cat((dec_inp, batch_y_mark), dim=2)
        input_cat = torch.cat((enc_input, dec_input), dim=1)

        if self.cfg.model.model_name in ["Dlinear"]:
            outputs = self.model(batch_x)
        else:
            outputs = self.model(input_cat)
        f_dim = -1 if self.cfg.features == "MS" else 0
        outputs = outputs[:, -self.cfg.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.cfg.pred_len :, f_dim:]

        loss = self.loss(outputs, batch_y)
        return {"test_loss": loss, "y_pred": outputs, "label": batch_y}

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu()
        y_true = torch.cat([x["label"] for x in outputs]).cpu()
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        # reshape to 2D
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1, y_true.shape[-1])

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        self.update_system_related_info()
        self.update_experiment_related_info()

        self.output_dict.update(
            {
                "test_loss": test_loss.item(),
                "mae": mae.item(),
                "mse": mse.item(),
            }
        )
        print(f"MAE: {mae.item():.4f}, MSE: {mse.item():.4f}")
        # save output_dict as dataframe
        df = pd.DataFrame(self.output_dict, index=[0])
        df["train_epoch"] = self.last_training_epoch
        df["train_batch"] = self.trainer.num_training_batches
        df["secs/batch"] = np.mean(self.train_batch_time).item()
        df["secs/epoch"] = np.mean(self.train_batch_time).item() * self.trainer.num_training_batches
        df["model_hpams"] = str(self.cfg.model)
        df.to_csv(f"{self.cfg.save_output_path}/run_exp{self.cfg.exp_num}_seed{self.cfg.seed}_{self.cfg.model.model_name}.csv", index=False)
        print(f"Saved output_dict as dataframe to {self.cfg.save_output_path}/run_{self.cfg.exp_num}_{self.cfg.seed}_{self.cfg.model.model_name}.csv")

    def update_system_related_info(self):
        """Update system related info to self.output_dict"""
        # get device info
        device_properties = torch.cuda.get_device_properties(0)
        device_name = device_properties.name
        device_total_memory = device_properties.total_memory / 1e9
        num_threads = torch.get_num_threads()

        # get system info
        python_ver = platform.python_version()
        pytorch_ver = torch.__version__
        pytorch_lightning_ver = pl.__version__

        # get model info
        model_size_mb = pl.utilities.memory.get_model_size_mb(self.model)

        # get system info
        max_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"device_name: {device_name}, device_total_memory: {device_total_memory}, num_threads: {num_threads}, max_memory_mb: {max_memory_mb}")
        # update as key to self.output_dict
        self.output_dict.update(
            {
                "device_name": device_name,
                "device_total_memory": device_total_memory,
                "num_threads": num_threads,
                "model_size_mb": model_size_mb,
                "max_memory_mb": max_memory_mb,
                "python_ver": python_ver,
                "pytorch_ver": pytorch_ver,
                "pytorch_lightning_ver": pytorch_lightning_ver,
            }
        )

    def update_experiment_related_info(self):
        """Update experiment related info to self.output_dict"""
        exp_ess_dict = {}
        exp_ess_dict["data_name"] = self.cfg.data.data_name
        exp_ess_dict["model_name"] = self.cfg.model.model_name
        exp_ess_dict["seq_len"] = self.cfg.seq_len
        exp_ess_dict["label_len"] = self.cfg.label_len
        exp_ess_dict["pred_len"] = self.cfg.pred_len
        exp_ess_dict["features"] = self.cfg.features
        exp_ess_dict["exp_num"] = self.cfg.exp_num
        exp_ess_dict["exp_seed"] = self.cfg.seed
        exp_ess_dict["use_amp"] = self.cfg.use_amp
        exp_ess_dict["cpu_thread"] = self.cfg.cpu_thread.num_cpu
        exp_ess_dict["num_workers"] = self.cfg.optimization.num_workers

        exp_ess_dict["batch_size"] = self.cfg.optimization.batch_size

        # update as key to self.output_dict
        self.output_dict.update(exp_ess_dict)

    def build_model(self, cfg):
        if cfg.model.model_name == "Transformer":
            from models.Transformer import Model

            model = Model(cfg)
        elif cfg.model.model_name == "Autoformer":
            from models.Autoformer import Model

            model = Model(cfg)
        elif cfg.model.model_name == "Informer":
            from models.Informer import Model

            model = Model(cfg)
        elif cfg.model.model_name == "Dlinear":
            from models.Dlinear import Model

            model = Model(cfg)
        else:
            raise NotImplementedError
        return model

    def collect_model_metrics(self):
        self.num_params = sum([p.numel() for p in self.model.parameters()])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = self.select_scheduler(optimizer, self.cfg.optimization.optimizer.lradj)
        return [optimizer], [scheduler]

    def select_scheduler(self, optimizer, lradj):
        if lradj == "type1":
            lambda1 = lambda epoch: self.lr * (0.5 ** ((epoch - 1) // 1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif lradj == "type2":
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            # get the value of the key if present, otherwise return 1
            lambda2 = lambda epoch: lr_adjust.get(epoch) if epoch in lr_adjust else 1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        elif lradj == "3":
            lambda3 = lambda epoch: self.lr if epoch < 10 else self.lr * 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda3)
        elif lradj == "4":
            lambda4 = lambda epoch: self.lr if epoch < 15 else self.lr * 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda4)
        elif lradj == "5":
            lambda5 = lambda epoch: self.lr if epoch < 25 else self.lr * 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda5)
        elif lradj == "6":
            lambda6 = lambda epoch: self.lr if epoch < 5 else self.lr * 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda6)
        return scheduler
