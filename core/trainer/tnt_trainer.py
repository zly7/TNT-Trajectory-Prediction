import copy
import os
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.evaluation.competition_util import generate_forecasting_h5

from apex import amp
from apex.parallel import DistributedDataParallel

from core.trainer.trainer import Trainer
from core.model.TNT import TNT
from core.optim_schedule import ScheduledOptim
from core.util.viz_utils import show_pred_and_gt
from core.loss import TNTLoss
# from core.util.preprocessor.argoverse_preprocess_v2 import visualize_data

class TNTTrainer(Trainer):
    """
    TNT Trainer, train the TNT with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=30,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 multi_gpu=False,
                 enable_log=True,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True,
                 on_memory: bool = False,
                 global_graph_width: int = 64,
                 ):
        """
        trainer class for tnt
        :param trainset: see parent class
        :param evalset: see parent class
        :param testset: see parent class
        :param lr: see parent class
        :param betas: see parent class
        :param weight_decay: see parent class
        :param warmup_steps: see parent class
        :param with_cuda: see parent class
        :param cuda_device: see parent class
        :param multi_gpu: see parent class
        :param log_freq: see parent class
        :param model_path: str, the path to a trained model
        :param ckpt_path: str, the path to a stored checkpoint to be resumed
        :param verbose: see parent class
        """
        super(TNTTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            multi_gpu=multi_gpu,
            enable_log=enable_log,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose,
            on_memory=on_memory,
        )

        # init or load model
        self.horizon = horizon
        self.aux_loss = aux_loss

        self.lambda1 = 0.1
        self.lambda2 = 1.0
        self.lambda3 = 0.1

        # input dim: (20, 8); output dim: (30, 2)
        # model_name = VectorNet
        model_name = TNT  # 这么写真的很强
        self.model = model_name(
            self.trainset.num_features if hasattr(self.trainset, 'num_features') else self.testset.num_features,
            self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device,
            global_graph_width=global_graph_width
        )
        self.criterion = TNTLoss(
            self.lambda1, self.lambda2, self.lambda3,
            self.model.m, self.model.k, 0.01,
            aux_loss=self.aux_loss,
            device=self.device
        )

        # init optimizer
        self.optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(  # 这个是学习率的变化情况
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')

        self.model = self.model.to(self.device)
        if self.multi_gpu:
            self.model = DistributedDataParallel(self.model)
            self.model, self.optimizer = amp.initialize(self.model, self.optim, opt_level="O0")
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1)):
                print("[TNTTrainer]: Train the mode with multiple GPUs: {} GPUs.".format(int(os.environ['WORLD_SIZE'])))
        else:
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1)):
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))

        # record the init learning rate
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.write_log("LR", self.lr, 0)

        # resume training from ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                0.0,
                avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            n_graph = data.num_graphs
            data = data.to(self.device)

            if training:
                self.optm_schedule.zero_grad()
                loss, loss_dict = self.compute_loss(data)  # compute_loss承担主要的前向传播的任务
                                                        # 它这里loss的计算也相当于搞成一个moudle

                if self.multi_gpu:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step()

                # writing loss
                if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                    self.write_log("Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Cls_Loss",
                                loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Offset_Loss",
                                loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Traj_Loss",
                                loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Score_Loss",
                                loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    loss, loss_dict = self.compute_loss(data)

                    # writing loss
                    if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                        self.write_log("Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                loss.detach().item() / n_graph,
                avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                learning_rate = self.optm_schedule.step_and_update_lr()
                self.write_log("LR", learning_rate, epoch + 1)

        return avg_loss / num_sample

    def compute_loss(self, data):
        n = data.candidate_len_max[0]
        data.y = data.y.view(-1, self.horizon, 2).cumsum(axis=1)
        pred, aux_out, aux_gt = self.model(data)
        #拿到结果
        gt = {
            "target_prob": data.candidate_gt.view(-1, n),
            "offset": data.offset_gt.view(-1, 2),
            "y": data.y.view(-1, self.horizon * 2)
        }
        # 把seq_id和未來的軌跡放進run/info裡面，但是这个数据好像是一维的，不知道是不是预测的数据
        #在训练的时候是用50个点反向传播的
        # seq_id = data["seq_id"]
        # save_info = {}
        # save_info['seq_id'] = seq_id
        # save_info['pred_gt'] = pred['traj_with_gt'].detach().cpu().numpy()
        # with open('../../run/info.txt', mode='w') as f:
        #     f.write(pickle.dump(save_info))
        # return
        # 测试一下，等会删掉


        return self.criterion(pred, gt, aux_out, aux_gt)

    def test(self,
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=False,
             save_pred=False):
        """
        test the testset,
        :param miss_threshold: float, the threshold for the miss rate, default 2.0m
        :param compute_metric: bool, whether compute the metric
        :param convert_coordinate: bool, True: under original coordinate, False: under the relative coordinate
        :param save_pred: store the prediction or not, store in the Argoverse benchmark format
        """
        self.model.eval()

        forecasted_trajectories, gt_trajectories = {}, {}

        # k = self.model.k if not self.multi_gpu else self.model.module.k
        k = self.model.k
        # horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon
        horizon = self.model.horizon

        # debug
        out_dict = {}
        out_cnt = 0

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

                origs = data.orig.numpy()
                rots = data.rot.numpy()
                seq_ids = data.seq_id.numpy()

                if gt is None:
                    compute_metric = False

                # inference and transform dimension
                if self.multi_gpu:
                    out = self.model.module(data.to(self.device))
                    # out = self.model.inference(data.to(self.device))
                else:
                    data_t = copy.deepcopy(data)

                    out = self.model.inference(data_t.to(self.device))  # 这里面会改data的值,用deep copy很成功
                    out_all_dic, aux, aux_ = self.model(data.to(self.device))
                    for key in out_all_dic.keys():
                        out_all_dic[key] = out_all_dic[key].cpu().numpy()

                dim_out = len(out.shape)

                # debug
                out_dict[out_cnt] = out.cpu().numpy()
                out_cnt += 1
                out_all_dic_seq_id = {}

                pred_y = out.unsqueeze(dim_out).view((batch_size, k, horizon, 2)).cpu().numpy()

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    seq_id = seq_ids[batch_id]
                    forecasted_trajectories[seq_id] = [self.convert_coord(pred_y_k, origs[batch_id], rots[batch_id])
                                                       if convert_coordinate else pred_y_k
                                                       for pred_y_k in pred_y[batch_id]]
                    gt_trajectories[seq_id] = self.convert_coord(gt[batch_id], origs[batch_id], rots[batch_id]) \
                        if convert_coordinate else gt[batch_id]

                    out_all_dic_seq_id[seq_id] = {}
                    for key in out_all_dic.keys():
                        out_all_dic_seq_id[seq_id][key] = out_all_dic[key][batch_id]  # 把batch size 拆出来


                # 搞一个batch 画一次图
                if plot:
                    for batch_id in range(batch_size):
                        seq_id = seq_ids[batch_id]
                        import pandas as pd
                        raw_path = "/home/zhuhe/Dataset/interm_data_vis/train_intermediate/raw/features_" + str(
                            seq_id) + ".pkl"
                        raw_data = pd.read_pickle(raw_path)
                        from core.util.preprocessor.argoverse_preprocess_v2 import visualize_data_zly
                        temp_keys = raw_data.keys()
                        data_restore = {}
                        for key in temp_keys:
                            data_restore[key] = raw_data[key].values[0]
                        visualize_data_zly(data_restore, seq_id, forecasted_trajectories[seq_id], out_all_dic_seq_id[seq_id])

        # compute the metric
        if compute_metric:
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                k,
                horizon,
                miss_threshold
            )
            print("[TNTTrainer]: The test result: {};".format(metric_results))
        # todo:画图
        # plot the result
        # if plot:
        #     # fig, ax = plt.subplots()
        #     # for key in forecasted_trajectories.keys():
        #     #     ax.set_xlim(-15, 15)
        #     #     show_pred_and_gt(ax, gt_trajectories[key], forecasted_trajectories[key])
        #     #     plt.pause(3)
        #     #     ax.clear()
        #     for batch_id in range(batch_size):
        #         seq_id = seq_ids[batch_id]
        #         import pandas as pd
        #         raw_path = "/home/zhuhe/Dataset/interm_data_small/train_intermediate/raw/features_" + str(seq_id) + ".pkl"
        #         raw_data = pd.read_pickle(raw_path)
        #         from core.util.preprocessor.argoverse_preprocess_v2 import visualize_data_zly
        #         temp_keys = raw_data.keys()
        #         data_restore = {}
        #         for key in temp_keys:
        #
        #             data_restore[key] = raw_data[key].values[0]
        #         visualize_data_zly(data_restore, seq_id, forecasted_trajectories[seq_id], out_all_dic)


        # todo: save the output in argoverse format
        if save_pred:
            for key in forecasted_trajectories.keys():
                forecasted_trajectories[key] = np.asarray(forecasted_trajectories[key])
            generate_forecasting_h5(forecasted_trajectories, self.save_folder)

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted

