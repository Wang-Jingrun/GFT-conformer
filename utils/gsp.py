import math, torch
import numpy as np
import torch.nn.functional as F
import os

class GSP:
    def __init__(self, n_f=512, win_l=None, n_s=128, device='cpu', U_path="", training=False):
        self.device = torch.device(device)
        self.n_f = n_f
        self.win_l = n_f if win_l is None else win_l
        self.n_s = n_s

        if os.path.exists(os.path.join(U_path, "gft_U.pth")):
            self.load(os.path.join(U_path, "gft_U.pth"))
        else:
            if training == False:  # 测试
                raise FileNotFoundError("The file 'gft_U.pth' does not exist.")
            else: # 训练
                # 初始化GFT变换矩阵
                base_matrix = self.init_A_matrix()
                self.init_U_matrix(base_matrix)
                self.save(U_path)

    def init_A_matrix(self):
        A = np.zeros((self.n_f, self.n_f), np.float32)
        for i in range(self.n_f):
            for j in range(self.n_f):
                if i != j:
                    A[i, j] += self.n_f - abs(i - j)
        return A

    def init_U_matrix(self, matrix):
        # 特征分解
        evalues, evectors = np.linalg.eig(matrix)

        # 排序特征向量
        sorted_indices = np.argsort(evalues)[: : -1]
        sorted_evectors = evectors[:, sorted_indices]

        self.U = torch.from_numpy(sorted_evectors).type(torch.FloatTensor).to(self.device)
        self.U_T = torch.from_numpy(sorted_evectors.T).type(torch.FloatTensor).to(self.device)
        self.U.requires_grad = False
        self.U_T.requires_grad = False

    def load(self, U_path):
        self.U = torch.load(U_path)
        self.U_T = self.U.T

    def save(self, save_path):
        torch.save(self.U, os.path.join(save_path, "gft_U.pth"))

    def ST_GFT(self, input_f):
        # (batch_size, aduio)
        frames = math.ceil((input_f.shape[1] - self.win_l) / self.n_s + 1)
        # 长度不够补0
        input_f = F.pad(input_f, (0, self.win_l + (frames - 1) * self.n_s - input_f.shape[1]), "constant")
        # unfold 的输入必须是4维度的数据，即 (Batch_size, Channel, High, Wide)
        output_F = F.unfold(input_f.unsqueeze(1).unsqueeze(3), kernel_size=(self.win_l, 1), stride=(self.n_s, 1))

        if output_F.shape[1] < self.n_f:
            output_F = torch.cat([output_F, torch.zeros((output_F.shape[0], self.n_f - output_F.shape[1], output_F.shape[2]), device=self.device)], dim=1)

        return self.U_T @ output_F

    def iST_GFT(self, input_F):
        # (batch_size, N_gft, frames)
        frames = input_F.shape[2]
        output_f = torch.zeros((input_F.shape[0], self.win_l + (frames - 1) * self.n_s), device=self.device)

        input_F = self.U @ input_F

        for frame in range(frames):
            output_f[:, frame * self.n_s: frame * self.n_s + self.win_l] += input_F[:, :, frame][:, :self.win_l]
            # 由于是矩形窗，故重叠部分除以2
            if frame != 0:
                output_f[:, frame * self.n_s:  frame * self.n_s + (self.win_l - self.n_s)] /= 2

        return output_f

