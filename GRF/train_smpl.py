"""
    Copyright (c) 2022, InterDigital R&D France. All rights reserved. This source
    code is made available under the license found in the LICENSE.txt at the root
    directory of the repository.
"""
import sys
sys.path.append('../UnderPressure')


# UnderPressure NN related
import anim, metrics, models, util
from data import TOPOLOGY, Contacts, Dataset

# Python
import math, time
from pathlib import Path

# PyTorch
import torch
from torch.nn.utils.rnn import pad_sequence

MIRROR_LUT = TOPOLOGY.lut(TOPOLOGY.mirrored())
training_loss = []
validation_loss = []
test_loss = []

def split(dataset, ratio, window_length):
    # select random non-overlapping windows for validation set
    nframes = sum(item["poses"].shape[-3] for item in dataset)
    valid_nwindows =  int((1 - ratio) * nframes) // window_length + 1
    windows = []
    for index, item in enumerate(dataset):
        starts = torch.arange(0, item["poses"].shape[-3] - 2 * window_length + 1, window_length)
        # print(starts)
        indices = torch.full_like(starts, index)
        # print(indices)
        windows.append(torch.stack([indices, starts], dim=-1))
    windows = torch.cat(windows)
    windows = windows[torch.randperm(windows.shape[0])[:valid_nwindows]]
    
    # split according to selected windows
    valid_items, train_items = [], []
    for index, item in enumerate(dataset):
        valid_starts = windows[windows[:, 0] == index, 1].sort()[0]
        starts = torch.cat([valid_starts, torch.as_tensor([0]), valid_starts + window_length])
        stops = torch.cat([valid_starts + window_length, valid_starts, torch.as_tensor([item["poses"].shape[-3]])])
        items = dataset.slices(index, starts, stops)
        train_items += [item for item in items[len(valid_starts):] if item["poses"].shape[-3] > 0]
        valid_items += items[:len(valid_starts)]
    return Dataset(train_items), Dataset(valid_items)

def prepare(split_ratio, sequence_length, sequence_overlap):
    # split
    dataset = Dataset.trainset()["poses", "CoP", "GRF"]
    trainset, validset = split(dataset, split_ratio, sequence_length)

    # test
    testset = Dataset.testset()["poses", "CoP", "GRF"]

    # slice into overlapping windows
    if set(a.shape[-3] for a in trainset["poses"]) != {sequence_length}:
        trainset = trainset.windowed(sequence_length, sequence_overlap)
    if set(a.shape[-3] for a in validset["poses"]) != {sequence_length}:
        validset = validset.windowed(sequence_length, 0)
    if set(a.shape[-3] for a in testset["poses"]) != {sequence_length}:
        testset = testset.windowed(sequence_length, 0)

    return trainset.shuffle(), validset, testset

def rnd_transform(positions, forces):                                                # N x F x J x 3, N x F x 2 x 16
    bs, device = positions.shape[0], positions.device
    
    # Mirrorring
    mirror = torch.rand(bs, device=device) < 0.5                                    # N

    positions_mirrored = positions.clone()

    positions_mirrored[mirror] = positions[mirror][..., MIRROR_LUT, :]
    forces_mirrored = forces.clone()
    forces_mirrored[mirror] = forces[mirror][..., [1, 0], :]
    
    return positions_mirrored, forces_mirrored                                        # N x F x J x 3, N x F x 2 x 16

class Trainer(util.Timeline):    
    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        
        # model and optimiser
        self.model = models.DeepNetwork()
        self.model = self.model.initialize().to(self.device)
        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters()))
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=kwargs["learning_rate"])
        self.mse_weight = kwargs["mse_weight"]
        
        # trainset, validset, testset = prepare(kwargs["split_ratio"], kwargs["sequence_length"], kwargs["sequence_overlap"])
        # trainset = trainset["poses", "CoP", "GRF"]

        trainset = torch.load(os.path.join(folder_path, 'trainset.pt'))
        validset = torch.load(os.path.join(folder_path, 'validset.pt'))
        testset  = torch.load(os.path.join(folder_path, 'testset.pt'))
    
        dataloader = trainset.dataloader(
            batch_size=kwargs["batch_size"],
            shuffle=True,
            device=self.device,
        )
        
        # prepare validset: set motions in model representation
        # GroundLink: replace pressure with GRF and CoP
        self.validset = dict(
            poses=    torch.stack(list(validset["poses"])),
            cop=        torch.stack(list(validset["CoP"])),
            grf=        torch.stack(list(validset["GRF"])),
        )
        self.testset  = testset
        
        # logging support
        self.ckp = kwargs["ckp"]
        
        # instanciate timeline
        num_epochs = int(kwargs["iterations"] / len(dataloader) + 0.5)
        super().__init__(dataloader, num_epochs, *[
            util.Schedule(period=100,    fn=self._losses_logging),    # log loss values every X batches
            util.Schedule(period=1000,    fn=self._validation),        # validation every X batches
            # util.Schedule(period=1000,    fn=self._test),        # test every X batches
        ])

        self.L2_metric = torch.nn.MSELoss(reduction='none')
        # self.L1_metric = torch.nn.L1Loss(reduction='mean')
        
    def iteration(self, batch):
        # Modified for GroundLink
        poses, cop, grf, mass = batch["poses"], batch["CoP"], batch["GRF"], batch["mass"]

        # seq_len = poses.shape[1]
        # weight  = (9.81 * mass).repeat(1, seq_len)
        
        contact_cop_grf = torch.cat((cop, grf),3)
        poses, forces_target = rnd_transform(poses.float(), contact_cop_grf)
        
        
        # make predictions and compute loss
        contact_pred = self.model.GRFs(poses)

        self.mse = metrics._mse_loss(contact_pred, forces_target)
        # self.mse = metrics._mse_loss(contact_pred, contact_cop_grf)
        loss = self.mse_weight * self.mse
    
        # optimize
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()

    def _losses_logging(self):
        item, epoch = self.item + 1, self.epoch + 1
        print("[{}/{}][{}/{}]   MSE = {:.5e}  ".format(item, self.nitems, epoch, self.nepochs, self.mse))
        training_loss.append("[{}/{}][{}/{}]   MSE = {:.5e}\n".format(item, self.nitems, epoch, self.nepochs, self.mse))
    
    def _validation(self):
        print("Validation #{}".format(self.iter))
        
        # Modified for GroundLink
        # Make predictions
        with torch.no_grad():
            contact_pred = []
            for poses in self.validset["poses"].split(128):
                contact_pred.append(self.model.GRFs(poses.float().to(self.device)).detach().cpu())
            forces_pred = torch.cat(contact_pred)

            target = torch.cat((self.validset["cop"], self.validset["grf"]),3)
            rmse = metrics.RMSE(forces_pred, target=target).item()
            mse = metrics._mse_loss(forces_pred, target)

            torch.save(dict(model=self.model.state_dict()), self.ckp)
            print("RMSE = " + str(rmse))
            print("MSE = " + str(mse))
            validation_loss.append("Validation #{}   RMSE = {}    MSE = {}\n".format(self.iter, rmse, mse))

    def _test(self):
        print("Test #{}".format(self.iter))
		
		# Modified for GroundLink
		# Make predictions
        vGRF_L, vGRF_R, vRPE = [], [], []
        recon_metric = torch.nn.MSELoss(reduction='mean')
        with torch.no_grad():
            for i, sample in enumerate(list(self.testset)):
                poses = sample["poses"].to(self.device)
                # cop   = sample["CoP"]
                grf   = sample["GRF"].to(self.device)
                mass  = sample["mass"].to(self.device)
                weight  = (9.81 * mass).repeat(poses.shape[0])
                GRFs_pred = self.model.GRFs(poses.float().unsqueeze(0)).squeeze(0)

                # Simulation
                z_true 	= poses[:,0,2] # B, T
                seq_len = z_true.shape[0]
                z0      = z_true[0].clone() # Initial solution
                v0      = torch.zeros_like(z0)
                z, v    = [], []
                z.append(z0)
                v.append(v0)
                pred_grf_sum = GRFs_pred[...,-1].sum(-1) *(1e3/weight)
                dt = 1/90
                for f in range(seq_len-1):
                    rgt     = pred_grf_sum[f]
                    vt      = v[-1] + (rgt - 1) * dt
                    zt      = z[-1] + vt * dt
                    if f % 250 == 0:
                        v.append(v0)
                        z.append(z_true[f])
                    else:
                        v.append(vt)
                        z.append(zt)
                z_sim   = torch.stack(z, dim=0)

                vGRF_L.append(metrics._mse_loss(GRFs_pred[:,0,-1], grf[:,0,-1]))
                vGRF_R.append(metrics._mse_loss(GRFs_pred[:,1,-1], grf[:,1,-1]))
                vRPE.append(recon_metric(z_sim*1e3, z_true*1e3)*1e-3)

            print("vGRF_L = {:.2f}, vGRF_R = {:.2f}  ".format(torch.stack(vGRF_L).mean().item(),
                                                                            torch.stack(vGRF_R).mean().item(),))
            print("vRPE = {:.2f}, ".format(torch.stack(vRPE).mean().item()))

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    parser = ArgumentParser()
    checkpointpath = './checkpoint'
    checkpointname = 'baseline'
    checkpoint = os.path.join(checkpointpath, checkpointname+'.tar')

    parser.add_argument("-ckp", default=checkpoint, type=Path,                    help="Path to make checkpoint during training ........................ default: 'checkpoint.tar'")
    parser.add_argument("-test_subject", default="S5", type=str,                help="Test subject ID ............................. default: S7")
    parser.add_argument("-device", default="cuda", type=str,                    help="Device used to run training .................................... default: cuda")
    parser.add_argument("-learning_rate", default=3e-5, type=float,             help="Adam optimisation algorithm learning rate ...................... default: 3e-5")
    parser.add_argument("-mse_weight", default=0.002, type=float,                help="MSE loss weight ............................................... default: 0.002")
    parser.add_argument("-batch_size", default=64, type=int,                    help="Batch size ..................................................... default: 64")
    parser.add_argument("-iterations", default=1e5, type=int,                    help="Number of training iterations .................................. default: 1e8")
    parser.add_argument("-split_ratio", default=0.7, type=float,                help="Train/Validation split ratio ................................... default: 0.9")
    parser.add_argument("-sequence_length", default=140, type=int,                help="Training sequences length ...................................... default: 240")
    parser.add_argument("-sequence_overlap", default=139, type=int,                help="Training sequences overlap ................................. default: 239")
    args = parser.parse_args()
    folder_path = "/mimer/NOBACKUP/groups/alvis_cvl/cuole/phys_grd/dataset_" + args.test_subject
    Trainer(**vars(parser.parse_args())).run()
