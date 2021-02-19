import os
import torch
import mir_eval
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from beat.utils import center_crop, causal_crop
from beat.plot import plot_activations
from beat.loss import GlobalMSELoss
from beat.peak import find_beats
from beat.filter import FIRFilter

class Base(pl.LightningModule):
    """ Base module with train and validation loops.

        Args:
            nparams (int): Number of conditioning parameters.
            lr (float, optional): Learning rate. Default: 3e-4
            train_loss (str, optional): Training loss function from ['l1', 'stft', 'l1+stft']. Default: 'l1+stft'
            save_dir (str): Path to save audio examples from validation to disk. Default: None
            num_examples (int, optional): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                    lr = 3e-4, 
                    save_dir = None,
                    num_examples = 4,
                    **kwargs):
        super(Base, self).__init__()
        self.save_hyperparameters()

        # these lines need to be commented out when trying
        # to jit these models in `export.py`
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.gmse = GlobalMSELoss()

    def forward(self, x, p):
        pass

    @torch.jit.unused   
    def training_step(self, batch, batch_idx):
        input, target = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # apply lowpass filters
        pred_beats, _ = self.beat_filter(pred[...,0:1,:], target[...,0:1,:])
        pred_downbeats, _ = self.downbeat_filter(pred[...,1:2,:], target[...,1:2,:]) 

        # combine back
        pred = torch.cat((pred_beats, pred_downbeats), dim=1)
        #target = torch.cat((target_beats, target_downbeats), dim=1)

        # crop the input and target signals
        if self.hparams.causal:
            target = causal_crop(target, pred.shape[-1])
        else:
            target = center_crop(target, pred.shape[-1])

        # compute the error using appropriate loss      
        loss, _, _ = self.gmse(pred, target)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # apply lowpass filters
        pred_beats, _ = self.beat_filter(pred[...,0:1,:], target[...,0:1,:])
        pred_downbeats, _ = self.downbeat_filter(pred[...,1:2,:], target[...,1:2,:]) 

        # combine back (we don't filter the target)
        pred = torch.cat((pred_beats, pred_downbeats), dim=1)
        #target = torch.cat((target_beats, target_downbeats), dim=1)

        # crop the input and target signals
        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        # compute the validation error using all losses
        #bce_loss = self.bce(pred, target_crop)
        #l1_loss = self.l1(pred, target_crop)
        #l2_loss = self.l2(pred, target_crop)
        gmse_loss, _, _ = self.gmse(pred, target_crop)

        self.log('val_loss', gmse_loss)
        #self.log('val_loss/L1', l1_loss)
        #self.log('val_loss/L2', l2_loss)
        #self.log('val_loss/Beat', pos)
        #self.log('val_loss/No Beat', neg)

        # move tensors to cpu for logging
        outputs = {
            "input" : input_crop.cpu().numpy(),
            "target": target_crop.cpu().numpy(),
            "pred"  : pred.cpu().numpy()}

        return outputs

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : []}

        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx,...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        # compute metrics 
        beat_f1_scores = []
        downbeat_f1_scores = []
        for idx in np.arange(len(outputs["input"])):
            t = outputs["target"][idx].squeeze()
            p = outputs["pred"][idx].squeeze()

            # separate the beats and downbeat activations
            t_beats = t[0,:]
            t_downbeats = t[1,:]
            p_beats = p[0,:]
            p_downbeats = p[1,:]

            ref_beats, est_beats, _ = find_beats(t_beats, 
                                                 p_beats, 
                                                 beat_type="beat",
                                                 sample_rate=self.hparams.sample_rate)
            ref_downbeats, est_downbeats, _ = find_beats(t_downbeats, 
                                                         p_downbeats, 
                                                         beat_type="downbeat",
                                                         sample_rate=self.hparams.sample_rate)
            # evaluate beats - trim beats before 5 seconds.
            ref_beats = mir_eval.beat.trim_beats(ref_beats)
            est_beats = mir_eval.beat.trim_beats(est_beats)
            scores = mir_eval.beat.evaluate(ref_beats, est_beats)
            beat_f1_scores.append(scores["F-measure"])

            # evaluate downbeats - trim beats before 5 seconds.
            ref_downbeats = mir_eval.beat.trim_beats(ref_downbeats)
            est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
            scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)
            downbeat_f1_scores.append(scores["F-measure"])

        self.log('val_loss/Beat F-measure', np.mean(beat_f1_scores))
        self.log('val_loss/Downbeat F-measure', np.mean(downbeat_f1_scores))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()

            t_beats = t[0,:]
            t_downbeats = t[1,:]
            p_beats = p[0,:]
            p_downbeats = p[1,:]

            ref_beats, est_beats, est_sm = find_beats(t_beats, 
                                                      p_beats, 
                                                      beat_type="beat",
                                                      sample_rate=self.hparams.sample_rate)

            ref_downbeats, est_downbeats, est_downbeat_sm = find_beats(t_downbeats, 
                                                                       p_downbeats, 
                                                                       beat_type="downbeat",
                                                                       sample_rate=self.hparams.sample_rate)
            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            #self.logger.experiment.add_audio(f"target/{idx}", 
            #                                 t, self.global_step, 
            #                                 sample_rate=self.hparams.sample_rate)
            #self.logger.experiment.add_audio(f"pred+target/{idx}",   
            #                                 (p+t)/2, self.global_step, 
            #                                 sample_rate=self.hparams.sample_rate)

            # log beats plots
            self.logger.experiment.add_image(f"act/{idx}",
                                             plot_activations(ref_beats, 
                                                              est_beats, 
                                                              est_sm,
                                                              self.hparams.sample_rate,
                                                              ref_downbeats=ref_downbeats,
                                                              est_downbeats=est_downbeats,
                                                              est_downbeats_sm=est_downbeat_sm,
                                                              ),
                                             self.global_step)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                torchaudio.save(os.path.join(self.hparams.save_dir, 
                                f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-2)
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser