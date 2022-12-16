import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
#from wavebeat.tcn import TCNModel
from wavebeat.dstcn import dsTCNModel
#from wavebeat.lstm import LSTMModel
#from wavebeat.waveunet import WaveUNetModel
from wavebeat.data import DownbeatDataset

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--model_type', type=str, default='dstcn', help='tcn, lstm, waveunet, or dstcn')
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--beatles_audio_dir', type=str, default='./data')
parser.add_argument('--beatles_annot_dir', type=str, default='./data')
parser.add_argument('--ballroom_audio_dir', type=str, default='./data')
parser.add_argument('--ballroom_annot_dir', type=str, default='./data')
parser.add_argument('--hainsworth_audio_dir', type=str, default='./data')
parser.add_argument('--hainsworth_annot_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
parser.add_argument('--target_factor', type=int, default=256)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--focal_gamma', type=float, default=2.0)
parser.add_argument('--validation_fold', type=int, default=None)

checkpoint_callback = ModelCheckpoint(
    verbose=True,
    monitor='val_loss/Joint F-measure',
    mode='max'
)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
#if temp_args.model_type == 'tcn':
#    parser = TCNModel.add_model_specific_args(parser)
#elif temp_args.model_type == 'lstm':
#    parser = LSTMModel.add_model_specific_args(parser)
#elif temp_args.model_type == 'waveunet':
#    parser = WaveUNetModel.add_model_specific_args(parser)
if temp_args.model_type == 'dstcn':
    parser = dsTCNModel.add_model_specific_args(parser)
else:
    raise RuntimeError(f"Invalid model_type: {temp_args.model_type}")

# parse them args
args = parser.parse_args()

datasets = ["beatles", "ballroom", "hainsworth", "rwc_popular"]

# set the seed
pl.seed_everything(42)

#
args.default_root_dir = os.path.join("lightning_logs", "full")
print(args.default_root_dir)

# create the trainer
trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])

# setup the dataloaders
train_datasets = []
val_datasets = []

for dataset in datasets:
    if dataset == "beatles":
        audio_dir = args.beatles_audio_dir
        annot_dir = args.beatles_annot_dir
    elif dataset == "ballroom":
        audio_dir = args.ballroom_audio_dir
        annot_dir = args.ballroom_annot_dir
    elif dataset == "hainsworth":
        audio_dir = args.hainsworth_audio_dir
        annot_dir = args.hainsworth_annot_dir
    elif dataset == "rwc_popular":
        audio_dir = args.rwc_popular_audio_dir
        annot_dir = args.rwc_popular_annot_dir

    train_dataset = DownbeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    target_factor=args.target_factor,
                                    subset="train",
                                    fraction=args.train_fraction,
                                    augment=args.augment,
                                    half=True if args.precision == 16 else False,
                                    preload=args.preload,
                                    length=args.train_length,
                                    dry_run=args.dry_run,
                                    validation_fold=args.validation_fold)
    train_datasets.append(train_dataset)

    val_dataset = DownbeatDataset(audio_dir,
                                 annot_dir,
                                 dataset=dataset,
                                 audio_sample_rate=args.audio_sample_rate,
                                 target_factor=args.target_factor,
                                 subset="val",
                                 augment=False,
                                 half=True if args.precision == 16 else False,
                                 preload=args.preload,
                                 length=args.eval_length,
                                 dry_run=args.dry_run,
                                 validation_fold=args.validation_fold)
    val_datasets.append(val_dataset)

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)

train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                shuffle=args.shuffle,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False)    

# create the model with args
dict_args = vars(args)
dict_args["nparams"] = 2
dict_args["target_sample_rate"] = args.audio_sample_rate / args.target_factor

if args.model_type == 'tcn':
    model = TCNModel(**dict_args)
    rf = model.compute_receptive_field()
    print(f"Model has receptive field of {(rf/args.sample_rate)*1e3:0.1f} ms ({rf}) samples")
elif args.model_type == 'lstm':
    model = LSTMModel(**dict_args)
elif args.model_type == 'waveunet':
    model = WaveUNetModel(**dict_args)
elif args.model_type == 'dstcn':
    model = dsTCNModel(**dict_args)

# summary: https://velog.io/@rapidrabbit76/torchsummary-Forwardbackward-pass-size 
torchsummary.summary(model, [(1,args.train_length)], device="cpu")  #input_size = [(1,args.train_length)],
#MJ: The wavebeat model summary is as follows: args.train_length= 2097152=2^21 samples (22050Hz)
# Total params: 2,761,602
# Trainable params: 2,761,602
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 8.00
# Forward/backward pass size (MB): 6024.12
# Params size (MB): 10.53
# Estimated Total Size (MB): 6042.66

# train!
trainer.fit(model, train_dataloader, val_dataloader)