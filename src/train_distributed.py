#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0

#system
import sys
import os
import random
import logging
import argparse
import time
#general
import numpy as np
from statistics import mean
from tqdm import tqdm
#torch
from sagemaker_training import environment
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
import torch.cuda.amp as amp #automatic mixed precision
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#HF transformer modules
from transformers import SamModel
from transformers import SamProcessor
from datasets import load_from_disk
#custom loss
import monai # for custome DiceCELoss

if __name__ == '__main__':
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as arguments to the script.
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accum_freq", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="facebook/sam-vit-base", help='the HF hub model ID, choose between [facebook/sam-vit-base, facebook/sam-vit-large, facebook/sam-vit-huge]')
    #parameters for DDP
    parser.add_argument("--dist_backend", type=str, default="nccl", help='backend to use for dist. training (default: NVIDIA Collective Communications Library (NCCL))')
    parser.add_argument('--workers', type=int, default=int(os.environ["SM_NUM_GPUS"]), help='number of data loading workers (default: <num_gpus>)')
    parser.add_argument("--num_cpu", type=int, default=int(os.environ["SM_NUM_CPUS"]))
    parser.add_argument("--num_gpu", type=int, default=int(os.environ["SM_NUM_GPUS"]))
    # data, model, and output directories (defaults are stored in SM environment variables)
    # see here for envrionment vars: https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--valid_dir", type=str, default=os.environ["SM_CHANNEL_VALID"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    #set some env variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256' #avoid memory fragmentation
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO' #get detailed output for debugging
    os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
    
    args = parser.parse_args()

    #initialize process group https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#distributed-pytorch-training
    training_env = environment.Environment()
    smdataparallel_enabled = training_env.additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False)
    if smdataparallel_enabled:
        try:
            import smdistributed.dataparallel.torch.torch_smddp
            import smdistributed.dataparallel.torch.distributed as dist
            from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
            args.dist_backend = 'smddp'
        except ImportError: 
            print('smdistributed module not available, falling back to NCCL collectives.')

    #initialize process group
    dist.init_process_group(backend=args.dist_backend,init_method="env://") #use environment variables to get world size and local rank
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger.info(f"Num CPU: {args.num_cpu}")
    logger.info(f"Num GPU: {args.num_gpu}")
    logger.info(f"Rank {rank} in world size of {world_size}...")
    logger.info(f"Distributed process group initiated with {args.dist_backend} backend...(Rank: {rank})")
    
    # LOAD DATA & INITIALIZE DATALOADER
    def get_bounding_box(ground_truth_map):
      # get bounding box from mask
      y_indices, x_indices = np.where(ground_truth_map > 0)
      x_min, x_max = np.min(x_indices), np.max(x_indices)
      y_min, y_max = np.min(y_indices), np.max(y_indices)
      # add perturbation to bounding box coordinates
      H, W = ground_truth_map.shape
      x_min = max(0, x_min - np.random.randint(0, 20))
      x_max = min(W, x_max + np.random.randint(0, 20))
      y_min = max(0, y_min - np.random.randint(0, 20))
      y_max = min(H, y_max + np.random.randint(0, 20))
      bbox = [x_min, y_min, x_max, y_max]
      return bbox
    
    class CustomSAMDataset(Dataset):
      def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    
      def __len__(self):
        return len(self.dataset)
    
      def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs
          
    #instantiate processor, dataset and dataloader
    data_train = load_from_disk(args.train_dir)
    data_valid = load_from_disk(args.valid_dir)
    processor = SamProcessor.from_pretrained(args.model_id) #instantiate the processor associated with SAM
    train_dataset = CustomSAMDataset(dataset=data_train, processor=processor) #instantiate dataset
    valid_dataset = CustomSAMDataset(dataset=data_valid, processor=processor) #instantiate dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        #shuffle=True,
        num_workers=int(args.num_gpu), # one worker per GPU
        pin_memory=True,
        sampler=train_sampler,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        #shuffle=False,
        num_workers=int(args.num_gpu), # one worker per GPU
        pin_memory=True,
        sampler=valid_sampler,
    )

    def l(lstring):
        """
        Log info on main process only.
        """
        if dist.get_rank() == 0:
            logger.info(lstring)
            
    l("distributed dataloaders initialized")
    
    #download model config and weights from HF hub and initialize model class
    model = SamModel.from_pretrained(args.model_id)
    #freeze vision and prompt encoder weights, i.e., make sure we only compute gradients for the maks decoder
    for name, param in model.named_parameters():
      if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)
    l(f"model {args.model_id} downloaded from HF hub...")

    #define optimizer and loss
    optimizer = Adam(model.mask_decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # DiceCE returns weighted sum of Dice and Cross Entropy losses. see here: https://docs.monai.io/en/stable/losses.html#diceceloss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    l("optimizer and loss initialized...")
    
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True) # set find_unused_parameters=True to ensure only mask decoder weights are updated
    l("distributed model initialized...")

    #run training
    training_start_time = time.time()
    l('training started')
    # create a GradScaler object for mixed precision training
    scaler = amp.GradScaler()
    #set model to training mode  
    model.train()   
    
    #Training loop
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()  # record the start time of the epoch
        epoch_losses = []
        batch_idx=0
        for batch in train_dataloader:
            # forward pass is run in mixed precision, this reduces activation memory
            with amp.autocast():
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                  input_boxes=batch["input_boxes"].to(device),
                                  multimask_output=False)
                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                epoch_losses.append(loss.item()) #collect losses (for comparability during validation)
                
            #scale loss by gradient_accum_freq to account for accumulation of gradients
            #see here: https://stackoverflow.com/questions/65842691/final-step-of-pytorch-gradient-accumulation-for-small-datasets/65913698#65913698
            loss_norm = loss / args.gradient_accum_freq
            #accumulate gradients
            scaler.scale(loss_norm).backward() # scales the loss and computes the gradients in mixed precision
            
            # optimize once gradients have accumulated over n=gradient_accum_freq batches or if end of data
            if ((batch_idx + 1) % args.gradient_accum_freq == 0) or (batch_idx + 1 == len(train_dataloader)):
                scaler.step(optimizer)  # update the weights using the scaled gradients
                scaler.update()  # update the GradScaler object for the next iteration
                optimizer.zero_grad()  # re-set gradients to zero for next n=gradient_accum_freq iteration over minibatches
            batch_idx+=1

        #validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = []
        with amp.autocast(): #enable autocast during forward pass (minimal impact as there is no .backward() pass)
            with torch.no_grad(): #disable grad calculation to save memory
                for item in valid_dataloader:
                    outputs = model(pixel_values=item["pixel_values"].to(device),
                                      input_boxes=item["input_boxes"].to(device),
                                      multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = item["ground_truth_mask"].float().to(device)
                    val_loss_item = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                    val_loss.append(val_loss_item.item())
        
        epoch_end_time = time.time()  # Record the end time of the epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate the duration of the epoch
        
        l(f'epoch {epoch + 1} completed...')
        l(f'Loss: training loss: {mean(epoch_losses)}; validation loss: {mean(val_loss)};')
        l(f'Perf: duration: {round(epoch_duration, 2)}; throughput: {round(len(train_dataset)/epoch_duration, 2)};')

        
    training_end_time = time.time()
    training_duration = training_end_time-training_start_time
    l(f'full training completed')
    l(f'total training duration [sec]: {round(training_duration, 2)}')
    l(f'avg training throughput [samples/sec]: {round((args.num_epochs * len(train_dataset))/training_duration, 2)}')

    #save pytorch.bin and config.json such that model can be loaded
    #save trained model
    if dist.get_rank() == 0:
        l('training completed')
        model.module.save_pretrained(args.model_dir) 
        #model.module.save_model(args.model_dir)
        l('model saved')
        dist.destroy_process_group()