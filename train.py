import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import csv
from tqdm import tqdm

# Import custom modules
from model import get_model, MODEL_CONFIG, EFFICIENT_MODEL_CONFIG
from data import get_loaders
from caltech_data import get_caltech_train_loader,get_caltech_val_loader
from cifar_data import get_cifar_train_loader,get_cifar_val_loader

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 5e-4,
    "min_lr": 5e-6,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "output_dir": "output",
    "log_interval": 20,
    "save_interval": 10,
}

#####################################
# Utilities for Training & Evaluation
#####################################

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr):
    """Create a cosine learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_lr / TRAIN_CONFIG["learning_rate"], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, filename):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'accuracy': accuracy,
    }
    
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def save_pretrain_checkpoint(model, dataset_name, output_dir='checkpoints'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{dataset_name}_checkpoint.pth')
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint for {dataset_name} at: {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """Load checkpoint from file"""
    if not os.path.isfile(filename):
        print(f"No checkpoint found at {filename}")
        return 0, 0.0
    
    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['accuracy']

def create_optimizer(model, lr, weight_decay):
    """Create optimizer for model"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

def create_scheduler(optimizer, num_epochs, steps_per_epoch, warmup_epochs=5, min_lr=5e-6):
    """Create learning rate scheduler"""
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr
    )


def log_metrics(dataset_name, epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset_name}_log.csv")
    write_header = not os.path.exists(log_file)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", 
                "train_loss", "train_acc1", "train_acc5", 
                "val_loss", "val_acc1", "val_acc5"
            ])
        writer.writerow([
            epoch + 1, 
            train_loss, train_acc1, train_acc5, 
            val_loss, val_acc1, val_acc5
        ])


def replace_head(model, num_classes):
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def pretrain_on_dataset(model, train_loader, val_loader, num_classes, args,dataset_name):
    model = replace_head(model, num_classes)
    model.to(args.device)

    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    print(f"\n=== Pretraining on {dataset_name} ===")
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, args.device
        )
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, args.device)

        best_acc = max(val_acc1, best_acc)
        print(f"[{dataset_name}] Epoch {epoch+1}: Acc@1={val_acc1:.2f}% | Best={best_acc:.2f}%")
        log_metrics(dataset_name, epoch, train_loss, train_acc1,train_acc5, val_loss, val_acc1,val_acc5)
    
    save_pretrain_checkpoint(model, dataset_name)

    return model



#####################################
# Training and Evaluation Functions
#####################################

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, mixup_fn=None):
    """Train model for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Initialize tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", 
                leave=True, ncols=100, unit="batch")
    
    for images, target in pbar:
        # Move data to device
        images = images.to(device)
        target = target.to(device)
        
        # Apply mixup or cutmix if available
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)
        
        # Forward pass
        output = model(images)
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        if mixup_fn is None:  # Only measure accuracy if not using mixup/cutmix
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Loss': f"{losses.avg:.4f}",
            'Acc@1': f"{top1.avg:.2f}%" if mixup_fn is None else "N/A",
            'LR': f"{lr:.6f}"
        })
    
    return losses.avg, top1.avg, top5.avg

def validate(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # Initialize tqdm progress bar
    pbar = tqdm(val_loader, desc="Validation", leave=True, ncols=100, unit="batch")
    
    with torch.no_grad():
        for images, target in pbar:
            # Move data to device
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses.avg:.4f}",
                'Acc@1': f"{top1.avg:.2f}%",
                'Acc@5': f"{top5.avg:.2f}%"
            })
    
    print(f"* Validation: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%")
    return losses.avg, top1.avg, top5.avg

#####################################
# Main Training Loop
#####################################

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.tag if args.tag else '')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = get_model(model_name='swin_t', efficient=args.efficient, num_classes=1000)

    # === Pretraining Stage: CIFAR ===
    if args.pretrain_cifar and not args.skip_pretrain:
        print("creating dataloaders for cifar100...")
        cifar_train_loader = get_cifar_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
        cifar_val_loader = get_cifar_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
        model = pretrain_on_dataset(model, cifar_train_loader, cifar_val_loader, num_classes=100, args=args, dataset_name='cifar100')

    # === Pretraining Stage: Caltech ===
    if args.pretrain_caltech and not args.skip_pretrain:
        print("creating dataloaders for caltech256...")
        caltech_train_loader = get_caltech_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
        caltech_val_loader = get_caltech_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
        model = pretrain_on_dataset(model, caltech_train_loader, caltech_val_loader, num_classes=257, args=args, dataset_name='caltech256')

    # === Final Training Stage ===
    print("creating dataloaders for tinyimagenet...")
    train_loader, val_loader, mixup_fn = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=MODEL_CONFIG["img_size"],
        use_mixup=args.mixup
    )
    model = replace_head(model, num_classes=200)
    model = model.to(args.device)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )
    
    # Create loss function
    if args.mixup:
        # Use soft target cross entropy loss for mixup/cutmix
        criterion = SoftTargetCrossEntropy()
    else:
        # Use label smoothing cross entropy loss
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    
    # Optionally resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume)
            print(f"Resumed from epoch {start_epoch} with accuracy {best_acc:.2f}%")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Evaluation only
    if args.evaluate:
        print("Running evaluation")
        validate(model, val_loader, criterion, args.device)
        return
    
    # Print training configuration
    print(f"Starting training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Using mixup: {args.mixup}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, args.device, mixup_fn)
        
        # Evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, args.device)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)
        
        # Log metrics to CSV
        lr = scheduler.get_last_lr()[0]
        log_metrics("main", epoch, train_loss, train_acc1, val_loss, val_acc1, log_dir=args.log_dir)

        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train: Loss {train_loss:.4f}, Acc@1 {train_acc1:.2f}%, Acc@5 {train_acc5:.2f}%")
        print(f"  Valid: Loss {val_loss:.4f}, Acc@1 {val_acc1:.2f}%, Acc@5 {val_acc5:.2f}%")
        print(f"  Best accuracy: {best_acc:.2f}%")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc1,
                os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth')
            )
        
        # Always save the best model
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc1,
                os.path.join(output_dir, 'model_best.pth')
            )

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser(description='Swin Transformer for Tiny ImageNet')
    
    # Model parameters
    parser.add_argument('--efficient', action='store_true', help='Use efficient model variant')
    
    # pre-training...
    parser.add_argument('--pretrain_cifar', action='store_true', help='Pretrain on CIFAR first')
    parser.add_argument('--pretrain_caltech', action='store_true', help='Pretrain on Caltech after CIFAR')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip all pretraining stages')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'], help='Number of epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=TRAIN_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=TRAIN_CONFIG['min_lr'], help='Minimum learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=TRAIN_CONFIG['warmup_epochs'], help='Warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG['weight_decay'], help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=TRAIN_CONFIG['label_smoothing'], help='Label smoothing factor')
    parser.add_argument('--device', default=TRAIN_CONFIG['device'], help='Device to use')
    parser.add_argument('--mixup', action='store_true', help='Use mixup and cutmix augmentation')
    
    # Data loading
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    # Checkpointing
    parser.add_argument('--output-dir', default=TRAIN_CONFIG['output_dir'], help='Path to save output')
    parser.add_argument('--tag', default='', help='Tag for the experiment')
    parser.add_argument('--save-interval', type=int, default=TRAIN_CONFIG['save_interval'], help='Save checkpoint every N epochs')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    
    # logs
    parser.add_argument('--log-dir', default='logs', type=str, help='Directory to save all training logs')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)