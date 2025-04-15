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
import pandas as pd # Added for reading logs
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for confusion matrix styling
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Added for CM

# Import custom modules
from model import get_model, MODEL_CONFIG, EFFICIENT_MODEL_CONFIG
from data import get_loaders
from caltech_data import get_caltech_train_loader,get_caltech_val_loader
from cifar_data import get_cifar_train_loader,get_cifar_val_loader

# Training Configuration
TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 100, # Reduced for faster demonstration if needed
    "learning_rate": 5e-4,
    "min_lr": 5e-6,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "output_dir": "output",
    "log_interval": 20,
    "save_interval": 10, # Reduced for faster demonstration
    "log_dir": "logs", # Added log dir to config
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
        if self.count > 0:
             self.avg = self.sum / self.count
        else:
             self.avg = 0


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if batch_size == 0:
            return [torch.tensor(0.0) for _ in topk]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr, base_lr):
    """Create a cosine learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        # Ensure the final learning rate doesn't go below min_lr
        return max(min_lr / base_lr, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, accuracy, filename):
    """Save model checkpoint"""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Created directory: {os.path.dirname(filename)}")

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
    """Save model checkpoint after pretraining"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{dataset_name}_checkpoint.pth')
    # Save only the model state dict for pretraining checkpoints
    torch.save(model.state_dict(), path)
    print(f"Saved pretraining checkpoint for {dataset_name} at: {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None, load_optimizer_scheduler=True):
    """Load checkpoint from file"""
    if not filename or not os.path.isfile(filename):
        print(f"No checkpoint found at {filename}")
        return 0, 0.0

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location='cpu')

    # Handle both full checkpoints and state_dict-only checkpoints
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume it's just a state_dict
        model.load_state_dict(checkpoint)
        # If only state_dict loaded, cannot resume optimizer/scheduler/epoch
        print("Loaded model state_dict only. Cannot resume optimizer, scheduler, or epoch.")
        return 0, checkpoint.get('accuracy', 0.0) # Return 0 epoch, try to get accuracy

    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)

    if load_optimizer_scheduler:
        if optimizer is not None and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                 print(f"Could not load optimizer state: {e}. Continuing without loading optimizer.")


        if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
             try:
                 scheduler.load_state_dict(checkpoint['scheduler'])
             except Exception as e:
                 print(f"Could not load scheduler state: {e}. Continuing without loading scheduler.")


    else:
        print("Skipping loading optimizer and scheduler state.")

    return epoch, accuracy

def create_optimizer(model, lr, weight_decay):
    """Create optimizer for model"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

def create_scheduler(optimizer, num_epochs, steps_per_epoch, base_lr, warmup_epochs=5, min_lr=5e-6):
    """Create learning rate scheduler"""
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr=min_lr,
        base_lr=base_lr # Pass base_lr here
    )


def log_metrics(dataset_name, epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, log_dir="logs"):
    """Logs training and validation metrics to a CSV file."""
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
            epoch + 1, # Log 1-based epoch
            f"{train_loss:.4f}" if train_loss is not None else "N/A",
            f"{train_acc1:.2f}" if train_acc1 is not None else "N/A",
            f"{train_acc5:.2f}" if train_acc5 is not None else "N/A",
            f"{val_loss:.4f}" if val_loss is not None else "N/A",
            f"{val_acc1:.2f}" if val_acc1 is not None else "N/A",
            f"{val_acc5:.2f}" if val_acc5 is not None else "N/A"
        ])


def replace_head(model, num_classes):
    """Replaces the classification head of the model."""
    in_features = 0
    if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
         in_features = model.head.in_features
    elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'): # common alternative name
         in_features = model.fc.in_features
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear): # Another common name
         in_features = model.classifier.in_features
    elif hasattr(model, 'num_features'): # Timm models often have this property
        in_features = model.num_features
    else:
        raise AttributeError("Cannot determine the input features of the model's classification head. Tried 'head', 'fc', 'classifier'.")

    # Replace the head
    if hasattr(model, 'head'):
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        # Fallback for models where head is not explicitly named 'head', 'fc' or 'classifier'
        # This might need adjustment based on the specific architecture if get_model returns something unusual
         print("Warning: Replacing head using a generic approach based on Timm's num_features. Ensure this is correct for the model.")
         model.head = nn.Linear(in_features, num_classes) # Assume we can add a 'head' attribute

    print(f"Replaced model head with a new one for {num_classes} classes.")
    return model


#####################################
# Plotting Functions
#####################################

def plot_metrics(log_file, output_dir, dataset_name):
    """Plots loss and accuracy curves from a log file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}. Skipping plotting.")
        return

    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}. Skipping plotting.")
        return

    if df.empty:
        print(f"Log file {log_file} is empty. Skipping plotting.")
        return

    plt.style.use('seaborn-v0_8-grid') # Use a nice style
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    if 'train_loss' in df.columns and pd.to_numeric(df['train_loss'], errors='coerce').notna().any():
        ax1.plot(df['epoch'], pd.to_numeric(df['train_loss'], errors='coerce'), label='Train Loss', color=color, linestyle='--')
    if 'val_loss' in df.columns and pd.to_numeric(df['val_loss'], errors='coerce').notna().any():
        ax1.plot(df['epoch'], pd.to_numeric(df['val_loss'], errors='coerce'), label='Validation Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Plot Accuracy
    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    if 'train_acc1' in df.columns and pd.to_numeric(df['train_acc1'], errors='coerce').notna().any():
         ax2.plot(df['epoch'], pd.to_numeric(df['train_acc1'], errors='coerce'), label='Train Acc@1', color=color, linestyle='--')
    if 'val_acc1' in df.columns and pd.to_numeric(df['val_acc1'], errors='coerce').notna().any():
         ax2.plot(df['epoch'], pd.to_numeric(df['val_acc1'], errors='coerce'), label='Validation Acc@1', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower left')

    plt.title(f'{dataset_name} - Training & Validation Metrics')
    fig.tight_layout() # otherwise the right y-label is slightly clipped

    # Save plot
    plot_filename = os.path.join(output_dir, f"{dataset_name}_metrics_plot.png")
    plt.savefig(plot_filename)
    print(f"Metrics plot saved to {plot_filename}")
    plt.close(fig) # Close the figure to free memory

def plot_confusion_matrix(all_preds, all_targets, num_classes, output_dir, dataset_name):
    """Computes and plots the confusion matrix."""
    if all_preds is None or all_targets is None:
        print(f"No prediction data available for {dataset_name}. Skipping confusion matrix.")
        return

    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))

    # Determine figure size based on number of classes
    figsize = max(8, num_classes // 5) # Adjust divisor as needed

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))

    # Determine whether to show values based on matrix size
    show_values = num_classes <= 30 # Only show values for smaller matrices

    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d' if show_values else None) # Only show numbers if show_values is True

    plt.title(f'{dataset_name} - Confusion Matrix')
    plt.tight_layout()

    # Save plot
    cm_filename = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plt.savefig(cm_filename)
    print(f"Confusion matrix saved to {cm_filename}")
    plt.close(fig) # Close the figure


#####################################
# Pretraining Function
#####################################

def pretrain_on_dataset(model, train_loader, val_loader, num_classes, args, dataset_name, output_dir, log_dir):
    """Pretrains the model on a given dataset (CIFAR or Caltech)."""
    print(f"\n=== Pretraining Stage: {dataset_name} ===")
    model = replace_head(model, num_classes) # Ensure head matches dataset
    model.to(args.device)

    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs, # Use main epoch count for pretraining? Or specific pretrain epochs? Using main for now.
        steps_per_epoch=steps_per_epoch,
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )
    # Use label smoothing for pretraining as well
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(args.device)

    best_acc = 0.0
    for epoch in range(args.epochs): # Use same number of epochs as main training?
        print(f"\n--- {dataset_name} Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, args.device, args=args
        )
        val_loss, val_acc1, val_acc5, _, _ = validate( # Get metrics only during training loop
             model, val_loader, criterion, args.device, return_preds_targets=False
        )

        # Log metrics for this pretraining stage
        log_metrics(dataset_name, epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, log_dir=log_dir)

        # Save best model based on validation accuracy for this stage
        if val_acc1 > best_acc:
            best_acc = val_acc1
            save_pretrain_checkpoint(model, dataset_name, output_dir=os.path.join(output_dir, 'checkpoints'))
            print(f"[{dataset_name}] New best accuracy: {best_acc:.2f}% (Epoch {epoch+1}). Checkpoint saved.")
        else:
             print(f"[{dataset_name}] Epoch {epoch+1}: Acc@1={val_acc1:.2f}% | Best={best_acc:.2f}%")


    print(f"--- Finished Pretraining on {dataset_name} ---")

    # --- Plotting and Final Validation for this Stage ---
    log_file = os.path.join(log_dir, f"{dataset_name}_log.csv")
    plot_metrics(log_file, output_dir, dataset_name)

    # Load the best checkpoint for this stage for final validation and CM
    best_checkpoint_path = os.path.join(output_dir, 'checkpoints', f'{dataset_name}_checkpoint.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best {dataset_name} model for final validation...")
        # Load only state_dict, don't need optimizer/scheduler here
        load_checkpoint(model, filename=best_checkpoint_path, load_optimizer_scheduler=False)
    else:
        print(f"Warning: Best checkpoint {best_checkpoint_path} not found. Using model from last epoch for validation.")


    print(f"Running final validation on {dataset_name} to generate Confusion Matrix...")
    final_val_loss, final_val_acc1, final_val_acc5, all_preds, all_targets = validate(
        model, val_loader, criterion, args.device, return_preds_targets=True
    )
    print(f"Final {dataset_name} Validation: Loss={final_val_loss:.4f}, Acc@1={final_val_acc1:.2f}%, Acc@5={final_val_acc5:.2f}%")

    if all_preds is not None and all_targets is not None:
         plot_confusion_matrix(all_preds.cpu().numpy(), all_targets.cpu().numpy(), num_classes, output_dir, dataset_name)
    else:
         print(f"Could not generate confusion matrix for {dataset_name} due to missing prediction data.")


    # Important: Return the model (potentially loaded with best weights)
    return model


#####################################
# Training and Evaluation Functions
#####################################

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device, args, mixup_fn=None):
    """Train model for one epoch"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    steps_per_epoch = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
                leave=False, ncols=100, unit="batch") # Changed leave to False for cleaner nested loops

    for batch_idx, (images, target) in enumerate(pbar):
        # Move data to device
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Apply mixup or cutmix if available
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        # Forward pass
        output = model(images)

        # Handle cases where mixup changes target format
        if mixup_fn is not None and len(target.shape) > 1:
             loss = criterion(output, target) # SoftTargetCrossEntropy handles smoothed labels
             # Accuracy calculation is ambiguous with mixup, often skipped or calculated differently
             acc1, acc5 = [torch.tensor(0.0), torch.tensor(0.0)] # Placeholder
        else:
             loss = criterion(output, target) # Standard CE or LabelSmoothing
             acc1, acc5 = accuracy(output, target, topk=(1, 5))


        # Update meters
        losses.update(loss.item(), images.size(0))
        if mixup_fn is None: # Only update accuracy if not using mixup
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adjust learning rate based on step, not epoch (important for cosine schedule with warmup)
        scheduler.step()

        # Update progress bar
        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'Loss': f"{losses.avg:.4f}",
            'Acc@1': f"{top1.avg:.2f}%" if mixup_fn is None else "N/A",
            'LR': f"{lr:.6f}"
        })

    # Return average metrics for the epoch
    return losses.avg, top1.avg, top5.avg

def validate(model, val_loader, criterion, device, return_preds_targets=False):
    """Evaluate model on validation set"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_preds_list = []
    all_targets_list = []

    pbar = tqdm(val_loader, desc="Validation", leave=False, ncols=100, unit="batch") # Changed leave to False

    with torch.no_grad():
        for images, target in pbar:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            if return_preds_targets:
                 preds = torch.argmax(output, dim=1)
                 all_preds_list.append(preds.cpu()) # Move to CPU immediately
                 all_targets_list.append(target.cpu()) # Move to CPU immediately


            pbar.set_postfix({
                'Loss': f"{losses.avg:.4f}",
                'Acc@1': f"{top1.avg:.2f}%",
                'Acc@5': f"{top5.avg:.2f}%"
            })

    all_preds = None
    all_targets = None
    if return_preds_targets and len(all_preds_list) > 0:
         all_preds = torch.cat(all_preds_list)
         all_targets = torch.cat(all_targets_list)


    # No need to print here if called during training loop, will be printed in main loop
    # If called standalone (e.g., for final CM), the calling function should print
    return losses.avg, top1.avg, top5.avg, all_preds, all_targets

#####################################
# Main Training Loop
#####################################

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True # Enable cuDNN benchmark for speed

    # --- Setup Directories ---
    base_output_dir = args.output_dir
    experiment_output_dir = os.path.join(base_output_dir, args.tag) if args.tag else base_output_dir
    log_dir = args.log_dir # Use dedicated log dir
    os.makedirs(experiment_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_output_dir, 'checkpoints') # Subdir for checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Output Directory: {experiment_output_dir}")
    print(f"Log Directory: {log_dir}")
    print(f"Checkpoint Directory: {checkpoint_dir}")


    # --- Create Model ---
    print("Initializing model...")
    model = get_model(model_name='swin_t', efficient=args.efficient)
    print(f"Model: swin_t (Efficient: {args.efficient})")


    # --- Pretraining Stages ---
    if not args.skip_pretrain:
         if args.pretrain_cifar:
             print("\n>>> Starting CIFAR-100 Pretraining Stage <<<")
             cifar_train_loader = get_cifar_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
             cifar_val_loader = get_cifar_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
             model = pretrain_on_dataset(model, cifar_train_loader, cifar_val_loader,
                                         num_classes=100, args=args, dataset_name='cifar100',
                                         output_dir=experiment_output_dir, log_dir=log_dir)
             print("\n>>> Finished CIFAR-100 Pretraining Stage <<<")


         if args.pretrain_caltech:
             print("\n>>> Starting Caltech-256 Pretraining Stage <<<")
             # If CIFAR pretraining happened, the model already has a head for 100 classes.
             # If not, it has the original 1000 class head. `pretrain_on_dataset` handles replacement.
             caltech_train_loader = get_caltech_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
             caltech_val_loader = get_caltech_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
             model = pretrain_on_dataset(model, caltech_train_loader, caltech_val_loader,
                                         num_classes=257, args=args, dataset_name='caltech256',
                                         output_dir=experiment_output_dir, log_dir=log_dir)
             print("\n>>> Finished Caltech-256 Pretraining Stage <<<")
    else:
        print("Skipping all pretraining stages as requested.")


    # --- Final Training Stage: Tiny ImageNet ---
    print("\n>>> Starting Final Training Stage: Tiny ImageNet <<<")
    print("Creating dataloaders for Tiny ImageNet...")
    train_loader, val_loader, mixup_fn = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=MODEL_CONFIG["img_size"],
        use_mixup=args.mixup
    )

    print("Replacing model head for Tiny ImageNet (200 classes)...")
    model = replace_head(model, num_classes=200)
    model = model.to(args.device)

    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (final stage): {num_params:,}")

    # Create optimizer and scheduler for the final stage
    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    steps_per_epoch_main = len(train_loader)
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch_main,
        base_lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )

    # Create loss function for the final stage
    if args.mixup:
        criterion = SoftTargetCrossEntropy().to(args.device)
        print("Using Mixup/Cutmix augmentation with SoftTargetCrossEntropy loss.")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(args.device)
        print(f"Using Label Smoothing Cross Entropy loss (smoothing={args.label_smoothing}).")

    # Optionally resume from a final stage checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(checkpoint_dir, args.resume)
        if os.path.isfile(resume_path):
            print(f"Attempting to resume final stage training from: {resume_path}")
            # Load optimizer and scheduler state when resuming main training
            start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, resume_path, load_optimizer_scheduler=True)
            print(f"Resumed final stage from epoch {start_epoch}. Previous best accuracy: {best_acc:.2f}%")
            start_epoch = start_epoch # Checkpoint saves epoch+1, so start from the returned value
        else:
            print(f"Resume checkpoint not found at '{resume_path}'. Starting final training from scratch.")

    # Evaluation only mode for the final model
    if args.evaluate:
        print("--- Running Evaluation Only Mode ---")
        eval_checkpoint_path = args.resume if args.resume else os.path.join(experiment_output_dir, 'model_best.pth')
        if os.path.isfile(eval_checkpoint_path):
             print(f"Loading model from: {eval_checkpoint_path} for evaluation...")
             # Don't load optimizer/scheduler for evaluation
             load_checkpoint(model, filename=eval_checkpoint_path, load_optimizer_scheduler=False)
             print("Running validation...")
             val_loss, val_acc1, val_acc5, all_preds, all_targets = validate(
                 model, val_loader, criterion, args.device, return_preds_targets=True
             )
             print(f"\nEvaluation Results (Tiny ImageNet):")
             print(f"  Loss: {val_loss:.4f}")
             print(f"  Acc@1: {val_acc1:.2f}%")
             print(f"  Acc@5: {val_acc5:.2f}%")

             # Plot confusion matrix for evaluation
             if all_preds is not None and all_targets is not None:
                 plot_confusion_matrix(all_preds.cpu().numpy(), all_targets.cpu().numpy(), 200, experiment_output_dir, "main_eval")
             else:
                 print("Could not generate confusion matrix due to missing prediction data.")

        else:
             print(f"Evaluation checkpoint '{eval_checkpoint_path}' not found. Cannot evaluate.")
        return # Exit after evaluation


    # --- Main Training Loop ---
    print(f"\n--- Starting Final Training Loop (Tiny ImageNet) for {args.epochs - start_epoch} epochs ---")
    print(f"Batch size: {args.batch_size}, Initial LR: {args.lr}, Weight Decay: {args.weight_decay}, Mixup: {args.mixup}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\n--- Tiny ImageNet Epoch {epoch+1}/{args.epochs} ---")

        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, args.device, args, mixup_fn
        )

        # Validate
        val_loss, val_acc1, val_acc5, _, _ = validate( # Don't need preds/targets here
            model, val_loader, criterion, args.device, return_preds_targets=False
        )

        epoch_time = time.time() - epoch_start

        # Check if current epoch is best
        is_best = val_acc1 > best_acc
        if is_best:
            old_best = best_acc
            best_acc = val_acc1
            print(f"*** New Best Accuracy: {best_acc:.2f}% (Improved from {old_best:.2f}%) ***")
        else:
            print(f"Validation Acc@1: {val_acc1:.2f}% (Best: {best_acc:.2f}%)")


        # Log metrics to CSV for the main training stage
        log_metrics("main", epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, log_dir=log_dir)

        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, val_acc1,
                os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
            )

        # Always save the best model
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_acc, # Save best_acc here
                os.path.join(experiment_output_dir, 'model_best.pth') # Save best model in parent dir
            )

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Train -> Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%" if not args.mixup else f"  Train -> Loss: {train_loss:.4f}, Acc@1: N/A (Mixup)")
        print(f"  Valid -> Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")


    print(f"\n--- Finished Final Training Stage (Tiny ImageNet) ---")
    print(f"Best validation accuracy achieved: {best_acc:.2f}%")

    # --- Final Plotting and Confusion Matrix for Main Training ---
    # Plot loss/accuracy curves for the main training
    main_log_file = os.path.join(log_dir, "main_log.csv")
    plot_metrics(main_log_file, experiment_output_dir, "main")

    # Load the *best* model for the final confusion matrix
    best_model_path = os.path.join(experiment_output_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final confusion matrix...")
        # Create a fresh instance or reload into the current one
        # Re-create model to ensure clean state if needed, though loading state_dict should be fine
        final_model = get_model(model_name='swin_t', efficient=args.efficient) # Head already replaced
        # final_model = replace_head(final_model, num_classes=200)
        
        load_checkpoint(final_model, filename=best_model_path, load_optimizer_scheduler=False)
        final_model.to(args.device)
        final_model.eval()

        print("Running validation on best model for final confusion matrix...")
        _, _, _, all_preds, all_targets = validate(
            final_model, val_loader, criterion, args.device, return_preds_targets=True
        )

        if all_preds is not None and all_targets is not None:
            plot_confusion_matrix(all_preds.cpu().numpy(), all_targets.cpu().numpy(), 200, experiment_output_dir, "main_best_model")
        else:
             print("Could not generate final confusion matrix due to missing prediction data.")

    else:
        print(f"Best model checkpoint '{best_model_path}' not found. Cannot generate confusion matrix for the best model.")

    print("\n>>> All Training Stages Complete <<<")


def parse_args():
    parser = argparse.ArgumentParser(description='Swin Transformer Training with Pretraining Options')

    # --- Model ---
    parser.add_argument('--model-name', type=str, default='swin_t', help='Name of the model architecture (e.g., swin_t)')
    parser.add_argument('--efficient', action='store_true', help='Use efficient model variant (if available)')
    # parser.add_argument('--no-pretrained', action='store_true', help='Do not use ImageNet pretrained weights initially')

    # --- Pre-training ---
    parser.add_argument('--pretrain-cifar', action='store_true', help='Pretrain on CIFAR-100 first')
    parser.add_argument('--pretrain-caltech', action='store_true', help='Pretrain on Caltech-256 (after CIFAR if specified, otherwise from ImageNet)')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip all pretraining stages and train directly on Tiny ImageNet')
    # parser.add_argument('--pretrain-epochs', type=int, default=30, help='Number of epochs for each pretraining stage (if different from main epochs)') # Optional: Separate epoch control

    # --- Main Training ---
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'], help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'], help='Number of epochs to train')
    parser.add_argument('--lr', '--learning-rate', type=float, default=TRAIN_CONFIG['learning_rate'], help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=TRAIN_CONFIG['min_lr'], help='Minimum learning rate for scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=TRAIN_CONFIG['warmup_epochs'], help='Number of warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG['weight_decay'], help='Optimizer weight decay')
    parser.add_argument('--label-smoothing', type=float, default=TRAIN_CONFIG['label_smoothing'], help='Label smoothing factor (if not using mixup)')
    parser.add_argument('--mixup', action='store_true', help='Use mixup and cutmix augmentation (disables label smoothing)')

    # --- Data & Device ---
    parser.add_argument('--img-size', type=int, default=MODEL_CONFIG['img_size'], help='Input image size') # Make img_size configurable if needed
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', default=TRAIN_CONFIG['device'], help='Device to use (e.g., "cuda", "cpu")')

    # --- Checkpointing & Logging ---
    parser.add_argument('--output-dir', default=TRAIN_CONFIG['output_dir'], help='Base directory to save checkpoints and logs')
    parser.add_argument('--log-dir', default=TRAIN_CONFIG['log_dir'], help='Directory within output-dir to save CSV logs and plots')
    parser.add_argument('--tag', default='', type=str, help='Optional tag for experiment directory name')
    parser.add_argument('--save-interval', type=int, default=TRAIN_CONFIG['save_interval'], help='Save checkpoint every N epochs during main training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint to resume main training (or for evaluation)')

    # --- Misc ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation only on the validation set (requires --resume or finds model_best.pth)')

    args = parser.parse_args()

    # Set device based on argument
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # If mixup is used, disable label smoothing effect by setting it to 0
    if args.mixup:
        args.label_smoothing = 0.0
        print("Mixup enabled, label smoothing set to 0.")

    # Ensure log_dir is inside output_dir unless absolute path is given
    if not os.path.isabs(args.log_dir):
        args.log_dir = os.path.join(args.output_dir, args.tag if args.tag else '', args.log_dir)


    return args


if __name__ == '__main__':
    args = parse_args()
    # Ensure necessary directories exist based on final paths
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
