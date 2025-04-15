import os
import time
import argparse
import numpy as np
import pandas as pd # Added for reading logs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt # Added for plotting
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # Added for CM

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
        # Prevent division by zero if count is 0
        self.avg = self.sum / self.count if self.count > 0 else 0


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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr, base_lr):
    """Create a cosine learning rate scheduler with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        # Ensure the final learning rate is at least min_lr
        return max(min_lr / base_lr, cosine_decay)

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
    print(f"Saved pretraining checkpoint for {dataset_name} at: {path}")


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """Load checkpoint from file"""
    if not os.path.isfile(filename):
        print(f"No checkpoint found at {filename}")
        return 0, 0.0

    print(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location='cpu')

    # Adjust for potential DataParallel prefix 'module.'
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.` prefix
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


    if optimizer is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
             print(f"Could not load optimizer state: {e}. This might happen if model structure changed.")


    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
         try:
            scheduler.load_state_dict(checkpoint['scheduler'])
         except KeyError as e:
             print(f"Could not load scheduler state: {e}. This might happen if scheduler type changed.")


    start_epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with validation accuracy {accuracy:.2f}%")

    return start_epoch, accuracy


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
            train_loss if train_loss is not None else 'N/A', # Handle None cases
            train_acc1 if train_acc1 is not None else 'N/A',
            train_acc5 if train_acc5 is not None else 'N/A',
            val_loss, val_acc1, val_acc5
        ])


def replace_head(model, num_classes):
    in_features = model.head.in_features # Get input features from existing head
    model.head = nn.Linear(in_features, num_classes)
    print(f"Replaced model head for {num_classes} classes.")
    return model


#####################################
# Plotting Utilities                #
#####################################

def plot_metrics(log_file, output_dir, dataset_name):
    """Plots training/validation loss and accuracy from log file."""
    try:
        df = pd.read_csv(log_file)
        df.replace('N/A', np.nan, inplace=True) # Replace 'N/A' with NaN
        df = df.astype({ # Convert relevant columns to numeric, errors='coerce' turns failures into NaN
             'train_loss': float, 'train_acc1': float, 'train_acc5': float,
             'val_loss': float, 'val_acc1': float, 'val_acc5': float
        }, errors='coerce')


        epochs = df['epoch']

        plt.style.use('seaborn-v0_8-grid') # Use a nice style
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot Loss
        axes[0].plot(epochs, df['train_loss'], 'bo-', label='Train Loss')
        axes[0].plot(epochs, df['val_loss'], 'ro-', label='Validation Loss')
        axes[0].set_title(f'{dataset_name} - Loss vs. Epochs')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Accuracy
        if df['train_acc1'].notna().any(): # Only plot train acc if data exists
             axes[1].plot(epochs, df['train_acc1'], 'bo-', label='Train Accuracy@1')
        axes[1].plot(epochs, df['val_acc1'], 'ro-', label='Validation Accuracy@1')
        # Optionally plot Acc@5
        # if df['train_acc5'].notna().any():
        #     axes[1].plot(epochs, df['train_acc5'], 'b--', label='Train Accuracy@5')
        # axes[1].plot(epochs, df['val_acc5'], 'r--', label='Validation Accuracy@5')
        axes[1].set_title(f'{dataset_name} - Accuracy vs. Epochs')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f'{dataset_name}_metrics_plot.png')
        plt.savefig(plot_filename)
        print(f"Metrics plot saved to {plot_filename}")
        plt.close(fig) # Close the figure to free memory

    except FileNotFoundError:
        print(f"Log file not found at {log_file}, skipping metrics plot.")
    except Exception as e:
        print(f"Could not generate metrics plot: {e}")


def plot_confusion_matrix(all_preds, all_targets, class_names, output_dir, dataset_name):
    """Plots the confusion matrix."""
    if not all_preds or not all_targets:
        print("No predictions or targets found, skipping confusion matrix.")
        return

    try:
        cm = confusion_matrix(all_targets, all_preds)
        # Adjust figure size based on number of classes
        figsize = max(8, len(class_names) // 6) # Heuristic for figure size
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')

        ax.set_title(f'{dataset_name} - Confusion Matrix')
        plt.tight_layout() # Adjust layout to prevent overlap
        cm_filename = os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png')
        plt.savefig(cm_filename)
        print(f"Confusion matrix saved to {cm_filename}")
        plt.close(fig) # Close the figure

    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

#####################################
# Pretraining Function              #
#####################################

def pretrain_on_dataset(model, train_loader, val_loader, num_classes, args, dataset_name, output_dir, log_dir):
    model = replace_head(model, num_classes)
    model.to(args.device)

    # Handle DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for pretraining on {dataset_name}")
        model = nn.DataParallel(model)

    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    # Ensure steps_per_epoch is calculated correctly
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
         print(f"Warning: train_loader for {dataset_name} is empty!")
         return model # Cannot train with empty loader

    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        base_lr=args.lr, # Pass base LR
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(args.device)

    print(f"\n=== Pretraining on {dataset_name} ({num_classes} classes) ===")
    best_acc = 0.0
    pretrain_output_dir = os.path.join(output_dir, f"pretrain_{dataset_name}")
    os.makedirs(pretrain_output_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{dataset_name}_log.csv")

    # --- Training Loop ---
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, epoch, args.epochs, args.device, dataset_name=f"Pretrain {dataset_name}", mixup_fn=None # No mixup for pretrain
        )
        val_loss, val_acc1, val_acc5, _, _ = validate( # Discard preds/targets here
             model, val_loader, criterion, args.device, dataset_name=f"Pretrain {dataset_name}"
        )

        best_acc = max(val_acc1, best_acc)
        print(f"[{dataset_name}] Epoch {epoch+1}: Val Acc@1={val_acc1:.2f}% | Best Val Acc@1={best_acc:.2f}%")

        # Log metrics
        log_metrics(dataset_name, epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, log_dir=log_dir)

        # Save checkpoint (optional, can save best only)
        # save_checkpoint(model.module if isinstance(model, nn.DataParallel) else model, optimizer, scheduler, epoch + 1, val_acc1,
        #                 os.path.join(pretrain_output_dir, f'ckpt_epoch_{epoch+1}.pth'))

    # --- Save Final/Best Pretrained Model ---
    # Retrieve the actual model from DataParallel wrapper if necessary
    final_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    save_pretrain_checkpoint(final_model_state, dataset_name, output_dir=pretrain_output_dir) # Pass state_dict directly

    # --- Generate Plots after Pretraining ---
    print(f"\n--- Generating plots for {dataset_name} pretraining ---")
    # Plot Loss/Accuracy Curves
    plot_metrics(log_file_path, pretrain_output_dir, dataset_name)

    # Generate Confusion Matrix (requires one last validation run)
    print(f"Running final validation on {dataset_name} for Confusion Matrix...")
    _, _, _, final_preds, final_targets = validate(model, val_loader, criterion, args.device, dataset_name=f"Final {dataset_name} Val")
    class_names = [str(i) for i in range(num_classes)] # Generic class names
    plot_confusion_matrix(final_preds, final_targets, class_names, pretrain_output_dir, dataset_name)

    # Return the base model (without DataParallel wrapper if it was used)
    return model.module if isinstance(model, nn.DataParallel) else model


#####################################
# Training and Evaluation Functions
#####################################

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs, device, dataset_name="Train", mixup_fn=None):
    """Train model for one epoch"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Initialize tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{dataset_name}]",
                leave=True, ncols=100, unit="batch")

    steps_per_epoch = len(train_loader) # Get steps per epoch

    for i, (images, target) in enumerate(pbar):
        # Move data to device
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Apply mixup or cutmix if available
        is_soft_target = False
        if mixup_fn is not None:
            images, target = mixup_fn(images, target)
            is_soft_target = True # Target is now soft

        # Forward pass
        output = model(images)

        # Calculate loss based on target type
        loss = criterion(output, target)

        # Measure accuracy and record loss
        # Accuracy calculation is only valid if not using soft targets (mixup/cutmix)
        batch_top1_avg = None
        batch_top5_avg = None
        if not is_soft_target:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            batch_top1_avg = top1.avg # Use running average for display
            batch_top5_avg = top5.avg

        # Update meters
        losses.update(loss.item(), images.size(0))

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (optional but often helpful)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Step the scheduler *after* the optimizer step
        # The LR scheduler expects step increments, adjust based on total steps
        current_step = epoch * steps_per_epoch + i
        scheduler.step(current_step) # Pass current step for LambdaLR-like schedulers


        # Update progress bar
        lr = optimizer.param_groups[0]['lr'] # More reliable way to get current LR
        postfix_dict = {
            'Loss': f"{losses.avg:.4f}",
            'LR': f"{lr:.6f}"
        }
        if batch_top1_avg is not None:
             postfix_dict['Acc@1'] = f"{batch_top1_avg:.2f}%"
        # if batch_top5_avg is not None:
        #     postfix_dict['Acc@5'] = f"{batch_top5_avg:.2f}%"
        pbar.set_postfix(postfix_dict)

    # Return epoch averages (handle case where accuracy wasn't measured)
    return losses.avg, top1.avg if top1.count > 0 else None, top5.avg if top5.count > 0 else None


def validate(model, val_loader, criterion, device, dataset_name="Validation"):
    """Evaluate model on validation set"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_preds = []
    all_targets = []

    # Initialize tqdm progress bar
    pbar = tqdm(val_loader, desc=f"[{dataset_name}]", leave=False, ncols=100, unit="batch")

    with torch.no_grad():
        for images, target in pbar:
            # Move data to device
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Forward pass
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Collect predictions for Confusion Matrix
            _, predicted_indices = torch.max(output.data, 1)
            all_preds.extend(predicted_indices.cpu().numpy())
            all_targets.extend(target.cpu().numpy())


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

    print(f"* [{dataset_name}]: Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}% Loss {losses.avg:.4f}")
    # Return averages AND the collected predictions/targets
    return losses.avg, top1.avg, top5.avg, all_preds, all_targets

#####################################
# Main Training Loop
#####################################

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) # Add random seed for python's random module
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) # Seed all GPUs
        # These can sometimes slow down training or cause issues, enable if needed
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True # Usually speeds up training

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.tag if args.tag else time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.tag if args.tag else os.path.basename(output_dir)) # Log dir specific to this run
    os.makedirs(log_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Using device: {args.device}")

    # Create model - start with ImageNet classes (1000) or a base number
    # The head will be replaced before each training stage.
    print("Initializing model...")
    model = get_model(model_name='swin_t', efficient=args.efficient, num_classes=1000) # Start with 1000 classes

    # === Pretraining Stage: CIFAR ===
    if args.pretrain_cifar and not args.skip_pretrain:
        print("\n--- Starting CIFAR-100 Pretraining Stage ---")
        print("Creating dataloaders for cifar100...")
        cifar_train_loader = get_cifar_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
        cifar_val_loader = get_cifar_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
        model = pretrain_on_dataset(
            model, cifar_train_loader, cifar_val_loader,
            num_classes=100, args=args, dataset_name='cifar100',
            output_dir=output_dir, log_dir=log_dir
            )
        print("--- CIFAR-100 Pretraining Stage Complete ---")


    # === Pretraining Stage: Caltech ===
    if args.pretrain_caltech and not args.skip_pretrain:
        print("\n--- Starting Caltech-256 Pretraining Stage ---")
        print("Creating dataloaders for caltech256...")
        caltech_train_loader = get_caltech_train_loader(args.batch_size, num_workers=args.workers, shuffle=True)
        caltech_val_loader = get_caltech_val_loader(args.batch_size, num_workers=args.workers, shuffle=False)
        model = pretrain_on_dataset(
             model, caltech_train_loader, caltech_val_loader,
             num_classes=257, args=args, dataset_name='caltech256',
             output_dir=output_dir, log_dir=log_dir
        )
        print("--- Caltech-256 Pretraining Stage Complete ---")

    # === Final Training Stage (Tiny ImageNet) ===
    print("\n--- Starting Final Training Stage (Tiny ImageNet) ---")
    final_num_classes = 200 # Tiny ImageNet has 200 classes
    print(f"Creating dataloaders for Tiny ImageNet ({final_num_classes} classes)...")
    train_loader, val_loader, mixup_fn = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=MODEL_CONFIG["img_size"], # Make sure this matches Swin-T expectation
        use_mixup=args.mixup
    )

    # Replace head for the final dataset
    model = replace_head(model, num_classes=final_num_classes)
    model.to(args.device)

    # Handle DataParallel if multiple GPUs are available for the final stage
    if torch.cuda.device_count() > 1 and not args.evaluate: # Don't wrap if only evaluating
        print(f"Using {torch.cuda.device_count()} GPUs for final training stage.")
        model = nn.DataParallel(model)


    # Print model information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Final Model - Number of trainable parameters: {num_params:,}")

    # Create optimizer
    optimizer = create_optimizer(model.module if isinstance(model, nn.DataParallel) else model, args.lr, args.weight_decay)


    # Create scheduler
    steps_per_epoch_final = len(train_loader)
    if steps_per_epoch_final == 0:
        print("Error: Final train loader is empty!")
        return

    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch_final,
        base_lr=args.lr, # Pass base LR
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )

    # Create loss function
    if args.mixup and mixup_fn is not None:
        print("Using Mixup/CutMix augmentation with SoftTargetCrossEntropy loss.")
        criterion = SoftTargetCrossEntropy().to(args.device)
    else:
        print(f"Using Label Smoothing Cross Entropy loss with smoothing={args.label_smoothing}.")
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(args.device)

    # Optionally resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
         # Ensure the model head matches the checkpoint's expected classes before loading
         # Note: This might require knowing the number of classes the checkpoint was saved with.
         # If resuming a final stage checkpoint, the head should already be correct (200).
         # If resuming a pretraining checkpoint, this logic needs adjustment.
         # For simplicity, assume resume is for the final stage.
         model_to_load = model.module if isinstance(model, nn.DataParallel) else model
         start_epoch, best_acc = load_checkpoint(model_to_load, optimizer, scheduler, args.resume)
         # Sync epoch for DataParallel case? Usually handled inside load_checkpoint if needed.

    # Evaluation only
    if args.evaluate:
        print("\n--- Running Evaluation Only ---")
        if not args.resume:
             print("Warning: Evaluating without loading a checkpoint (`--resume` not specified). Using initial model weights.")
        _, val_acc1, _, final_preds, final_targets = validate(model, val_loader, criterion, args.device, dataset_name="Evaluation")
        print(f"Evaluation Accuracy@1: {val_acc1:.3f}%")
        # Generate plots for evaluation run
        class_names = [str(i) for i in range(final_num_classes)] # Generic class names
        eval_output_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)
        plot_confusion_matrix(final_preds, final_targets, class_names, eval_output_dir, "tinyimagenet_eval")
        # Cannot plot loss/acc curves without training history
        return

    # Print training configuration
    print(f"\nStarting final training for {args.epochs} epochs (from epoch {start_epoch})")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial Learning rate: {args.lr}")
    print(f"Minimum Learning rate: {args.min_lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Using mixup: {args.mixup and mixup_fn is not None}")
    print(f"Label smoothing: {args.label_smoothing if not (args.mixup and mixup_fn is not None) else 'N/A (using SoftTargetCE)'}")

    # --- Final Training loop ---
    log_file_path_main = os.path.join(log_dir, "main_log.csv")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train for one epoch
        train_loss, train_acc1, train_acc5 = train_one_epoch(
             model, train_loader, criterion, optimizer, scheduler, epoch, args.epochs, args.device, dataset_name="TinyImageNet Train", mixup_fn=mixup_fn)

        # Evaluate on validation set
        val_loss, val_acc1, val_acc5, _, _ = validate( # Discard preds/targets during epoch validation
             model, val_loader, criterion, args.device, dataset_name="TinyImageNet Val")

        # Calculate epoch time
        epoch_time = time.time() - epoch_start

        # Check if current model is the best
        is_best = val_acc1 > best_acc
        if is_best:
            best_acc = val_acc1
            print(f"** New Best Val Acc@1: {best_acc:.3f}% **")

        # Log metrics to CSV
        # Handle None for train_acc if mixup was used
        log_metrics("main", epoch,
                    train_loss, train_acc1, train_acc5,
                    val_loss, val_acc1, val_acc5,
                    log_dir=log_dir)


        # Print epoch summary
        print(f"--- Epoch {epoch+1}/{args.epochs} Summary ---")
        print(f"  Time: {epoch_time:.2f}s")
        train_acc1_str = f"{train_acc1:.2f}%" if train_acc1 is not None else "N/A (Mixup)"
        train_acc5_str = f"{train_acc5:.2f}%" if train_acc5 is not None else "N/A (Mixup)"
        print(f"  Train: Loss {train_loss:.4f}, Acc@1 {train_acc1_str}, Acc@5 {train_acc5_str}")
        print(f"  Valid: Loss {val_loss:.4f}, Acc@1 {val_acc1:.2f}%, Acc@5 {val_acc5:.2f}%")
        print(f"  Best Valid Acc@1 so far: {best_acc:.2f}%")
        print("-" * (len(f"--- Epoch {epoch+1}/{args.epochs} Summary ---"))) # Divider


        # Retrieve the actual model state dict, handling DataParallel
        model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
             save_checkpoint(
                 model_state_to_save, optimizer, scheduler, epoch + 1, val_acc1,
                 os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth')
             )

        # Always save the best model based on validation accuracy
        if is_best:
             save_checkpoint(
                 model_state_to_save, optimizer, scheduler, epoch + 1, best_acc, # Save best_acc here
                 os.path.join(output_dir, 'model_best.pth')
             )

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")

    # --- Generate Final Plots after Training ---
    print(f"\n--- Generating plots for final Tiny ImageNet training ---")
    # Plot Loss/Accuracy Curves from the main log file
    plot_metrics(log_file_path_main, output_dir, "tinyimagenet_main")

    # Load the best model checkpoint for final validation and CM
    best_model_path = os.path.join(output_dir, 'model_best.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final validation...")
        # Re-create model instance and load state dict (necessary if DataParallel was used during training)
        final_model = get_model(model_name='swin_t', efficient=args.efficient, num_classes=final_num_classes)
        load_checkpoint(final_model, filename=best_model_path) # Load only model weights
        final_model.to(args.device)
        if torch.cuda.device_count() > 1: # Apply DataParallel if needed for validation
             final_model = nn.DataParallel(final_model)
    else:
        print("Best model checkpoint not found. Using model from last epoch for final validation.")
        final_model = model # Use the model from the last training epoch

    # Generate Confusion Matrix using the best (or last) model
    print("Running final validation on Tiny ImageNet for Confusion Matrix...")
    # Use the correct criterion for validation
    final_val_criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(args.device) # Use standard CE for final validation CM

    _, _, _, final_preds, final_targets = validate(
        final_model, val_loader, final_val_criterion, args.device, dataset_name="Final Best Model Val"
        )
    class_names = [str(i) for i in range(final_num_classes)] # Generic class names
    plot_confusion_matrix(final_preds, final_targets, class_names, output_dir, "tinyimagenet_main_best")

    print("--- Final Training Stage Complete ---")


def parse_args():
    parser = argparse.ArgumentParser(description='Swin Transformer Training with Pretraining and Plotting')

    # Model parameters
    parser.add_argument('--efficient', action='store_true', help='Use efficient model variant')

    # pre-training...
    parser.add_argument('--pretrain-cifar', action='store_true', help='Pretrain on CIFAR-100 first')
    parser.add_argument('--pretrain-caltech', action='store_true', help='Pretrain on Caltech-256 (after CIFAR if specified)')
    parser.add_argument('--skip-pretrain', action='store_true', help='Skip all pretraining stages')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'], metavar='N', help=f'Input batch size (default: {TRAIN_CONFIG["batch_size"]})')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'], metavar='N', help=f'Number of epochs to train (default: {TRAIN_CONFIG["epochs"]})')
    parser.add_argument('--lr', '--learning-rate', type=float, default=TRAIN_CONFIG['learning_rate'], metavar='LR', help=f'Initial learning rate (default: {TRAIN_CONFIG["learning_rate"]})')
    parser.add_argument('--min-lr', type=float, default=TRAIN_CONFIG['min_lr'], metavar='MINLR', help=f'Minimum learning rate for cosine scheduler (default: {TRAIN_CONFIG["min_lr"]})')
    parser.add_argument('--warmup-epochs', type=int, default=TRAIN_CONFIG['warmup_epochs'], metavar='N', help=f'Number of warmup epochs (default: {TRAIN_CONFIG["warmup_epochs"]})')
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG['weight_decay'], metavar='WD', help=f'Weight decay (default: {TRAIN_CONFIG["weight_decay"]})')
    parser.add_argument('--label-smoothing', type=float, default=TRAIN_CONFIG['label_smoothing'], metavar='LS', help=f'Label smoothing factor (default: {TRAIN_CONFIG["label_smoothing"]})')
    parser.add_argument('--device', default=TRAIN_CONFIG['device'], help='Device to use (cuda or cpu)')
    parser.add_argument('--mixup', action='store_true', default=False, help='Use mixup and cutmix augmentation (requires timm mixup implementation in get_loaders)')

    # Data loading
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='Number of data loading workers (default: 4)')

    # Checkpointing
    parser.add_argument('--output-dir', default=TRAIN_CONFIG['output_dir'], help='Path to save output (checkpoints, plots)')
    parser.add_argument('--tag', default='', help='Experiment tag to append to output/log directories')
    parser.add_argument('--save-interval', type=int, default=TRAIN_CONFIG['save_interval'], metavar='N', help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume final training stage from checkpoint path')

    # logs
    parser.add_argument('--log-dir', default='logs', type=str, help='Directory to save all training logs (.csv files)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='Random seed (default: 42)')
    parser.add_argument('--evaluate', action='store_true', help='Perform evaluation only (requires --resume)')
    # parser.add_argument('--no-plots', action='store_true', help='Disable generating plots') # Optional: Add if you want to disable plots

    args = parser.parse_args()

    # Set device explicitly
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    return args

if __name__ == '__main__':
    # Need 'random' for seeding
    import random
    args = parse_args()
    main(args)
