# ==========================================
# Step 1: Environment Setup, Data Preprocessing, and Dataset Class
# ==========================================
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ------------------------------------------
# [Device Check]
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("="*60)
# # print(f"System initialization complete! Current computing device: {device}")
# if device.type == 'cuda':
#     print(f"Successfully detected GPU: {torch.cuda.get_device_name(0)}")
# print("="*60)

# print(f"PyTorch Version: {torch.__version__}")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Current CUDA Version: {torch.version.cuda}")
# print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ------------------------------------------
# [Dataset Definition]
# ------------------------------------------
class CelebADataset(Dataset):
    def __init__(self, df, img_dir, target_col, bias_col, transform=None):
        """
        df: DataFrame containing dataset metadata
        img_dir: Physical directory path where images are stored
        target_col: Target attribute column name (e.g., 'Smiling')
        bias_col: Bias/Sensitive attribute column name (e.g., 'High_Cheekbones')
        """
        # Reset index to ensure iteration doesn't crash due to discontinuous indices
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_col = target_col
        self.bias_col = bias_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        img_name = self.df.loc[idx, 'image_id']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read image and ensure it is in RGB format
        image = Image.open(img_path).convert('RGB')
        
        # Apply data augmentation and normalization transforms
        if self.transform:
            image = self.transform(image)
            
        # Dynamically fetch labels and convert to PyTorch Tensors
        target = torch.tensor(self.df.loc[idx, self.target_col], dtype=torch.long)
        bias = torch.tensor(self.df.loc[idx, self.bias_col], dtype=torch.long)
        
        return image, target, bias

# ------------------------------------------
# [Transforms Definition]
# ------------------------------------------
# Training set: Add horizontal flip for basic data augmentation to increase model generalizability
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test set: Only apply resizing and normalization, no random augmentation
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# Step 2: Data Preparation and DataLoader Generation (Dynamic)
# ==========================================
def prepare_dataloaders(target_col, bias_col, batch_size=128, num_workers=4):
    """
    Download data, dynamically clean labels, and prepare 3 types of training DataLoaders 
    (Baseline, Oversampled, Undersampled) as well as Val/Test Loaders.
    """
    print("\n" + "-"*60)
    print(f"Data Download and Loading | Target: {target_col}, Bias: {bias_col}")
    print("-"*60)
    
    # 1. Download data using kagglehub (skips automatically if cached locally)
    dataset_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    img_dir = os.path.join(dataset_path, "img_align_celeba", "img_align_celeba")
    attr_path = os.path.join(dataset_path, "list_attr_celeba.csv")
    partition_path = os.path.join(dataset_path, "list_eval_partition.csv")
    
    # 2. Read and merge CSV files
    df_attr = pd.read_csv(attr_path)
    df_partition = pd.read_csv(partition_path)
    df = pd.merge(df_attr, df_partition, on='image_id')
    
    # ==========================================
    #  [Ultimate Failsafe] Prevent CUDA Errors
    # ==========================================
    print(f"Cleaning data: Converting {target_col} and {bias_col} to strict 0 and 1...")
    df[target_col] = (df[target_col] == 1).astype(int)
    df[bias_col] = (df[bias_col] == 1).astype(int)
    
    assert df[target_col].isin([0, 1]).all(), f"Fatal Error: {target_col} labels were not fully converted!"
    assert df[bias_col].isin([0, 1]).all(), f"Fatal Error: {bias_col} labels were not fully converted!"
    
    # 3. Split Dataset
    df_train = df[df['partition'] == 0].reset_index(drop=True)
    df_val   = df[df['partition'] == 1].reset_index(drop=True)
    df_test  = df[df['partition'] == 2].reset_index(drop=True)
    print(f"Data split complete ➔ Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    # ==========================================
    # [Group Balanced Strategy Preparation]
    # ==========================================
    # Dynamically calculate 4 intersectional sub-groups 
    # (00: Y=0, A=0 | 01: Y=0, A=1 | 10: Y=1, A=0 | 11: Y=1, A=1)
    df_train['group'] = df_train[target_col] * 2 + df_train[bias_col]
    group_counts = df_train['group'].value_counts().sort_index()
    print(f"👥 Training set group distribution (00, 01, 10, 11): {group_counts.to_dict()}")
    
    # (A) Oversampling Preparation: Calculate weights and build Sampler
    sample_weights = np.array([1.0 / group_counts[g] for g in df_train['group']])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # (B) Undersampling Preparation: Downsample to the smallest group size
    min_count = group_counts.min()
    print(f"🔪 Executing Undersampling ➔ Reducing all group sizes to the smallest group count ({min_count} samples)")
    df_train_under = df_train.groupby('group').sample(n=min_count, random_state=42).reset_index(drop=True)
    
    # ==========================================
    # [Build all Datasets and DataLoaders]
    # ==========================================
    print("\n" + "-"*60)
    print(f"Building DataLoaders (num_workers={num_workers})")
    print("-"*60)
    
    # Instantiate Datasets (passing dynamic target_col and bias_col)
    train_dataset = CelebADataset(df_train, img_dir, target_col, bias_col, transform=train_transform)
    train_dataset_under = CelebADataset(df_train_under, img_dir, target_col, bias_col, transform=train_transform)
    val_dataset = CelebADataset(df_val, img_dir, target_col, bias_col, transform=eval_transform)
    test_dataset = CelebADataset(df_test, img_dir, target_col, bias_col, transform=eval_transform)
    
    # Instantiate DataLoaders (Baseline, Oversampled, Undersampled only)
    loaders = {}
    
    # 1. Baseline Loader (Random Sampling)
    loaders['baseline_train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    # 2. Oversampled Loader (Uses sampler, shuffle must be False)
    loaders['oversampled_train'] = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)
    
    # 3. Undersampled Loader (Reduced data size, shuffled randomly)
    loaders['undersampled_train'] = DataLoader(train_dataset_under, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    # 4. Val & Test Loaders
    loaders['val_loader'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loaders['test_loader'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print("3 Training DataLoaders and Val/Test DataLoaders are ready!")
    
    return loaders, df_train

# ==========================================
# Step 3: Model Architecture and Custom Loss Functions
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# ------------------------------------------
# [Model Architecture] Standard ResNet18
# ------------------------------------------
class StandardResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(StandardResNet, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ------------------------------------------
# [Loss 1] Group-Weighted Cross Entropy Loss (Reweighting)
# ------------------------------------------
class GroupWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, group_weights, device):
        super(GroupWeightedCrossEntropyLoss, self).__init__()
        self.group_weights = group_weights.to(device)

    def forward(self, logits, targets, biases):
        groups = targets * 2 + biases
        weights = self.group_weights[groups]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        return (ce_loss * weights).mean()

def get_reweighted_loss(df_train, device):
    """Calculate inverse proportional weights based on the 'group' column of the training set."""
    group_counts = df_train['group'].value_counts().sort_index().values
    raw_weights = 1.0 / group_counts
    normalized_weights = raw_weights / raw_weights.mean()
    weight_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
    return GroupWeightedCrossEntropyLoss(weight_tensor, device)

# ------------------------------------------
# [Loss 2] Group-Weighted Focal Loss
# ------------------------------------------
class GroupWeightedFocalLoss(nn.Module):
    def __init__(self, group_weights, device, gamma=2.0):
        super(GroupWeightedFocalLoss, self).__init__()
        self.group_weights = group_weights.to(device)
        self.gamma = gamma

    def forward(self, logits, targets, biases):
        groups = targets * 2 + biases
        weights = self.group_weights[groups]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = weights * ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

def get_focal_loss(df_train, device):
    """Calculate inverse proportional weights based on the 'group' column of the training set and incorporate the Focal mechanism."""
    group_counts = df_train['group'].value_counts().sort_index().values
    raw_weights = 1.0 / group_counts
    normalized_weights = raw_weights / raw_weights.mean()
    weight_tensor = torch.tensor(normalized_weights, dtype=torch.float32)
    return GroupWeightedFocalLoss(weight_tensor, device, gamma=2.0)

# ==========================================
# Step 4: Training Loop and Evaluation Function
# ==========================================
import time
import os
from tqdm import tqdm

def train_debias_model(model, train_loader, val_loader, criterion, optimizer, device, 
                       num_epochs=5, task_name='Task', method_name='Baseline', loss_type='CE'):
    print(f"\n🚀 [Start Training] Task: {task_name} | Method: {method_name}")
    best_wg_acc = 0.0
    save_path_wg = f'best_wg_model_{task_name}_{method_name}.pth'

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs} [Train]", leave=False)
        for images, targets, biases in train_bar:
            images, targets, biases = images.to(device), targets.to(device), biases.to(device)
            optimizer.zero_grad()
            logits = model(images)
            
            # Check if biases need to be passed to calculate custom Loss
            if loss_type in ['Reweighting', 'Focal']:
                loss = criterion(logits, targets, biases)
            else:
                loss = criterion(logits, targets)
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        # --- Validation Phase ---
        model.eval()
        all_preds, all_targets, all_biases = [], [], []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for images, targets, biases in val_bar:
                images, targets, biases = images.to(device), targets.to(device), biases.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_biases.append(biases.cpu())
                
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_biases = torch.cat(all_biases)
        
        # Calculate metrics for the four intersectional groups
        epoch_overall_acc = (all_preds == all_targets).float().mean().item()
        group_accs = {}
        for t in [0, 1]:
            for b in [0, 1]:
                mask = (all_targets == t) & (all_biases == b)
                group_accs[f'{t}{b}'] = (all_preds[mask] == all_targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0
                
        epoch_worst_acc = min(group_accs.values())
        print(f"Epoch {epoch+1:02d} | WG Acc: {epoch_worst_acc:.4f} | Overall: {epoch_overall_acc:.4f}")
        
        if epoch_worst_acc > best_wg_acc:
            best_wg_acc = epoch_worst_acc
            torch.save(model.state_dict(), save_path_wg)
            
    print(f"[{method_name}] Training Complete! Best WG Acc: {best_wg_acc:.4f}\n")
    return save_path_wg

def evaluate_model(model, test_loader, device, model_path):
    if not os.path.exists(model_path): return None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    all_preds, all_targets, all_biases = [], [], []
    
    with torch.no_grad():
        for images, targets, biases in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_biases.append(biases.cpu())
            
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_biases = torch.cat(all_biases)
    
    overall_acc = (all_preds == all_targets).float().mean().item()
    group_accs = {}
    for t in [0, 1]:
        for b in [0, 1]:
            mask = (all_targets == t) & (all_biases == b)
            group_accs[f'{t}{b}'] = (all_preds[mask] == all_targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0
            
    worst_group_acc = min(group_accs.values())
    return {
        'overall_acc': overall_acc,
        'worst_group_acc': worst_group_acc,
        'acc_00': group_accs['00'], 'acc_01': group_accs['01'],
        'acc_10': group_accs['10'], 'acc_11': group_accs['11']
    }

# ==========================================
# Step 5: [Full Pipeline] Training + Bootstrapped CI + Gap Bracket Plotting
# ==========================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    BATCH_SIZE = 128
    NUM_WORKERS = 4  
    EPOCHS = 5           # Number of training epochs (Increase for better performance)
    LR = 1e-4            # Learning rate
    N_BOOTSTRAPS = 10    # Number of bootstrap samples for confidence intervals
    
    # Train and evaluate only Task 1 and Task 3
    tasks = [
        {'id': 1, 'name': 'Task 1 (Smiling)', 'target': 'Smiling', 'bias': 'High_Cheekbones'},
        {'id': 3, 'name': 'Task 3 (Mouth Open)', 'target': 'Smiling', 'bias': 'Mouth_Slightly_Open'}
    ]
    
    # De-biasing methods (excluding Focal Loss)
    experiments = ['Baseline', 'Oversampling', 'Undersampling', 'Reweighting']
    
    # Labels for plotting
    label_maps = {
        'Task 1 (Smiling)': ['No Smiling\nLow Cheekbones', 'No Smiling\nHigh Cheekbones', 'Smiling\nLow Cheekbones', 'Smiling\nHigh Cheekbones'],
        'Task 3 (Mouth Open)': ['No Smiling\nMouth Closed', 'No Smiling\nMouth Open', 'Smiling\nMouth Closed', 'Smiling\nMouth Open']
    }

    print("\n" + "="*60)
    print(f"⚡ Starting [Full Pipeline]: Training models + Bootstrapping (N={N_BOOTSTRAPS}) + Plotting")
    print("="*60)

    # Helper function to find saved model weights
    def find_weight_file(task_id, method):
        for f in os.listdir('.'):
            if f.startswith(f"best_wg_model_Task_{task_id}") and method in f and f.endswith(".pth"):
                return f
        return None

    # Helper function for evaluation with Bootstrapping
    def evaluate_model_bootstrap(model, test_loader, device, model_path, n_bootstraps):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        all_preds, all_targets, all_biases = [], [], []
        
        # Perform one full inference pass to save time
        with torch.no_grad():
            for images, targets, biases in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_biases.append(biases.cpu())
                
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_biases = torch.cat(all_biases)
        
        n_samples = len(all_targets)
        bootstrap_results = []
        
        # Perform N random sampling with replacement
        for b in range(n_bootstraps):
            indices = torch.randint(0, n_samples, (n_samples,))
            
            b_preds = all_preds[indices]
            b_targets = all_targets[indices]
            b_biases = all_biases[indices]
            
            overall_acc = (b_preds == b_targets).float().mean().item()
            group_accs = {}
            for t in [0, 1]:
                for bias_val in [0, 1]:
                    mask = (b_targets == t) & (b_biases == bias_val)
                    group_accs[f'{t}{bias_val}'] = (b_preds[mask] == b_targets[mask]).float().mean().item() if mask.sum() > 0 else 0.0
            
            bootstrap_results.append({
                'overall_acc': overall_acc,
                'acc_00': group_accs['00'],
                'acc_01': group_accs['01'],
                'acc_10': group_accs['10'],
                'acc_11': group_accs['11']
            })
            
        return bootstrap_results

    # ---------------------------------------------------------
    # MAIN PIPELINE: Training -> Evaluation -> Plotting
    # ---------------------------------------------------------
    for task in tasks:
        print(f"\nProcessing {task['name']}...")
        
        # 1. Prepare DataLoaders
        loaders, df_train = prepare_dataloaders(task['target'], task['bias'], BATCH_SIZE, NUM_WORKERS)
        labels = label_maps[task['name']]
        task_metrics = [] 
        
        # 2. Train Models for each experiment
        for method_name in experiments:
            print(f"\n--- Training {method_name} for {task['name']} ---")
            
            # Initialize model and optimizer
            model = StandardResNet(num_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            
            # Configure loader and loss function based on the method
            if method_name == 'Baseline':
                train_loader = loaders['baseline_train']
                criterion = nn.CrossEntropyLoss().to(device)
                loss_type = 'CE'
            elif method_name == 'Oversampling':
                train_loader = loaders['oversampled_train']
                criterion = nn.CrossEntropyLoss().to(device)
                loss_type = 'CE'
            elif method_name == 'Undersampling':
                train_loader = loaders['undersampled_train']
                criterion = nn.CrossEntropyLoss().to(device)
                loss_type = 'CE'
            elif method_name == 'Reweighting':
                train_loader = loaders['baseline_train']
                criterion = get_reweighted_loss(df_train, device)
                loss_type = 'Reweighting'
                
            # Execute training loop
            train_debias_model(
                model=model, 
                train_loader=train_loader, 
                val_loader=loaders['val_loader'], 
                criterion=criterion, 
                optimizer=optimizer, 
                device=device,
                num_epochs=EPOCHS, 
                task_name=task['name'].replace(" ", "_"), 
                method_name=method_name, 
                loss_type=loss_type
            )
        
        # 3. Evaluate and collect Bootstrap data
        print(f"\nEvaluating {task['name']}...")
        for method_name in experiments:
            model_path = find_weight_file(task['id'], method_name)
            
            if model_path:
                print(f"Found weights: {model_path} (Bootstrapping {N_BOOTSTRAPS} times...)")
                model = StandardResNet(num_classes=2).to(device)
                metrics_list = evaluate_model_bootstrap(model, loaders['test_loader'], device, model_path, N_BOOTSTRAPS)
                
                for metrics in metrics_list:
                    row = {
                        'Method': method_name,
                        'Overall Accuracy': metrics['overall_acc'],
                        labels[0]: metrics['acc_00'],
                        labels[1]: metrics['acc_01'],
                        labels[2]: metrics['acc_10'],
                        labels[3]: metrics['acc_11']
                    }
                    task_metrics.append(row)
            else:
                print(f"Missing weights for: Task {task['id']} - {method_name}")
        
        if not task_metrics:
            continue

        # 4. Plotting (CI Error Bars + Gap Brackets)
        df_results = pd.DataFrame(task_metrics)
        
        df_melted = df_results.melt(
            id_vars=['Method'],
            value_vars=['Overall Accuracy', labels[0], labels[1], labels[2], labels[3]],
            var_name='Group', value_name='Accuracy'
        )
        
        plt.figure(figsize=(14, 8))
        sns.set_theme(style="whitegrid")
        
        # Draw bar plots with 95% Confidence Intervals
        ax = sns.barplot(
            data=df_melted, x='Method', y='Accuracy', hue='Group',
            palette=['#34495e', '#3498db', '#2ecc71', '#e74c3c', '#f1c40f'], 
            edgecolor='black', errorbar='ci', capsize=0.05, err_kws={'linewidth': 1.5}
        )
        
        plt.title(f"{task['name']} - Accuracy with 95% Confidence Interval & Gap", fontsize=18, pad=20)
        plt.xlabel('De-biasing Method', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        
        # Increase Y-axis limit to create space for top brackets
        plt.ylim(0, 1.25) 
        
        # Draw Baseline Worst-Group horizontal dashed line (average of 10 samples)
        if 'Baseline' in df_results['Method'].values:
            baseline_data = df_results[df_results['Method'] == 'Baseline']
            means = [baseline_data[lbl].mean() for lbl in labels]
            worst_baseline_acc = min(means)
            plt.axhline(y=worst_baseline_acc, color='r', linestyle='--', alpha=0.6, label='Baseline Worst Group')
            
        # Draw Gap brackets for each method
        available_methods = [m for m in experiments if m in df_results['Method'].unique()]
        
        for m_idx, method in enumerate(available_methods):
            method_data = df_results[df_results['Method'] == method]
            accs = [method_data[lbl].mean() for lbl in labels]
            
            max_idx = np.argmax(accs)
            min_idx = np.argmin(accs)

            try:
                max_bar = ax.containers[max_idx + 1][m_idx]
                min_bar = ax.containers[min_idx + 1][m_idx]

                x_max = max_bar.get_x() + max_bar.get_width() / 2
                y_max = max_bar.get_height()

                x_min = min_bar.get_x() + min_bar.get_width() / 2
                y_min = min_bar.get_height()

                # Determine bracket height (Highest bar + safe distance to avoid error bars)
                group_max_y = max(accs)
                line_y = group_max_y + 0.12
                
                # 1. Top horizontal line
                ax.plot([x_min, x_max], [line_y, line_y], color='black', lw=1.2)
                
                # 2. Vertical arrows pointing to max/min bars
                ax.annotate('', xy=(x_max, y_max + 0.05), xytext=(x_max, line_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
                ax.annotate('', xy=(x_min, y_min + 0.05), xytext=(x_min, line_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

                # 3. Gap percentage text box in the middle of the line
                mid_x = (x_max + x_min) / 2
                gap_pct = abs(y_max - y_min) * 100
                
                ax.text(mid_x, line_y, f'gap\n{gap_pct:.1f}pp', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5dc', edgecolor='gray', alpha=0.9))
                
                # 4. Exact accuracy percentages above the bars
                text_bg = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7)
                ax.text(x_max, y_max + 0.015, f'{y_max*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=text_bg)
                ax.text(x_min, y_min + 0.015, f'{y_min*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=text_bg)
                
            except Exception as e:
                print(f"Error drawing Gap annotation: {e}")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_filename = f"Plot_CI_Gap_{task['name'].replace(' ', '_')}.png"
        plt.savefig(save_filename, dpi=300)
        print(f"Successfully saved plot: {save_filename}")

    print("\nPipeline Complete! Models trained, evaluated, and plotted successfully.")
    plt.show()