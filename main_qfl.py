import argparse
import torch
import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
import medmnist
from medmnist import INFO

# Importamos el núcleo lógico
from qfl_core import SlimmableVQC, QFLClient, hybrid_aggregation

# ==============================================================================
# CONFIGURACIÓN GLOBAL DE VISUALIZACIÓN
# ==============================================================================
# Pon esto en False si NO quieres generar la imagen de ejemplos (samples_*.png)
VISUALIZE_SAMPLES = True 

# ==============================================================================
# MOTOR DE VISUALIZACIÓN Y TRADUCCIÓN (EN/PT)
# ==============================================================================
class Visualizer:
    def __init__(self, language='en', font_size=12, output_dir='./resultados'):
        self.language = language
        self.output_dir = os.path.join(output_dir, 'images')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuración Global de Matplotlib
        plt.rcParams.update({'font.size': font_size})
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        
        # Diccionario de Traducciones (PT / EN)
        self.texts = {
            'en': {
                'dist_title': 'Data Distribution per Client (Non-IID)',
                'xlabel_client': 'Client ID',
                'ylabel_count': 'Number of Samples',
                'legend_class': 'Class Label',
                'acc_title': 'Global Model Accuracy Evolution',
                'xlabel_round': 'Communication Round',
                'ylabel_acc': 'Test Accuracy',
                'sample_title': 'Visual Challenge: Class Ambiguity',
                'sample_subtitle': 'Samples used in QFL-Adaptive validation'
            },
            'pt': {
                'dist_title': 'Distribuição de Dados por Cliente (Non-IID)',
                'xlabel_client': 'ID do Cliente',
                'ylabel_count': 'Quantidade de Amostras',
                'legend_class': 'Classe / Rótulo',
                'acc_title': 'Evolução da Acurácia Global',
                'xlabel_round': 'Rodada de Comunicação',
                'ylabel_acc': 'Acurácia de Teste',
                'sample_title': 'Desafio Visual: Ambiguidade entre Classes',
                'sample_subtitle': 'Exemplos usados na validação do QFL-Adaptive'
            }
        }
    
    def get_text(self, key):
        """Retorna el texto traducido según self.language"""
        return self.texts.get(self.language, self.texts['en']).get(key, key)

    def plot_confusing_samples(self, dataset, class_names, dataset_name):
        """
        Dibuja ejemplos de las dos clases para mostrar la dificultad visual.
        Solo se ejecuta si VISUALIZE_SAMPLES = True.
        """
        print(f"   [Plotting] Generando comparativa visual (Challenge) para {dataset_name}...")
        
        # Usamos un loader temporal para sacar imágenes
        loader = DataLoader(dataset, batch_size=100, shuffle=True)
        data_iter = iter(loader)
        images, labels = next(data_iter)
        
        # Filtramos ejemplos de clase 0 y clase 1
        imgs_c0 = images[labels == 0][:5]
        imgs_c1 = images[labels == 1][:5]
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        fig.suptitle(f"{self.get_text('sample_title')} ({dataset_name.upper()})", fontsize=14)
        
        # Fila 1: Clase 0
        for i in range(min(5, len(imgs_c0))):
            ax = axes[0, i]
            img = imgs_c0[i].squeeze().numpy()
            img = img * 0.5 + 0.5 # Des-normalizar
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 2: ax.set_title(f"Class 0: {class_names[0]}", fontsize=11, pad=5)

        # Fila 2: Clase 1
        for i in range(min(5, len(imgs_c1))):
            ax = axes[1, i]
            img = imgs_c1[i].squeeze().numpy()
            img = img * 0.5 + 0.5 # Des-normalizar
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 2: ax.set_title(f"Class 1: {class_names[1]}", fontsize=11, pad=5)

        plt.tight_layout()
        filename = f"{self.output_dir}/samples_{dataset_name}_{self.language}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   [Gráfico Guardado]: {filename}")

    def plot_data_distribution(self, clients, num_classes, dataset_name, class_names):
        """Gráfico de barras apiladas de la distribución de datos."""
        client_ids = []
        data_counts = {i: [] for i in range(num_classes)}
        
        print(f"   [Plotting] Generando distribución de datos para {dataset_name}...")
        
        for client in clients:
            client_ids.append(f"C{client.client_id}")
            
            # --- ITERACIÓN SEGURA (Fix Subset Error) ---
            client_labels = []
            for _, targets in client.loader:
                if targets.dim() > 1: targets = targets.squeeze()
                client_labels.extend(targets.tolist())
            
            for cls_idx in range(num_classes):
                count = client_labels.count(cls_idx)
                data_counts[cls_idx].append(count)

        df_dist = pd.DataFrame(data_counts, index=client_ids)
        # Asignamos nombres reales a las columnas para la leyenda
        if len(class_names) == num_classes:
            df_dist.columns = class_names
        
        # Plot
        ax = df_dist.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.9)
        plt.title(f"{self.get_text('dist_title')} - {dataset_name}")
        plt.xlabel(self.get_text('xlabel_client'))
        plt.ylabel(self.get_text('ylabel_count'))
        
        # Ajuste de leyenda
        plt.legend(title=self.get_text('legend_class'), bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"{self.output_dir}/dist_{dataset_name}_{self.language}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"   [Gráfico Guardado]: {filename}")

    def plot_metrics_history(self, history_df, filename_suffix):
        """Curva de aprendizaje."""
        rounds = history_df['round']
        plt.figure()
        plt.plot(rounds, history_df['accuracy'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
        plt.title(self.get_text('acc_title'))
        plt.xlabel(self.get_text('xlabel_round'))
        plt.ylabel(self.get_text('ylabel_acc'))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/acc_{filename_suffix}.png", dpi=300)
        plt.close()

# ==============================================================================
# CARGA DE DATASETS (MODO DIFÍCIL / HARD MODE)
# ==============================================================================
def get_dataset(name):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    class_names = ["0", "1"] # Default

    if name == 'mnist':
        ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # HARD MODE: 3 vs 5 (Curvas similares)
        idx = (ds.targets == 3) | (ds.targets == 5)
        ds.targets = ds.targets[idx]
        ds.data = ds.data[idx]
        ds.targets[ds.targets == 3] = 0
        ds.targets[ds.targets == 5] = 1
        class_names = ["Dígito 3", "Dígito 5"]
        return ds, 2, class_names

    elif name == 'fashion':
        ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        # HARD MODE: Pullover (2) vs Coat (4)
        idx = (ds.targets == 2) | (ds.targets == 4)
        ds.targets = ds.targets[idx]
        ds.data = ds.data[idx]
        ds.targets[ds.targets == 2] = 0
        ds.targets[ds.targets == 4] = 1
        class_names = ["Pullover", "Coat"]
        return ds, 2, class_names

    elif name == 'pneumonia':
        info = INFO['pneumoniamnist']
        DataClass = getattr(medmnist, info['python_class'])
        ds = DataClass(split='train', transform=transform, download=True)
        class_names = ["Normal", "Pneumonia"]
        return ds, 2, class_names

    elif name == 'breast':
        info = INFO['breastmnist']
        DataClass = getattr(medmnist, info['python_class'])
        ds = DataClass(split='train', transform=transform, download=True)
        class_names = ["Maligno", "Benigno"]
        return ds, 2, class_names

    else:
        raise ValueError(f"Dataset {name} no soportado.")

# ==============================================================================
# MAIN EJECUTABLE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--channel_threshold', type=float, default=0.4)
    parser.add_argument('--bad_channel_prob', type=float, default=0.3)
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--clients', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./resultados')
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--font_size', type=int, default=14)
    args = parser.parse_args()

    SEED = 43
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    viz = Visualizer(language=args.language, font_size=args.font_size, output_dir=args.output_dir)
    print(f"\n--- STARTING QFL-ADAPTIVE: {args.dataset.upper()} (Lang: {args.language}) ---")

    # 1. Preparar Datos
    full_ds, num_classes, class_names = get_dataset(args.dataset)
    
    train_len = int(0.8 * len(full_ds))
    test_len = len(full_ds) - train_len
    train_ds, test_ds = random_split(full_ds, [train_len, test_len])
    test_loader = DataLoader(test_ds, batch_size=32)
    
    subset_size = len(train_ds) // args.clients
    client_subsets = [Subset(train_ds, range(i*subset_size, (i+1)*subset_size)) for i in range(args.clients)]

    # 2. Inicializar
    global_model = SlimmableVQC(output_dim=num_classes)
    clients = [QFLClient(i, client_subsets[i]) for i in range(args.clients)]
    
    # >> GRÁFICO 1: Muestras Visuales (Controlado por Variable Global)
    if VISUALIZE_SAMPLES:
        viz.plot_confusing_samples(full_ds, class_names, args.dataset)
    
    # >> GRÁFICO 2: Distribución de Datos (Seguro)
    viz.plot_data_distribution(clients, num_classes, args.dataset, class_names)

    # 3. Bucle Federado
    results = []
    cumulative_bytes = 0

    for r in range(args.rounds):
        global_w = global_model.state_dict()
        updates = []
        round_sent = 0
        round_full_ref = 0
        
        for client in clients:
            client.update_and_personalize(global_w)
            client.train()
            params, p_type, sent, full = client.adaptive_upload(args.bad_channel_prob, args.channel_threshold)
            updates.append((params, p_type))
            round_sent += sent
            round_full_ref += full

        global_model = hybrid_aggregation(global_model, updates)
        
        global_model.eval()
        correct=0; total=0
        with torch.no_grad():
            for d, t in test_loader:
                d=d.to('cpu').float(); t=t.to('cpu').long()
                if t.dim()>1: t=t.squeeze()
                out = global_model(d)
                correct += (out.argmax(1)==t).sum().item()
                total += t.size(0)
        
        acc = correct/total if total>0 else 0
        cumulative_bytes += round_sent
        savings = (1 - (round_sent/round_full_ref))*100 if round_full_ref>0 else 0
        
        print(f"Round {r+1:02d} | Acc: {acc:.4f} | Savings: {savings:.1f}%")
        
        results.append({
            'dataset': args.dataset,
            'round': r+1,
            'accuracy': acc,
            'savings_pct': savings,
            'threshold': args.channel_threshold,
            'bad_channel_prob': args.bad_channel_prob
        })

    # Guardar
    df_res = pd.DataFrame(results)
    suffix = f"{args.dataset}_th{args.channel_threshold}_prob{args.bad_channel_prob}"
    csv_path = os.path.join(args.output_dir, f"metrics_{suffix}.csv")
    df_res.to_csv(csv_path, index=False)
    viz.plot_metrics_history(df_res, suffix)
    print(f"--- Fin. Resultados en: {args.output_dir} ---")

if __name__ == '__main__':
    main()
