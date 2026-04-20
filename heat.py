import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_all_layers(file_path, layers=None):
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return
    
    data = torch.load(file_path, map_location='cpu')
    if layers is None:
        layers = [f"layer_{i}" for i in range(12)] # Ajuste si tes clés commencent à layer_0 ou layer_1

    # On crée une grille (ex: 3 lignes x 4 colonnes pour 12 couches)
    n_layers = len(layers)
    ncols = 4
    nrows = (n_layers + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(layers):
        if layer_name not in data:
            continue
        
        matrix = data[layer_name]
        ax = axes[idx]

        # --- AMÉLIORATION DE LA LISIBILITÉ ---
        # 1. Clipping des valeurs extrêmes (percentile 98) pour éviter les "sinks" éblouissants
        vmax = np.percentile(matrix.numpy(), 98)
        
        # 2. Heatmap avec réglages robustes
        sns.heatmap(matrix, 
                    ax=ax, 
                    cmap='magma', 
                    cbar=False, 
                    vmax=vmax,
                    xticklabels=False, 
                    yticklabels=False)

        # 3. Lignes de séparation Vision (32) / Audio (16)
        ax.axhline(y=32, color='cyan', linestyle='-', linewidth=1.5, alpha=0.8)
        ax.axvline(x=32, color='cyan', linestyle='-', linewidth=1.5, alpha=0.8)

        # 4. Annotation des quadrants (seulement sur la première et dernière couche pour ne pas surcharger)
        if idx == 0 or idx == n_layers - 1:
            ax.set_xlabel("Keys (Vis | Aud)", fontsize=10)
            ax.set_ylabel("Queries (Vis | Aud)", fontsize=10)
        
        ax.set_title(f"Fusion {layer_name}", fontsize=12, fontweight='bold')

    # Supprimer les axes vides si n_layers < nrows * ncols
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Évolution de l'attention multimodale\n(Fichier: {os.path.basename(file_path)})", 
                 fontsize=20, y=1.02)
    
    plt.tight_layout()
    output_name = "evolution_attention_multimodale.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"✅ Grille de heatmaps sauvegardée sous : {output_name}")
    plt.show()

# Utilisation
layers_to_plot = [f"layer_{i}" for i in range(1, 12)] # Ton code semble utiliser layer_1 à layer_11
plot_all_layers("attentions_rank_0.pt", layers=layers_to_plot)



import torch
import os

def analyze_existing_attentions(file_path):
    if not os.path.exists(file_path):
        print(f"Fichier {file_path} introuvable.")
        return
    
    # Charger les données (dictionnaire de matrices 48x48)
    data = torch.load(file_path, map_location='cpu')
    
    print(f"Analyse du fichier : {file_path}")
    print(f"{'Couche':<10} | {'Vis->Vis':<10} | {'Aud->Aud':<10} | {'Aud->Vis (Fusion)':<15}")
    print("-" * 60)

    results = {}

    # On trie les couches pour l'affichage (layer_1, layer_2...)
    sorted_layers = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))

    for layer_name in sorted_layers:
        matrix = data[layer_name] # Tenseur [48, 48]
        
        # Découpage des secteurs
        # Rappel : Indices [0:32] = Vision, [32:48] = Audio
        vis_vis = matrix[:32, :32].mean().item()
        aud_aud = matrix[32:, 32:].mean().item()
        aud_vis = matrix[32:, :32].mean().item() # L'audio qui regarde la vision
        vis_aud = matrix[:32, 32:].mean().item() # La vision qui regarde l'audio

        print(f"{layer_name:<10} | {vis_vis:.4f}   | {aud_aud:.4f}   | {aud_vis:.4f}")
        
        results[layer_name] = {
            "vis_vis": vis_vis,
            "aud_aud": aud_aud,
            "aud_vis": aud_vis,
            "vis_aud": vis_aud
        }
    
    return results

# Lancer l'analyse sur ton fichier rank_0
stats = analyze_existing_attentions("attentions_rank_0.pt")