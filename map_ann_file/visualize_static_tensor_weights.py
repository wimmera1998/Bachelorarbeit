import torch
import matplotlib.pyplot as plt
import os

# Pfad zur Checkpoint-Datei
checkpoint_path = "/disk/vanishing_data/qo691/MapTR/work_dirs/maptr_tiny_r50_24e_cwl_fusion_b4_w4_a8_XU_PreTra_Re_Sk_bcwl_1/epoch_24.pth"

# Checkpoint laden
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint erfolgreich geladen.")
except FileNotFoundError:
    print(f"Checkpoint-Datei nicht gefunden unter: {checkpoint_path}")
    exit()

# Extrahiere die Modellparameter (meist unter 'state_dict' gespeichert)
state_dict = checkpoint.get('state_dict', checkpoint)

# Gewichtungsparameter für `fusion_weight_tensor` unter dem gefundenen Schlüssel
weight_key = "pts_bbox_head.transformer.fuser.fusion_weight_tensor"
if weight_key in state_dict:
    weights = state_dict[weight_key]

    print("\nExtrahierte Gewichte für 'fusion_weight_tensor':")
    print(f"Form der Gewichte: {weights.shape}\n")

    # Durchschnitt über die Channels berechnen
    fusion_weight_avg = weights.mean(dim=0)  # Durchschnitt über Channels
    print("\nFusion Weight Map (Durchschnitt über Channels):")
    print(fusion_weight_avg)

    # Bildname basierend auf dem Checkpoint-Pfad erstellen
    directory_name = os.path.basename(os.path.dirname(checkpoint_path))  # Ordnername
    checkpoint_file = os.path.basename(checkpoint_path).replace('.pth', '')  # Datei ohne .pth
    output_image_name = f"{directory_name}_{checkpoint_file}_heatmap.png"

    # Visualisierung als Heatmap
    print("Erstelle Heatmap der Durchschnittsgewichte und speichere als Bild...")
    fusion_weight_avg_np = fusion_weight_avg.numpy()  # Konvertiere in NumPy-Array

    # Heatmap-Visualisierung
    plt.figure(figsize=(10, 20))
    heatmap = plt.imshow(fusion_weight_avg_np, cmap='viridis', vmin=0, vmax=0.6)  # Begrenze Farben auf 0 bis 0.6
    cbar = plt.colorbar(heatmap, ticks=[0, 0.3, 0.6], label='Weight Value')  # Farblegende mit Label und angepassten Ticks
    cbar.ax.tick_params(labelsize=32)  # Größere Schrift in der Farblegende
    
    plt.xlabel('Width', fontsize=24)
    plt.ylabel('Height', fontsize=24)

    # Bild speichern
    plt.savefig(output_image_name, dpi=300, bbox_inches='tight')  # Speichert die Heatmap als PNG
    print(f"Heatmap wurde als Bild gespeichert: {output_image_name}")
    plt.show()

else:
    print(f"Gewichtungs-Parameter '{weight_key}' nicht im Checkpoint gefunden.")
