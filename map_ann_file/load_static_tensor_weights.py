import torch

# Pfad zur Checkpoint-Datei
checkpoint_path = "/disk/vanishing_data/qo691/MapTR/work_dirs/maptr_tiny_r50_24e_cwl_fusion_b4_w4_a8_XU_PreTra_Re_Sk_BHW_1/epoch_24.pth"

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
    print("Fusion Weight Map (Normal):")
    print(weights)

    # Durchschnitt über die Channels berechnen
    # Gewichte haben normalerweise die Form (out_channels, width, length)
    # Um den Durchschnitt zu berechnen, mitteln wir über die erste Dimension (Channels)
    fusion_weight_avg = weights.mean(dim=0)  # Durchschnitt über Channels
    print("\nFusion Weight Map (Durchschnitt über Channels):")

    torch.set_printoptions(threshold=20_000)

    
    print(fusion_weight_avg)

    # Optional: Speichern der Ergebnisse für spätere Analyse oder Visualisierung
    torch.save(weights, "fusion_weight_map.pt")
    torch.save(fusion_weight_avg, "fusion_weight_avg.pt")
    print("\nFusion Weight Maps wurden in 'fusion_weight_map.pt' und 'fusion_weight_avg.pt' gespeichert.")

else:
    print(f"Gewichtungs-Parameter '{weight_key}' nicht im Checkpoint gefunden.")
