import pickle
import os

def load_pkl(pkl_path):
    """Lädt eine .pkl-Datei und gibt den Inhalt zurück."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, pkl_path):
    """Speichert den gegebenen Inhalt in eine .pkl-Datei."""
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def display_elements(infos):
    """Zeigt alle Elemente mit Index an, um eine Auswahl zu ermöglichen."""
    for idx, info in enumerate(infos):
        lidar_path = info.get('lidar_path', 'Unknown')
        print(f"{idx}: {lidar_path}")

def modify_elements(infos):
    """Ermöglicht dem Nutzer, Elemente zu behalten oder zu entfernen."""
    display_elements(infos)
    
    keep_indices = input("Gib die Indizes der Elemente an, die du behalten möchtest (kommagetrennt): ")
    
    # Parse the input into a list of integers
    keep_indices = [int(i.strip()) for i in keep_indices.split(",") if i.strip().isdigit()]
    
    # Behalte nur die Elemente, deren Index angegeben wurde
    filtered_infos = [infos[i] for i in keep_indices if i < len(infos)]
    
    return filtered_infos

def main():
    # Basisverzeichnis des Skripts
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relativer Pfad zur val-pkl-Datei
    pkl_path = os.path.join(script_dir, 'nuscenes_infos_temporal_val-original_modified_2.pkl')

    # Neue pkl-Datei für die bearbeitete Version
    new_pkl_path = os.path.join(script_dir, 'nuscenes_infos_temporal_val-original_modified_2_20_samples.pkl')
        #150, 204, 320, 337, 499, 862, 1231, 1233, 1587, 1728, 1937, 2001, 2141, 3145, 4039, 4876, 5114, 5292, 5925, 5946
    #/datasets/nuScenes/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151214847928.pcd.bin

    
    # Lade die originale .pkl-Datei
    data = load_pkl(pkl_path)
    infos = data.get('infos', [])

    if not infos:
        print("Keine Infos zum Bearbeiten gefunden.")
        return

    # Bearbeiten der Elemente
    print("Welche Elemente möchtest du behalten?")
    modified_infos = modify_elements(infos)

    # Speichere die bearbeitete Version in eine neue .pkl-Datei
    data['infos'] = modified_infos
    save_pkl(data, new_pkl_path)
    

    print(f"Die bearbeitete Datei wurde unter {new_pkl_path} gespeichert.")

if __name__ == '__main__':
    main()
