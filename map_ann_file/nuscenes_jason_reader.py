import os
import json

# Funktion zum Auflisten aller JSON-Dateien in einem Verzeichnis und seinen Unterverzeichnissen
def find_json_files(base_dir):
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

# Funktion zum Anzeigen der verfügbaren JSON-Dateien und Auswahl einer Datei
def select_json_file(json_files):
    if not json_files:
        print("Keine JSON-Dateien gefunden.")
        return None

    print("\nVerfügbare JSON-Dateien:")
    for idx, file_path in enumerate(json_files):
        print(f"{idx + 1}. {file_path}")

    while True:
        try:
            choice = int(input(f"\nBitte die Nummer der gewünschten JSON-Datei auswählen (1-{len(json_files)}): "))
            if 1 <= choice <= len(json_files):
                return json_files[choice - 1]
            else:
                print(f"Ungültige Auswahl. Bitte eine Nummer zwischen 1 und {len(json_files)} eingeben.")
        except ValueError:
            print("Bitte eine gültige Nummer eingeben.")

# Funktion zum Anzeigen der Struktur eines JSON-Objekts (für Debugging)
def print_json_structure(data, indent=0):
    # Wenn es ein Dictionary ist, zeige die Schlüssel an
    if isinstance(data, dict):
        for key, value in data.items():
            print(' ' * indent + f"{key}: {type(value).__name__}")
            # Rekursiv die Struktur tieferliegender Ebenen anzeigen
            print_json_structure(value, indent + 2)
    # Wenn es eine Liste ist, zeige das erste Element der Liste an (wenn vorhanden)
    elif isinstance(data, list):
        if len(data) > 0:
            print(' ' * indent + f"List of {len(data)} items")
            print_json_structure(data[0], indent + 2)
        else:
            print(' ' * indent + "Empty list")
    else:
        # Wenn es kein Dictionary oder eine Liste ist, zeige den Datentyp an
        print(' ' * indent + f"Value: {type(data).__name__}")

# Funktion zum Verarbeiten der JSON-Datei mit optionaler Begrenzung von Sample Tokens und Vectors
def process_json_file(file_path, max_samples=5, max_vectors=5):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # JSON-Datei laden
    except json.JSONDecodeError as e:
        print(f"Fehler beim Laden der JSON-Datei: {e}")
        return
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {file_path}")
        return

    # Struktur der JSON-Datei anzeigen
    print(f"\nStruktur der Datei: {file_path}")
    print_json_structure(data)

    # Überprüfen, ob das geladene Objekt ein Dictionary ist
    if isinstance(data, dict):
        print(f"\nVerarbeite Datei: {file_path}")
        # Falls es ein Dictionary ist, gehe über die Schlüssel und Werte
        for key, entry in data.items():
            # Überspringe den Schlüssel 'meta'
            if key == 'meta':
                continue

            if isinstance(entry, list):
                print(f"Verarbeite Einträge für Schlüssel: {key}")
                for i, item in enumerate(entry):
                    if i >= max_samples:
                        print(f"Angezeigte {max_samples} Sample Tokens. Weitere Einträge sind vorhanden.")
                        break
                    if isinstance(item, dict):
                        sample_token = item.get('sample_token', 'kein sample_token')
                        vectors = item.get('vectors', [])
                        print(f"\nSample Token: {sample_token}")
                        vector_count = len(vectors)
                        displayed_vectors = min(max_vectors, vector_count)
                        
                        # Zeige nur die zusammenfassende Information über die Vektoren
                        for j, vector in enumerate(vectors[:displayed_vectors]):
                            cls_name = vector.get('cls_name', 'kein cls_name')
                            pts_num = vector.get('pts_num', 0)
                            print(f"  Class: {cls_name}, Points: {pts_num}")
                        
                        print(f"  Angezeigte {displayed_vectors} von {vector_count} Vectors.")
                        if vector_count > max_vectors:
                            print("  Weitere Vectors sind vorhanden.")
                    else:
                        print(f"Unerwarteter Datentyp in der Liste: {type(item)}")
            else:
                print(f"Unerwarteter Datentyp für den Schlüssel {key}: {type(entry)}")
    else:
        print(f"Die JSON-Datei ist kein Dictionary: {type(data)}")



# Hauptprogramm
if __name__ == "__main__":
    # Verzeichnisse durchsuchen
    base_dirs = [
        "/fzi/ids/qo691/no_backup/MapTR/map_ann_file", 
        "/disk/no_backup/qo691/MapTR/val/work_dirs"
    ]

    all_json_files = []
    for base_dir in base_dirs:
        all_json_files.extend(find_json_files(base_dir))

    # Falls keine JSON-Dateien gefunden werden
    if not all_json_files:
        print("Keine JSON-Dateien in den angegebenen Verzeichnissen gefunden.")
    else:
        # Auswahl der JSON-Datei
        selected_file = select_json_file(all_json_files)

        if selected_file:
            # Ausgewählte Datei verarbeiten mit einer Begrenzung auf 5 Sample Tokens und 5 Vectors
            process_json_file(selected_file, max_samples=5, max_vectors=5)

