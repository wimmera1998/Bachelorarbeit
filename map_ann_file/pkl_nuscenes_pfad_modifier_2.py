import pickle
import os

# Dateiauswahl aus allen .pkl-Dateien im aktuellen Verzeichnis
def choose_file():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_files = [f for f in os.listdir(script_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        print("Keine .pkl-Dateien im Verzeichnis gefunden.")
        return None

    print("Bitte wähle eine .pkl-Datei zum Auslesen:")
    for idx, file_name in enumerate(pkl_files):
        print(f"{idx + 1}: {file_name}")
    
    choice = input(f"Gib die Nummer der Datei ein (1-{len(pkl_files)}): ")
    
    if choice.isdigit() and 1 <= int(choice) <= len(pkl_files):
        return os.path.join(script_dir, pkl_files[int(choice) - 1])
    else:
        print("Ungültige Auswahl. Bitte versuche es erneut.")
        return choose_file()

# Modifiziert die Pfade
def modify_paths(entries, chars_to_remove):
    for entry in entries:
        if 'sweeps' in entry:
            for sweep in entry['sweeps']:
                if 'data_path' in sweep and sweep['data_path'].startswith('/'):
                    sweep['data_path'] = sweep['data_path'][chars_to_remove:]  # Entferne die gewählte Anzahl von Zeichen

        if 'cams' in entry:
            for cam in entry['cams'].values():
                if 'data_path' in cam and cam['data_path'].startswith('/'):
                    cam['data_path'] = cam['data_path'][chars_to_remove:]  # Entferne die gewählte Anzahl von Zeichen

# Hauptfunktion
def main():
    pkl_file_path = choose_file()
    
    if not pkl_file_path:
        print("Keine Datei ausgewählt.")
        return

    if not os.path.exists(pkl_file_path):
        print(f"Die Datei {pkl_file_path} existiert nicht.")
        return

    # Lade die .pkl-Datei
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)

    # Zeige das erste Beispiel an und frage nach der Anzahl der Zeichen, die entfernt werden sollen
    entries = data.get('infos', [])
    first_entry_path = None
    if entries and 'sweeps' in entries[0] and entries[0]['sweeps']:
        first_entry_path = entries[0]['sweeps'][0]['data_path']
    elif entries and 'cams' in entries[0] and entries[0]['cams']:
        first_entry_path = next(iter(entries[0]['cams'].values()))['data_path']

    if first_entry_path:
        print(f"Beispielpfad: {first_entry_path}")
        print(f"Pfad-Vorschau (erste drei Ordner): {'/'.join(first_entry_path.split('/')[:4])}/")
        chars_to_remove = int(input("Wieviele Zeichen sollen von links entfernt werden? "))

        # Verarbeite und modifiziere die Daten
        modify_paths(entries, chars_to_remove)

        # Speichere die modifizierten Daten in einer neuen .pkl-Datei
        output_file_path = os.path.splitext(pkl_file_path)[0] + '_modified_2.pkl'
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(data, output_file)

        print(f"Modifizierte .pkl-Datei gespeichert: {output_file_path}")
    else:
        print("Keine Pfade gefunden, die modifiziert werden können.")

# Skript ausführen
if __name__ == "__main__":
    main()
