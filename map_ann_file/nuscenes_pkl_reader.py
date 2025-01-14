import pickle
import os

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

def print_entry(entry, limit=3):
    output = []
    output.append("Eintrag:")
    output.append(f"  lidar_path: {entry.get('lidar_path', 'Nicht verfügbar')}")
    output.append(f"  token: {entry.get('token', 'Nicht verfügbar')}")
    output.append(f"  prev: {entry.get('prev', 'Nicht verfügbar')}")
    output.append(f"  next: {entry.get('next', 'Nicht verfügbar')}")
    output.append(f"  frame_idx: {entry.get('frame_idx', 'Nicht verfügbar')}")
    output.append(f"  Sweeps (erste {limit} Einträge):")
    sweeps = entry.get('sweeps', [])
    for i, sweep in enumerate(sweeps[:limit]):
        output.append(f"    Sweep {i + 1}:")
        output.append(f"      data_path: {sweep.get('data_path', 'Nicht verfügbar')}")
    return "\n".join(output)

def ask_to_save_output(output):
    choice = input("Möchtest du diese Ausgabe in eine Datei speichern? (ja/nein): ")
    if choice.lower() == 'ja':
        file_name = input("Bitte gib den Namen der Datei ein (ohne Dateiendung): ")
        with open(f"{file_name}.txt", "w") as file:
            file.write(output)
        print(f"Die Ausgabe wurde in {file_name}.txt gespeichert.")

def main():
    pkl_file_path = choose_file()
    if not pkl_file_path:
        print("Keine Datei ausgewählt.")
        return
    if not os.path.exists(pkl_file_path):
        print(f"Die Datei {pkl_file_path} existiert nicht.")
        return
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    entries = data.get('infos', [])
    output = []
    for entry in entries:
        output.append(print_entry(entry))
        output.append("\n" + "="*40 + "\n")
    output_str = "\n".join(output)
    print(output_str)
    ask_to_save_output(output_str)

if __name__ == "__main__":
    main()
