import pandas as pd
import os
import joblib


def save_dataframe_to_csv(df, filename='output.csv', index=False, sep=',', encoding='utf-8'):
    """
    Speichert ein pandas DataFrame als CSV-Datei.
    Saves a pandas DataFrame as a CSV file.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Das übergebene Objekt ist kein pandas DataFrame. / The provided object is not a pandas DataFrame.")

    # Absoluten Pfad bestimmen und Verzeichnis erstellen, falls nötig
    abs_path = os.path.abspath(filename)
    dir_path = os.path.dirname(abs_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    try:
        df.to_csv(abs_path, index=index, sep=sep, encoding=encoding)
        print(f"Datei erfolgreich gespeichert unter: {abs_path} / File successfully saved at: {abs_path}")
    except Exception as e:
        print(f"Fehler beim Speichern der Datei: {e} / Error saving file: {e}")


def load_dataframe_from_csv(filepath, sep=',', encoding='utf-8'):
    """
    Lädt eine CSV-Datei in ein pandas DataFrame.
    Loads a CSV file into a pandas DataFrame.

    Parameter / Parameters:
    - filepath: Pfad zur CSV-Datei / Path to the CSV file.
    - sep: Trennzeichen der Datei / Separator used in the file.
    - encoding: Zeichenkodierung der Datei / File encoding.

    Rückgabe / Returns:
    - pd.DataFrame: Das geladene DataFrame / The loaded DataFrame.
    """
    # Prüfen, ob die Datei existiert / Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath} / File not found: {filepath}")

    try:
        # CSV-Datei laden / Load CSV file
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        print(f"Datei erfolgreich geladen: {filepath} / File successfully loaded: {filepath}")
        return df
    except Exception as e:
        # Fehlerbehandlung beim Laden / Error handling when loading
        print(f"Fehler beim Laden der Datei: {e} / Error loading file: {e}")
        return None


def save_model(model, path):
    """
    Speichert ein CatBoost-Modell an einem gegebenen Pfad.
    """
    model.save_model(path)
    print(f" Modell gespeichert unter: {path}")

def load_model(path):
    """
    Lädt ein Modell-Objekt von einem gegebenen Pfad.
    """
    model = joblib.load(path)
    print(f" Modell geladen von: {path}")
    return model
