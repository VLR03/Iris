import os
import numpy as np
from music21 import converter, instrument, note, chord, stream
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_midi_files(directory):
    midi_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi_files.append(os.path.join(directory, filename))
    return midi_files


def extract_notes_chords(midi_file):
    midi_stream = converter.parse(midi_file)
    parts = instrument.partitionByInstrument(midi_stream)

    notes = []
    for part in parts:
        if isinstance(part, instrument.Instrument):
            notes.append(part.instrumentName)
        for element in part.flat:
            if isinstance(element, note.Note):
                notes.append((element.pitch, element.offset, part.partName))
            elif isinstance(element, chord.Chord):
                notes.append(('.'.join(str(n) for n in element.pitches), element.offset, part.partName))

    return notes


def process_data(directory):
    midi_files = load_midi_files(directory)
    all_notes = []

    for midi_file in midi_files:
        notes = extract_notes_chords(midi_file)
        all_notes.extend(notes)

    # Separar os dados por instrumento
    instruments_data = {}
    for note_data in all_notes:
        instrument_name = note_data[-1]  # Último elemento é o nome do instrumento
        if instrument_name not in instruments_data:
            instruments_data[instrument_name] = []
        instruments_data[instrument_name].append(note_data[:-1])  # Remover o nome do instrumento

    # Salvando dados para cada instrumento
    for instrument_name, instrument_data in instruments_data.items():
        label_encoder = LabelEncoder()
        encoded_notes = label_encoder.fit_transform(instrument_data)

        np.save(os.path.join(directory, f'encoded_notes_{instrument_name}.npy'), encoded_notes)
        np.save(os.path.join(directory, f'label_encoder_{instrument_name}.npy'), label_encoder.classes_)


if __name__ == "__main__":
    data_directory = "MidiFiles"
    process_data(data_directory)
