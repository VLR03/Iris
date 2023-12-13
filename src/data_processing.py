import os
import numpy as np
from music21 import converter, instrument, note, chord
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def load_midi_files(directory):
    midi_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midi_files.append(os.path.join(directory, filename))
    return midi_files


def extract_notes_chords(midi_file):
    try:
        midi_stream = converter.parse(midi_file)
    except Exception as e:
        print(f"Error parsing MIDI file {midi_file}: {e}")
        return []

    notes = []
    instrument_name = 'Unknown'

    note_code_mapping = {}
    code_counter = 0

    for element in midi_stream.recurse():
        if isinstance(element, instrument.Instrument):
            instrument_name = element.bestName() if element.bestName() else 'Unknown'
        elif isinstance(element, (note.Note, chord.Chord)):
            try:
                if isinstance(element, note.Note):
                    pitch_data = str(element.pitch)
                elif isinstance(element, chord.Chord):
                    pitches = '.'.join(str(n) for n in element.pitches)
                    durations = '.'.join(str(element.duration.quarterLength))
                    pitch_data = f"{pitches};{durations}"

                if pitch_data not in note_code_mapping:
                    note_code_mapping[pitch_data] = code_counter
                    code_counter += 1

                offset = float(element.offset)
                note_data = (note_code_mapping[pitch_data], offset, instrument_name)
                notes.append(note_data)
            except Exception as ex:
                print(f"Error processing element in instrument {instrument_name}: {ex}")
                print(f"Element details: {element}")

    return notes


def process_data(directory):
    midi_files = load_midi_files(directory)
    all_notes = []

    for midi_file in midi_files:
        notes = extract_notes_chords(midi_file)
        all_notes.extend(notes)

    instruments_data = {}
    for note_data in all_notes:
        instrument_name = note_data[-1]
        if instrument_name not in instruments_data:
            instruments_data[instrument_name] = []
        instruments_data[instrument_name].append(note_data[:-1])

    for instrument_name, instrument_data in instruments_data.items():
        if not instrument_data:
            print(f"Warning: No data for instrument {instrument_name}")
            continue

        print(f"Processing instrument: {instrument_name}")
        label_encoder = LabelEncoder()
        try:
            notes_encoded = np.array([note[0] for note in instrument_data])
            print(f"Unique values in instrument_data: {np.unique(notes_encoded)}")
            print(f"Shape of instrument_data before fit_transform: {notes_encoded.shape}")
            scaler = MinMaxScaler()
            encoded_notes = scaler.fit_transform(notes_encoded.reshape(-1, 1))
        except Exception as e:
            print(f"Error processing instrument {instrument_name} : {e}")
            continue

        np.save(os.path.join(directory, f'encoded_notes_{instrument_name}.npy'), encoded_notes)
        label_encoder.fit(notes_encoded)
        transformed_labels = label_encoder.transform(notes_encoded)
        unique_labels = np.unique(transformed_labels)

        # Filter out unseen labels during inverse_transform
        inverse_labels = label_encoder.inverse_transform(unique_labels)
        np.save(os.path.join(directory, f'label_encoder_{instrument_name}.npy'), inverse_labels)


if __name__ == "__main__":
    data_directory = r"C:\Users\Vitor\PycharmProjects\Iris\MidiFiles"
    process_data(data_directory)
