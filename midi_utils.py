import pretty_midi
import numpy as np
from tokenizer import PAD

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def save_midi(_input, length, filename, fs=5):
    result = np.zeros((128, length + 1), dtype=np.int16)
    for idx, note in enumerate(_input):
        notes = note.split(" ")
        for _part in note.split(" "):
            if _part != PAD:
                result[int(_part), idx] = 1
    generate_to_midi = piano_roll_to_pretty_midi(result, fs=fs)
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    generate_to_midi.write(filename)
