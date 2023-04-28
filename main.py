import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft, fftshift, fftfreq
from scipy.signal import stft
import pandas as pd
import seaborn as sns
from midiutil import MIDIFile

path = "./audio/"
filename = path + "C major scale.mp3"

# Example Frequencies
C1, C2, C3, C4, C5, C6, C7 = 32.7, 65.4, 130.8, 261.6, 523.3, 1046.5, 2093
A1, A2, A3, A4, A5, A6, A7 = 55.0, 110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0
G1, G2, G3, G4, G5, G6, G7 = 49.0, 98.0, 196.0, 392.0, 784.0, 1568.0, 3134.0

# Parameters
amp_threshold = -25 # amplitudes below this amplitude are ignored
beat_length = 20 # error length for beats (to remove degenerate beats)
pitch_frames = 10 # number of additional frames to use when determining pitch
note_difference = 10 # how far apart a group of pitches has to be to be considered a separate note
method = "signal" # which fft method to use: mamual or signal
pitch_method = "groups" # which pitch detection method to use: groups (for multiple notes) or easy
bins = 78 # number of frequency bins (12 bins per octave)

# Dont change these!
sampling = 22050 # frequency of samples
window_length = 2048 # samples per fft window
overlap = 0.5 # percentage overlap between windows
hop_length = int(window_length*(1-overlap)) # distance between windows
hanning = np.hanning(window_length) # a half-cosine to be multiplied onto each window

x, rate = librosa.load(filename, sr = sampling)
samples = int(len(x)/hop_length)
duration = len(x) / sampling

# Using repeated scipy fast fourier transforms (not working)
if method == "manual":
    C = np.empty([samples, window_length])
    for i in range(samples):
        window = x[i*samples:(i*samples) + window_length]
        window = window*hanning # multiply the window by half-cosine to smooth the edges
        window = np.append(window, np.zeros(len(window))) # pad on zeros equal to the window's length
        transform = rfft(window) # doing the real-input version cuts the number of elements in half, hence the padding
        print(transform)
        transform = transform[0:-1] # rfft results in one extra element for some reason
        transform = transform / window_length # Parseval's theorem
        transform = np.abs(transform * np.conj(transform)) # autopower! (to make it a one-sided spectrum)
        #transform = 20*np.log10(transform) # convert to dB
        #C[i] = fftshift(transform)
        C[i] = transform

    #C = np.transpose(C)
    notes = librosa.amplitude_to_db(np.abs(C))

    frame = pd.DataFrame(C)
    print(frame)
    #sns.set()
    #ax = sns.heatmap(frame, cmap='coolwarm')

    #print(C)
    #freq = fftfreq(samples, hop_length)
    plt.figure()
    #librosa.display.specshow(notes, x_axis = "time", y_axis = "log")
    plt.imshow(C, aspect="auto", origin = "lower", cmap = "coolwarm")

# Using scipy's stft
elif method == "signal":
    C = stft(x, sampling, window = "hann", nfft = window_length, padded = True) # perform the repeated ffts
    notes = librosa.amplitude_to_db(abs(C[2]))
    notes[notes < amp_threshold] = -120 # remove values below a certain threshold
    notes = notes[0:bins] # use only the number of specified frequency bins

    # Create a matrix that's the change in amplitude between samples for each frequency
    d_notes = np.empty([notes.shape[0], notes.shape[1] - 1])
    d_notes = d_notes.transpose()
    bars = [] # a list that's a number of frequencies that saw an increase above a given threshold
    beat_threshold = 1
    for i, frame in enumerate(d_notes):
        count = 0
        for j, amp in enumerate(d_notes[i]):
            delta = notes[j][i+1] - notes[j][i]
            if delta < 0: d_notes[i][j] = 0 # only take positive changes
            else: d_notes[i][j] = delta

            if delta >= beat_threshold: count += 1
        bars.append(count)

    # Use the bars list to identify beats
    increases = [i for i in bars if i != 0]
    average = float(sum(increases) / len(increases))
    beats = []
    for i, increase in enumerate(bars):
        if increase >= average: # only consider the larger half of increases
            beats.append(i)
    
    # Remove repeated beats
    beat_groups = []
    beat_groups.append([beats[0]])
    last_beat = beats.pop(0)
    while len(beats) > 0:
        if (beats[0] - last_beat) > beat_length:
            beat_groups.append([beats[0]])
        else:
            beat_groups[-1].append(beats[0])
        last_beat = beats.pop(0)

    for group in beat_groups:
        beats.append(int(sum(group)/len(group)))

    # Identify the pitch of each beat
    pitches = []
    notes = notes.transpose()
    if pitch_method == "groups":
        beats_to_add = []
        for index, frame in enumerate(beats):
            beat_pitches = []
            for step in range(pitch_frames): # For each beat, repeat the process for some frames afterwards to reduce variance
                frequencies = notes[frame + step]
                d_frequencies = []
                # Create a list that's the derivative of the frame's frequencies
                for i in range(len(frequencies) - 1):
                    d_frequencies.append(frequencies[i+1] - frequencies[i])
                
                # Use the derivative to identify groups of frequencies (each of which corresponds to one note)
                groups = []
                for i in range(len(d_frequencies) - 1):
                    if (d_frequencies[i] <= 0 and d_frequencies[i+1] > 0): # Going down to going up denotes the start of a group
                        groups.append([i])
                    elif (d_frequencies[i] * d_frequencies[i+1] > 0): # Continuing in the same direction means this frequency is part of the same group
                        groups[-1].append(i)            
                for group in groups:
                    if len(group) >= 5: beat_pitches.append(sum(group)/len(group)) # add the groups to a grand list of groups for the current beat, ignoring tiny groups
                    #if len(group) >= 10: beat_pitches.append(np.median(group))
            
            # split all the pitches for the beat into the different notes (pitches close together will be averaged to one note)
            beat_pitches.sort()
            pitch_groups = []
            for i, pitch in enumerate(beat_pitches):
                if i == len(beat_pitches) - 1: pitch_groups[-1].append(pitch)
                elif i == 0: pitch_groups.append([pitch])
                else:
                    if beat_pitches[i+1] - beat_pitches[i] >= note_difference: pitch_groups.append([pitch])
                    else: pitch_groups[-1].append(pitch)
            
            # Record the average pitch of each group
            for i, group in enumerate(pitch_groups):
                if i == 0:
                    pitches.append(sum(group)/len(group))
                else:
                    pitches.append(sum(group)/len(group))
                    beats_to_add.append([index, beats[index]])
        
        for beat in beats_to_add: # adding in extra elements to the list of beats for additional notes
            beats.insert(beat[0], beat[1])

    elif pitch_method == "easy":
        for frame in beats:
            beat_pitches = []
            for step in range(pitch_frames):
                frequencies = []
                for frequency, amp in enumerate(notes[frame + step]):
                    if amp > -120: frequencies.append(frequency)
                if len(frequencies) > 0:
                    beat_pitches.append(np.median(frequencies)) # using median
                    #beat_pitches.append(sum(frequencies)/len(frequencies)) # using average
            pitches.append(sum(beat_pitches)/len(beat_pitches))

    notes = notes.transpose()

    # Plot it
    plt.imshow(notes, aspect = "auto", interpolation = "none")
    plt.scatter(beats, pitches)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.vlines(beats, ymin = 0, ymax = bins-1, color = "white", linewidth = 0.5)
    plt.hlines(24, xmin = 0, xmax = len(notes[0]) - 1, color = "black")
    plt.title("C Major Scale")
    yticks = np.linspace(0, bins, 10)
    ylabels = [int(C[0][int(i)]) for i in yticks]
    plt.yticks(ticks = yticks, labels = ylabels)
    plt.xlabel("Frames")
    plt.ylabel("Frequency (Hz)")

    # Write a MIDI file
    tempo = 35000
    ticks_per_quarternote = int(sampling*60/(tempo))
    midi = MIDIFile(1, eventtime_is_ticks = True, ticks_per_quarternote=ticks_per_quarternote)
    midi.addTempo(0, 0.0, tempo)
    for i, note in enumerate(pitches):
        duration = int(1.5*ticks_per_quarternote)
        pitch = (12*(np.log(C[0][int(note)]/220)/np.log(2))) + 57
        midi.addNote(0, 0, int(pitch), int(beats[i]), duration, 127)
    
    with open("output.midi", "wb") as output_file:
        midi.writeFile(output_file)

plt.show()