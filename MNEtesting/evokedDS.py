import mne
root = mne.datasets.sample.data_path() / "MEG" / "sample"
raw_file = root / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_file, verbose=False)

events = mne.find_events(raw, stim_channel="STI 014")
# we'll skip the "face" and "buttonpress" conditions to save memory
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True)
evoked = epochs["auditory/left"].average()

del raw  # reduce memory usage
print(0)