import os
import glob
import soundfile as sf
from pathlib import Path

audio_dir = "/mount/beat-tracking/beatles/data"
annot_dir = "/mount/beat-tracking/beatles/label"

#--audio_dir /home/cjstein/datasets/BallroomData \
#--annot_dir /home/cjstein/datasets/BallroomAnnotations \

#--audio_dir /home/cjstein/datasets/The_Beatles \
#--annot_dir /home/cjstein/datasets/The_Beatles_Annotations/beat/The_Beatles \

#--audio_dir /home/cjstein/datasets/hainsworth/beat \
#--annot_dir /home/cjstein/datasets/hainsworth/wavs \

audio_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"))
annot_files = glob.glob(os.path.join(annot_dir, "**", "*.txt"))

for idx, audio_file in enumerate(audio_files):
    audio, sr = sf.read(audio_file)
    print(idx, audio.shape)

    audioL = audio[:,0]
    audioR = audio[:,1]
    audioM = (audioL + audioR)/2

    audio_file_dir = str(Path(audio_file).resolve().parent)
    if not os.path.exists(audio_file_dir.replace("/data", "/data_lr")):
        os.makedirs(audio_file_dir.replace("/data", "/data_lr"))
    
    audio_file = audio_file.replace("/data/", "/data_lr/")

    left_filename = audio_file.replace(".wav", "_L.wav")
    right_filename = audio_file.replace(".wav", "_R.wav")
    mono_filename = audio_file.replace(".wav", "_L+R.wav")

    sf.write(left_filename, audioL, sr)
    sf.write(right_filename, audioR, sr)
    sf.write(mono_filename, audioM, sr)

