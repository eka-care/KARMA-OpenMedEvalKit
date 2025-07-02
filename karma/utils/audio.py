import librosa

def resample_audio(audio_array, orig_sr, target_sr=16000):
        if orig_sr == target_sr:
            return audio_array
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)