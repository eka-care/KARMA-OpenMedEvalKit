import torchaudio

def resample_audio(audio_array, orig_sr, target_sr=16000):
        if orig_sr == target_sr:
            return audio_array
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resampler(audio_array)