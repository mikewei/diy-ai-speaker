import os
import time
import queue
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
from pyannote.audio import Pipeline
from kokoro import KPipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional, NamedTuple, Dict

class Vad:
    def __init__(self, cache_dir='/dev/shm'):
        self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                                use_auth_token=os.environ['HF_TOKEN_FG'])
        self.cache_dir = cache_dir
    def detect(self, audio_array) -> bool:
        audio_file = os.path.join(self.cache_dir, 'vad_callback.wav')
        sf.write(audio_file, audio_array, 16000)
        output = self.pipeline(audio_file)
        for speech in output.get_timeline().support():
            return True
        return False

class S2T:
    def __init__(self, model_id='openai/whisper-large-v3-turbo', sample_rate=16000):
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
    def transcribe(self, audio_array) -> Optional[Dict]:
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            truncation=False,
            return_attention_mask=True,
        ).to(self.device, dtype=self.torch_dtype)
        lang_ids = self.model.detect_language(inputs.input_features)
        if lang_ids[0] != 50259 and lang_ids[0] != 50260:  # 50259: <|en|>, 50260: <|zh|>
            print(f'detected unknown language: {lang_ids}, {self.processor.decode(lang_ids)}')
            return None
        gen_kwargs = {}
        pred_ids = self.model.generate(**inputs, **gen_kwargs)
        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
        return dict(text=pred_text[0], lang='en' if lang_ids[0] == 50259 else 'zh')

class AudioPlayer:
    def __init__(self, duration, samplerate, channels=1):
        self.duration = duration
        self.sample_rate = samplerate
        self.block_size = int(self.sample_rate * self.duration)
        self.channels = channels
        self.playing_audio_q = queue.Queue()
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=self.channels, blocksize=self.block_size,
                                      callback=self.feed_audio)
        self.stream.start()
    def cleanup(self):
        self.stream.stop()
        self.stream.close()
    def play(self, audio_array):
        """ audio_array is a numpy array of shape (n, 1) """
        self.playing_audio_q.put(audio_array)
    def feed_audio(self, outdata, frames, time_info, status):
        if status:
            print(status)
        try:
            block = self.playing_audio_q.get_nowait()
            print(f'frames: {frames}')
            assert block.shape[0] <= frames
            # padding
            if block.shape[0] < frames:
                pad_len = frames - block.shape[0]
                padded = np.pad(block, ((0, pad_len), (0, 0)), mode='constant')
                outdata[:] = padded
            else:
                outdata[:] = block
        except queue.Empty:
            outdata.fill(0)

class AudioProcessor:
    def __init__(self, vad: Vad, s2t: S2T, player: AudioPlayer):
        self.vad = vad
        self.s2t = s2t
        self.player = player
        self.last_block_idx = 0
        self.vad_active_idx = None
        self.audio_buffer = None
        self.prev_audio_array = None
    def process(self, audio_array):
        """ audio_array is a numpy array of shape (n) """
        if self.vad.detect(audio_array):
            print(f'[{self.last_block_idx}] vad detected')
            if self.audio_buffer is None:
                self.audio_buffer = audio_array if self.prev_audio_array is None else np.concatenate([self.prev_audio_array, audio_array])
            else:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            self.vad_active_idx = self.last_block_idx
        elif self.vad_active_idx is not None:
            assert self.audio_buffer is not None
            # append one more block
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            self.player.play(self.audio_buffer.reshape(-1, 1))
            result = self.s2t.transcribe(self.audio_buffer)
            if result is not None:
                print(f'[{self.vad_active_idx}-{self.last_block_idx}] transcribing {result["text"]} ({result["lang"]})')
            self.audio_buffer = None
            self.vad_active_idx = None
        self.prev_audio_array = audio_array
        self.last_block_idx += 1

def recording_loop(duration, samplerate, callback=None):
    recording_audio_q = queue.Queue()
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        recording_audio_q.put(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, blocksize=int(samplerate*duration),
                            callback=audio_callback)
    stream.start()
    try:
        while True:
            block = recording_audio_q.get()
            audio_array = block[:, 0]
            if callback is None:
                print(type(audio_array), audio_array.shape)
            else:
                callback(audio_array)
    except KeyboardInterrupt:
        print('quit recoding')
    finally:
        stream.stop()

class Config(NamedTuple):
    sample_rate: int = 16000
    recording_block_duration: float = 0.5
    playing_block_duration: float = 2
    running_cache_dir: str = '/dev/shm'
    s2t_model_id: str = 'openai/whisper-large-v3-turbo'

if __name__ == '__main__':
    print('Initializing ...', flush=True)
    cfg = Config()
    vad = Vad(cfg.running_cache_dir)
    s2t = S2T(cfg.s2t_model_id, cfg.sample_rate)
    player = AudioPlayer(cfg.playing_block_duration, cfg.sample_rate)
    processor = AudioProcessor(vad, s2t, player)
    print('========== start recording ==========', flush=True)
    recording_loop(cfg.recording_block_duration, cfg.sample_rate, callback=processor.process)