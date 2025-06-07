from typing import Iterable, Optional, NamedTuple, Dict, Literal, List, Tuple, Iterator
from abc import ABC, abstractmethod
import os
import time
import queue
import itertools
import threading
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import openai
from pyannote.audio import Pipeline
from kokoro import KModel, KPipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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
    def __init__(self, model_id, samplerate, device):
        self.model_id = model_id
        self.sample_rate = samplerate
        self.device = device
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
        lang_code, lang_prob = self.detect_language(inputs.input_features, ['en', 'zh'], 0.7)
        if lang_code is None:
            return None
        gen_kwargs = {}
        pred_ids = self.model.generate(**inputs, **gen_kwargs)
        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)
        return dict(text=pred_text[0], lang=lang_code, prob=lang_prob)

    def detect_language(self, input_features, lang_codes: List[str], threshold: float = 0) -> Tuple[Optional[str], float]:
        req_lang_ids = [self.model.generation_config.lang_to_id[f'<|{lang_code}|>'] for lang_code in lang_codes]
        decoder_input_ids = (
            torch.ones((input_features.shape[0], 1), device=self.device, dtype=torch.long)
            * self.model.generation_config.decoder_start_token_id
        )
        with torch.no_grad():
            logits = self.model(input_features, decoder_input_ids=decoder_input_ids, use_cache=False).logits[:, -1]
        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(self.model.generation_config.lang_to_id.values())] = False
        logits[:, non_lang_mask] = -np.inf
        probs = torch.softmax(logits, dim=-1)
        predicted_lang_id, predicted_lang_prob = probs.argmax(-1)[0], probs.max(-1)[0].item()
        if predicted_lang_id in req_lang_ids:
            predicted_lang_code = lang_codes[req_lang_ids.index(predicted_lang_id)]
            if predicted_lang_prob > threshold:
                return predicted_lang_code, predicted_lang_prob
            else:
                print(f'detected language {predicted_lang_code} with low probability {predicted_lang_prob}')
        else:
            print(f'detected invalid language: {self.processor.decode(predicted_lang_id)}')
        return None, predicted_lang_prob


class TTS:
    def __init__(self, model_id, samplerate, device):
        self.model_id = model_id
        self.sample_rate = samplerate
        self.device = device
        assert self.sample_rate == 24000
        en_pipeline = KPipeline(lang_code='a', repo_id=self.model_id, model=False)

        def en_callable(text):
            if text == 'Kokoro':
                return 'kˈOkəɹO'
            elif text == 'Sol':
                return 'sˈOl'
            return next(en_pipeline(text)).phonemes

        model = KModel(repo_id=self.model_id).to(self.device).eval()
        self.pipeline = KPipeline(lang_code='z', repo_id=self.model_id, model=model, en_callable=en_callable)

    def generate_speech(self, text: str) -> Iterator[torch.FloatTensor]:
        # HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
        # Simple piecewise linear fn that decreases speed as len_ps increases
        def speed_callable(len_ps):
            speed = 0.8
            if len_ps <= 83:
                speed = 1
            elif len_ps < 183:
                speed = 1 - (len_ps - 83) / 500
            return speed * 1.1

        def transform_result(result):
            print(f'generated speech phonemes: {result.phonemes}')
            return result.audio.reshape(-1, 1)

        # todo: when text is too long, the generator will truncate the speech
        generator = self.pipeline(text, voice='zf_001', speed=speed_callable)
        return map(transform_result, generator)


class TextSpeaker(ABC):
    @abstractmethod
    def response(self, text: str):
        ...

    @abstractmethod
    def last_active_time(self):
        ...


class TextProcessor(ABC):
    @abstractmethod
    def request(self, text: str, speaker: TextSpeaker):
        ...


class EchoTextProcessor(TextProcessor):
    def request(self, text: str, speaker: TextSpeaker):
        speaker.response(text)


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict]):
        return self.client.chat.completions.create(model="qwen3-14b-diy", messages=messages)   # type: ignore


class SimpleChatProcessor(TextProcessor):
    class HistoryMsgs(NamedTuple):
        timestamp: float
        messages: List[Dict]
        tokens: int

    def __init__(self, chat_client: ChatClient, max_input_tokens: int, max_context_secs: int):
        self.chat_client = chat_client
        self.max_input_tokens = max_input_tokens
        self.max_context_secs = max_context_secs
        self.history: List[SimpleChatProcessor.HistoryMsgs] = []
        self.sys_prompt = '\n'.join([
            "- You are now a voice assistant, use short oral language to answer the user's question.",
            "- You are chatting with a user named Max.",
        ])
        self.sys_message = [{"role": "system", "content": self.sys_prompt}]
        self.sys_message_tokens = self.calc_tokens(self.sys_message)

    def request(self, text: str, speaker: TextSpeaker):
        try:
            user_message = [{"role": "user", "content": text}]
            history_messages = self.get_history_messages()
            input_messages = self.sys_message + history_messages + user_message
            print(f'chat input:\n{input_messages}')
            response = self.chat_client.chat(input_messages)
            self.history.append(self.HistoryMsgs(time.time(), user_message, self.calc_tokens(user_message)))
        except Exception as e:
            print(f'chat failed: {e}')
            return
        output_message = response.choices[0].message
        print(f'chat output:\n{output_message}')
        if output_message.content is not None:
            speaker.response(output_message.content)
            assistant_message = [{'role': output_message.role, 'content': output_message.content}]
            self.history.append(self.HistoryMsgs(time.time(), assistant_message, self.calc_tokens(assistant_message)))
        else:
            print(f'chat response is None')

    def get_history_messages(self) -> List[Dict]:
        result_pos = len(self.history)
        left_tokens: int = self.max_input_tokens - self.sys_message_tokens
        for i in range(len(self.history)-1, -1, -1):
            left_tokens -= self.history[i].tokens
            if left_tokens < self.sys_message_tokens:
                break
            if self.history[i].timestamp < time.time() - self.max_context_secs:
                break
            # the first message following the system message must be a user message
            if self.history[i].messages[0]['role'] == 'user':
                result_pos = i
        self.history = self.history[result_pos:]
        return list(itertools.chain.from_iterable(msgs.messages for msgs in self.history))

    def calc_tokens(self, messages: List[Dict]) -> int:
        """ a naive implementation of counting tokens """
        return sum(len(message['content'].split()) + 2 for message in messages)

    
class AudioPlayer:
    class Msg(NamedTuple):
        audio_array: np.ndarray
        block_idx: int

    def __init__(self, duration, samplerate, channels=1):
        self.duration = duration
        self.sample_rate = samplerate
        self.block_size = int(self.sample_rate * self.duration)
        self.channels = channels
        self.playing_audio_q = queue.Queue()
        self.q_reset_lock = threading.Lock()
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=self.channels, blocksize=self.block_size,
                                      callback=self.feed_audio)
        self.stream.start()

    def cleanup(self):
        self.stream.stop()
        self.stream.close()

    def play(self, audio_array, reset=False):
        """ audio_array is a numpy array of shape (n, channels) """
        assert audio_array.shape[1] == self.channels
        if reset:
            self.stop()
        for idx in range((audio_array.shape[0] + self.block_size - 1) // self.block_size):
            msg = self.Msg(audio_array, idx)
            self.playing_audio_q.put(msg)

    def stop(self):
        with self.q_reset_lock:
            self.playing_audio_q.queue.clear()

    def feed_audio(self, outdata, frames, time_info, status):
        assert self.block_size == frames
        if status:
            print(status)
        try:
            with self.q_reset_lock:
                msg = self.playing_audio_q.get_nowait()
            left_block_size = int(msg.audio_array.shape[0]) - int(msg.block_idx * self.block_size)
            assert left_block_size > 0
            # print(f'start playing block {msg.block_idx}')
            # padding
            if left_block_size < self.block_size:
                block = msg.audio_array[msg.block_idx * self.block_size :]
                pad_len = self.block_size - left_block_size
                padded = np.pad(block, ((0, pad_len), (0, 0)), mode='constant')
                outdata[:] = padded
            else:
                outdata[:] = msg.audio_array[msg.block_idx * self.block_size : (msg.block_idx + 1) * self.block_size]
        except queue.Empty:
            outdata.fill(0)


class AudioProcessor(TextSpeaker):
    def __init__(self, vad: Vad, s2t: S2T, tts: TTS, player: AudioPlayer, processor: TextProcessor):
        self.vad = vad
        self.s2t = s2t
        self.tts = tts
        self.player = player
        self.processor = processor
        self.last_block_idx = 0
        self.vad_active_idx = None
        self.audio_buffer = None
        self.prev_audio_array = None

    def process(self, audio_array):
        """ audio_array is a numpy array of shape (n) """
        detected_audio = audio_array if self.prev_audio_array is None else np.concatenate([self.prev_audio_array, audio_array])
        if self.vad.detect(detected_audio):
            if self.audio_buffer is None:
                self.audio_buffer = audio_array if self.prev_audio_array is None else np.concatenate([self.prev_audio_array, audio_array])
                print('vad detected, start recording ...')
            else:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            self.vad_active_idx = self.last_block_idx
            self.vad_active_time = time.time()
        elif self.vad_active_idx is not None:
            assert self.audio_buffer is not None
            # do not append one more block for latency reduction
            # self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            result = self.s2t.transcribe(self.audio_buffer)
            if result is not None:
                print(f'[{self.vad_active_idx}-{self.last_block_idx}] transcribing {result["text"]} ({result["lang"]}, {result["prob"]:.2f})')
                # audio echo test
                # if result["lang"] == 'zh':
                #     self.player.stop()
                #     self.player.play(self.audio_buffer.reshape(-1, 1))
                # tts echo test
                # speech = self.tts.generate_speech(result["text"])
                # if speech is not None:
                #     self.player.play(speech.reshape(-1, 1))
                # else:
                #     print(f'speech generation failed')
                # let's do chat
                self.processor.request(result["text"], self)
            self.audio_buffer = None
            self.vad_active_idx = None
        self.prev_audio_array = audio_array
        self.last_block_idx += 1

    def response(self, text: str):
        speech_gen = self.tts.generate_speech(text)
        first_speech = True
        for speech in speech_gen:
            self.player.play(speech, reset=first_speech)
            if first_speech:
                print(f'playing speech ... ')
            first_speech = False

    def last_active_time(self) -> float:
        return self.vad_active_time


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
    recording_sample_rate: int = 16000
    recording_block_duration: float = 0.5
    playing_sample_rate: int = 24000
    playing_block_duration: float = 0.25
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    running_cache_dir: str = '/dev/shm'
    s2t_model_id: str = 'openai/whisper-large-v3-turbo'
    tts_model_id: str = 'hexgrad/Kokoro-82M-v1.1-zh'
    chat_api_key: str = 'dummy-key'
    chat_base_url: str = 'http://localhost:8008/v1'
    chat_model: str = 'qwen3-14b-diy'
    chat_max_input_tokens: int = 2000
    chat_max_context_secs: int = 300


if __name__ == '__main__':
    print('Initializing ...', flush=True)
    cfg = Config()
    vad = Vad(cfg.running_cache_dir)
    s2t = S2T(cfg.s2t_model_id, cfg.recording_sample_rate, cfg.device)
    tts = TTS(cfg.tts_model_id, cfg.playing_sample_rate, cfg.device)
    chat_client = ChatClient(cfg.chat_api_key, cfg.chat_base_url, cfg.chat_model)
    text_processor = SimpleChatProcessor(chat_client, cfg.chat_max_input_tokens, cfg.chat_max_context_secs)
    audio_player = AudioPlayer(cfg.playing_block_duration, cfg.playing_sample_rate)
    audio_processor = AudioProcessor(vad, s2t, tts, audio_player, text_processor)
    print('========== start recording ==========', flush=True)
    recording_loop(cfg.recording_block_duration, cfg.recording_sample_rate, callback=audio_processor.process)