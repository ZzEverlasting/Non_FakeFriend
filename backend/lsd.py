import pyaudio, pygame
import numpy as np
import torch
import librosa
import time
import asyncio
from groq import Groq

from transformers import pipeline

class LSD:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.running = True

        #groq client
        self.client = Groq(api_key="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")

        #audio features
        self.audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.rate = 44100

        self.now_time = time.time()
        self.interval = 2 # this interval might be too short for not async


    async def classify_audio(self, segment):
        result = self.audio_classifier(segment, sampling_rate=self.rate)
        print(result)

    async def classify_content(self, segment):
        return

    async def audio_processing(self):
        if time.time() - self.now_time > self.interval and len(self.audio_buffer) > self.rate:  #get enough audio to classify
            segment = self.audio_buffer[-self.rate:]    #get last 44100 samples
            await asyncio.gather(
                self.classify_audio(segment),
                self.classify_content(segment)
            )
            self.now_time = time.time()
            self.audio_buffer = []



    def run(self):
        loop = asyncio.get_event_loop()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            data = self.stream.read(1024)
            # Process audio data here, this is the audio stream
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data)) #gather clips

            loop.run_until_complete(self.audio_processing())

            self.screen.fill((0, 0, 0))  # Clear screen
            pygame.display.flip()

        self.cleanup()

    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        pygame.quit()



if __name__ == "__main__":
    lsd = LSD()
    lsd.run()  