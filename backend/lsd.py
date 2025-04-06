import pyaudio, pygame
import numpy as np
import torch
import librosa
import time

from transformers import pipeline

class LSD:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.running = True
        self.audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.rate = 44100
        self.now_time = time.time()
        self.interval = 2 # this interval might be too short for not async


    def classify_audio(self):
        segment = self.audio_buffer[-self.rate:]    #get last 44100 samples
        result = self.audio_classifier(segment, sampling_rate=self.rate)
        self.now_time = time.time()
        print(result)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            data = self.stream.read(1024)
            # Process audio data here, this is the audio stream
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))

            if time.time() - self.now_time > self.interval and len(self.audio_buffer) > self.rate:  #get enough audio to classify
                self.classify_audio(self)
                self.classify_content(self)

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