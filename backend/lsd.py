import pyaudio, pygame
import numpy as np
import librosa
import time
import asyncio
import soundfile as sf
import os
import scipy.signal
from groq import Groq

from transformers import pipeline

class LSD:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.rate = 16000
        self.threshold = 200
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=1024)
        pygame.init()

        #pygame window
        self.width, self.height = 640, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.running = True

        #groq client
        self.client = Groq(api_key="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")
        self.language = 'en'

        #audio features
        self.audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
        self.audio_buffer = np.array([], dtype=np.int16)

        self.now_time = time.time()

        # tweak this interval to get more or less audio for classification
        self.interval = 3 


    async def classify_audio(self, segment):
        filename = "emoTemp.wav"
        sf.write(filename, segment, self.rate)  # rewrite sample rate
        result = await asyncio.to_thread(self.audio_classifier, filename, sampling_rate=self.rate)
        os.remove(filename)
        print(result)
        return

    async def classify_content(self, segment):
        filename = "temp.wav"
        sf.write(filename, segment, self.rate)
        with open(filename, "rb") as file:
            transcription = await asyncio.to_thread(
                self.client.audio.transcriptions.create,
                file=file,
                model="whisper-large-v3-turbo",
                language=self.language,
                timestamp_granularities = ["word", "segment"],
                response_format="verbose_json",
            )
        os.remove(filename)
        print(transcription.text)
        return


    async def audio_processing(self):
        if time.time() - self.now_time > self.interval:  #get enough audio to classify
            self.now_time = time.time()
            if len(self.audio_buffer) > self.rate * self.interval:
                segment = self.audio_buffer[-self.rate * self.interval:]    #get last #RATE samples
                self.audio_buffer = []

                if np.mean(np.abs(segment)) < self.threshold:
                    print("Segment too quiet, skipping...")
                    return
            
                await asyncio.gather(
                    self.classify_audio(segment),
                    self.classify_content(segment)
                )
        return
            
            
    def sentiment_function():
        return


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
        return

    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        pygame.quit()
        return



if __name__ == "__main__":
    lsd = LSD()
    lsd.run() 

    #test groq client
    # client = Groq(api_key="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")
    # with open("Recording.wav", "rb") as file:
    #         transcription = client.audio.transcriptions.create(
    #             file=file,
    #             model="whisper-large-v3-turbo",
    #             language='en',
    #             timestamp_granularities = ["word", "segment"],
    #             response_format="verbose_json",)
    # print(transcription.text)
     