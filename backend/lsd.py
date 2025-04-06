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
        self.audio_model = "superb/wav2vec2-base-superb-er"
        self.text_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        #tweakables
        self.rate = 16000   #good sample rate for whisper but idk how good it actualy is
        self.threshold = 0    # volume threshold for audio classification lower = more sensitive
        self.interval = 4         # tweak this interval (in sec) to get more or less audio for classification


        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=1024)
        
        #pygame window
        pygame.init()
        self.width, self.height = 640, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.running = True

        #groq client
        self.client = Groq(api_key="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")
        self.language = 'en'

        #audio features
        self.audio_classifier = pipeline("audio-classification", model=self.audio_model, device=0) #use device=0 for GPU, -1 for CPU
        self.audio_buffer = np.array([], dtype=np.int16)

        self.now_time = time.time()


    async def classify_audio(self, segment):
        filename = "emoTemp.wav"
        sf.write(filename, segment, self.rate)  # rewrite sample rate
        sentiment = await asyncio.to_thread(self.audio_classifier, filename, sampling_rate=self.rate)
        os.remove(filename)
        return sentiment[0]['label'], sentiment[0]['score']



    async def analyze_content(self, segment):
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
        # print(transcription.text)

        sentiment_pipeline = pipeline("sentiment-analysis", model=self.text_model, tokenizer=self.text_model, device=0)
        sentiment = sentiment_pipeline(transcription.text)
        return sentiment[0]['label'], sentiment[0]['score'], transcription.text

    

    async def audio_processing(self):
        if time.time() - self.now_time > self.interval:  #get enough audio to classify
            self.now_time = time.time()
            if len(self.audio_buffer) > self.rate * self.interval:
                segment = self.audio_buffer[-self.rate * self.interval:]    #get last #RATE samples
                self.audio_buffer = []

                if np.mean(np.abs(segment)) < self.threshold:
                    print("Segment too quiet, skipping...")
                    return
            
                results = await asyncio.gather(
                    self.classify_audio(segment),
                    self.analyze_content(segment)
                )
                return results
            
    #sentiment 2 vec. add more to the emotion map for fun
    def sentiment2vec(self, label_audio, label_text, score_audio, score_text):
        audio_sentiment_map = {"ang": 0.9, "hap": 0.6, "sad": 0.1, "neu": 0.3}
        text_sentiment_map = {"positive": 1, "neutral": 0.2, "negative": -1}
        hype = audio_sentiment_map[label_audio] * score_audio
        attitude = text_sentiment_map[label_text]  * score_text
        intensity = abs(attitude) + hype / 2
        return [hype, attitude, intensity]


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

            res = loop.run_until_complete(self.audio_processing())
            if(res):
                label1, label2, score1, score2 , trascript = res[0][0], res[1][0], res[0][1], res[1][1], res[1][2]
                emo_vec = self.sentiment2vec(label1, label2, score1, score2)
                print(emo_vec)

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
     