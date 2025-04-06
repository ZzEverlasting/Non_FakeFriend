import pyaudio, pygame
import numpy as np
import librosa
import time, os, random
import asyncio
import soundfile as sf
from colorsys import hsv_to_rgb
from groq import Groq

from transformers import pipeline

class LSD:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.audio_model = "superb/wav2vec2-base-superb-er"
        self.text_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.language = 'en'
        self.width, self.height = 640, 480

        #tweakables
        self.rate = 16000   #good sample rate for whisper but idk how good it actualy is
        self.threshold = 0    # volume threshold for audio classification lower = more sensitive
        self.interval = 4         # tweak this interval (in sec) to get more or less audio for classification


        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=1024)

        
        #pygame window
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.running = True

        #groq client
        self.client = Groq(api_key="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")


        #audio features
        self.audio_classifier = pipeline("audio-classification", model=self.audio_model, device=0) #use device=0 for GPU, -1 for CPU
        self.audio_buffer = np.array([], dtype=np.int16)

        #text features
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.text_model, tokenizer=self.text_model, device=0)

        self.now_time = time.time()
        self.ttime=0
        self.emo_vec = [0.0, 0.0, 0.0]
        self.upcoming_emo_vec = [0.0, 0.0, 0.0]

        self.analysis_task = None
        self.new_data_ready = False



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
        sentiment = await asyncio.to_thread(self.sentiment_pipeline, transcription.text)
        return sentiment[0]['label'], sentiment[0]['score'], transcription.text

    

    async def audio_processing(self):
        if time.time() - self.now_time > self.interval:  #get enough audio to classify
            self.now_time = time.time()
            if len(self.audio_buffer) > self.rate * self.interval:
                segment = self.audio_buffer[-self.rate * self.interval:]    #get last #RATE samples
                self.audio_buffer = []

                if np.mean(np.abs(segment)) < self.threshold:
                    print("You are too quiet, skipping...")
                else:
                    self.analysis_task = asyncio.create_task(self.background_AI_call(segment))
        return

    async def background_AI_call(self, segment):
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
        intensity = (abs(attitude) + hype) / 2
        self.upcoming_emo_vec = [hype, attitude, intensity]
    
    #turn emo vec into params for the animation
    def vec2param(self, emo_vec):
        hype, attitude, intensity = emo_vec
        hue = (attitude + 1) / 2  # 0 to 1
        color = hsv_to_rgb(hue, 1.0, min(1.0, 0.5 + intensity / 2))  # brighter with intensity
        color = tuple(int(c * 255) for c in color)  # convert to RGB 0–255
        speed = 0.01 + hype * 0.2
        wobble = 5 + hype * 20
        fractal_depth = int(2 + intensity * 2.5)
        trails = int(10 + intensity * 15)
        return color, wobble, speed, fractal_depth  #, trails
    

    def draw_spiral(self, t, color, wobble):
        for i in range(200):
            angle = i * 0.1 + t
            radius = 100 + np.sin(i * 0.3 + t) * wobble
            x = int(self.width/2 + np.cos(angle) * radius)
            y = int(self.height/2 + np.sin(angle) * radius)
            pygame.draw.circle(self.screen, color, (x, y), 3)


    def draw_fractal(self, x, y, angle, depth, color):
        if depth == 0:
            return
        length = 50 * depth
        x2 = x + int(np.cos(angle) * length)
        y2 = y + int(np.sin(angle) * length)
        pygame.draw.line(self.screen, color, (x, y), (x2, y2), 1)
        angle_offset = 0.3 + random.random() * 0.2
        self.draw_fractal(x2, y2, angle - angle_offset, depth - 1, color)
        self.draw_fractal(x2, y2, angle + angle_offset, depth - 1, color)


    def draw_lissajous(self, t, color):
        for i in range(300):
            x = int(self.width/2 + np.sin(3 * t + i * 0.02) * 200)
            y = int(self.height/2 + np.sin(4 * t + i * 0.03) * 200)
            pygame.draw.circle(self.screen, color, (x, y), 2)


    async def run(self):
        loop = asyncio.get_event_loop()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            data = self.stream.read(1024)
            # Process audio data here, this is the audio stream
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data)) #gather clips

            await self.audio_processing()
            if self.analysis_task and self.analysis_task.done():
                res = self.analysis_task.result()
                if res:
                    label1, label2, score1, score2 , transcript = res[0][0], res[1][0], res[0][1], res[1][1], res[1][2]
                    self.sentiment2vec(label1, label2, score1, score2)
                self.analysis_task = None

            print(self.emo_vec)
            alpha = 0.05
            self.ttime += 0.01
            self.emo_vec = [(1 - alpha) * cur + alpha * tgt for cur, tgt in zip(self.emo_vec, self.upcoming_emo_vec)]


            color, wobble, speed, depth = self.vec2param(self.emo_vec)
            self.ttime += speed

            # Vary with time & emotion
            self.draw_spiral(self.ttime, color, wobble)
            self.draw_lissajous(self.ttime, color)
            self.draw_fractal(self.width//2, self.height-20, -np.pi/2, depth, color)

            pygame.display.flip()
            await asyncio.sleep(0.001)

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
    asyncio.run(lsd.run())

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
     