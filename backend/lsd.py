import pyaudio, pygame
import numpy as np
import time, os
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
        self.frame = 1024
        self.width, self.height = 640, 480  # window size for pygame
        self.hue_shift = 0.0

        #tweakables
        self.rate = 16000   #good sample rate for whisper but idk how good it actualy is
        self.threshold = 0    # volume threshold for audio classification lower = more sensitive
        self.interval = 4         # tweak this interval (in sec) to get more or less audio for classification

        #fit for input stream
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.frame)

        
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
        audio_sentiment_map = {"ang": 0.9, "hap": 0.6, "sad": 0.1, "neu": 0.3}      # basic mapping of audio sentiment to numbers. Not that accurate
        text_sentiment_map = {"positive": 1, "neutral": 0.2, "negative": -1}
        hype = audio_sentiment_map[label_audio] * score_audio
        attitude = text_sentiment_map[label_text]  * score_text
        intensity = (abs(attitude) + hype) / 2
        self.upcoming_emo_vec = [hype, attitude, intensity]


    ## design specs:
    ## use hsv to convey emotion
    ## hue: red = more hype blue = less hype? but color should be 
    ## saturation: more attitude = more color, less attitude = more gray
    ## value: more intensity = more brightness, less intensity = more dark
    ## speed: more hype and intensity = faster
    ## size: more intensity = bigger circles, less intensity = smaller circles
    
    #turn emo vec into params for the animation
    def vec2param(self, emo_vec):
        hype, attitude, intensity = emo_vec
        hue = 1 - hype  # 0 to 1 clamp?
        saturation = max(0.01, attitude + 0.01)  # avoid zero
        value = min(1.0, 0.5 + intensity / 2)  # brightness
        color = hsv_to_rgb(hue, saturation, value)  # brighter with intensity
        color = tuple(int(c * 255) for c in color)  # convert to RGB 0–255
        speed = 0.01 + 0.15*intensity+ 0.15*hype
        size = 5 + intensity * 10
        return color, speed, size
    

    def draw_spinner(self, t, emo_vec, color, spacing, base_size, hue_shift, petals, layers):
        hype, attitude, intensity = emo_vec
        center_x = self.width // 2
        center_y = self.height // 2
        rmax = max(self.width, self.height)

        wobble_speed = 0.2 + hype * 0.2 + intensity * 0.2
        wobble_size = 5 + intensity * 12
        petal_count = int(petals + hype * 5)
        layer_depth = int(layers + intensity * 2)

        for layer in range(layer_depth):
            radius = (layer + spacing) * (rmax // (layer_depth + spacing))
            for i in range(petal_count):
                angle = (2 * np.pi / petal_count) * i + t * wobble_speed
                x = int(center_x + np.cos(angle) * radius)
                y = int(center_y + np.sin(angle) * radius)

                size = int(base_size + np.sin(t + i + layer) * wobble_size)
                color = self.cook_color(self.emo_vec, hue_shift=hue_shift)

                pygame.draw.circle(self.screen, color, (x, y), size)
    
    def cook_color(self, emo_vec, hue_shift=0):
        hype, attitude, intensity = emo_vec
        base_hue = (1 - hype) * 0.66
        hue = (base_hue + hue_shift) % 1.0

        # Attitude = saturation (less attitude = grayer)
        saturation = 0.5 + (attitude * 0.3)  # avoid zero
        # g = r * saturation
        # b = r * saturation

        # Intensity = value (more intensity = brighter)
        value = 0.4 + (1-intensity) * 0.4
        # r = (1 - hue_shift) * value % 1.0
        # g = (0.5* hue_shift) * value % 1.0
        # b = hue_shift * value % 1.0

        r, g, b = hsv_to_rgb(hue, saturation, value)
        return int(r * 255), int(g * 255), int(b * 255)


    # main loop
    async def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            data = self.stream.read(self.frame)
            # Process audio data here, this is the audio stream
            audio_data = np.frombuffer(data, dtype=np.int16)
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data)) #gather clips

            await self.audio_processing()
            if self.analysis_task and self.analysis_task.done():
                res = self.analysis_task.result()
                if res:
                    label1, label2, score1, score2 , transcript = res[0][0], res[1][0], res[0][1], res[1][1], res[1][2]
                    self.sentiment2vec(label1, label2, score1, score2)
                    print(f"emo_vector: {self.upcoming_emo_vec}")
                    print(f"Transcript: {transcript}")
                self.analysis_task = None

            #lerp
            alpha = 0.05
            self.emo_vec = [(1 - alpha) * current + alpha * target for current, target in zip(self.emo_vec, self.upcoming_emo_vec)]

            #this is actually kind trippy
            self.ttime += 0.01  #timer
            color, speed, size = self.vec2param(self.emo_vec)
            self.ttime += speed
            self.hue_shift = self.hue_shift + 0.05
            for i in range(3):
                self.hue_shift = i * 0.2
                self.draw_spinner(self.ttime + i * 0.5, self.emo_vec, color, 0.8, size, self.hue_shift + 5*(i-1), petals=16, layers=5)
                self.hue_shift = i * 0.4
                self.draw_spinner(self.ttime + i * 0.8, self.emo_vec, color, 0.5, size, self.hue_shift + 5*(i-1), petals=14, layers=4)
                self.hue_shift = i * 0.7
                self.draw_spinner(self.ttime + i * 0.2, self.emo_vec, color, 0.2, size, self.hue_shift + 5*(i-1), petals=12, layers=3)

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

    #test groq client in static mode
    # client = Groq(api_key="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")
    # with open("Recording.wav", "rb") as file:
    #         transcription = client.audio.transcriptions.create(
    #             file=file,
    #             model="whisper-large-v3-turbo",
    #             language='en',
    #             timestamp_granularities = ["word", "segment"],
    #             response_format="verbose_json",)
    # print(transcription.text)
