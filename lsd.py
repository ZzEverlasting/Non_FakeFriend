import pyaudio, pygame
import numpy as np

class LSD:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            data = self.stream.read(1024)
            # Process audio data here
            audio_data = np.frombuffer(data, dtype=np.int16)

            volume = np.abs(audio_data).mean() / 32768.0

            bar_height = int(volume * self.screen.get_height())
            self.screen.fill((0, 0, 0))  # Clear screen
            pygame.draw.rect(self.screen, (0, 255, 0), 
                             (self.screen.get_width() // 2 - 50, 
                              self.screen.get_height() - bar_height, 
                              100, 
                              bar_height))
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