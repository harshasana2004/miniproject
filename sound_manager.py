import numpy as np
import pygame

class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        # Generate a 440Hz beep
        sample_rate = 44100
        duration = 0.5  # seconds
        frequency = 440  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t) * 32767
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        audio_stereo = np.column_stack((audio, audio))
        self.beep_sound = pygame.sndarray.make_sound(audio_stereo)

    def play_beep(self):
        self.beep_sound.play()