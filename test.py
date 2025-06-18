import os
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer

# Load the API key from the environment

api_key = "f9e23b3ee234b4248ce53dde280a725594080964b4fec676562c1829fe91dd1c.091f2125-f7c9-4650-ba3d-55a54104cb67"
client = Neuphonic(api_key=api_key)

sse = client.tts.SSEClient()

# TTSConfig is a pydantic model so check out the source code for all valid options
tts_config = TTSConfig(
    speed=1.05,
    lang_code='en', # replace the lang_code with the desired language code.
    voice_id='<VOICE_ID>'  # use client.voices.list() to view all available voices
)

# Create an audio player with `pyaudio`
with AudioPlayer() as player:
    response = sse.send('Hello, world!', tts_config=tts_config)
    player.play(response)

    player.save_audio('output.wav')  # save the audio to a .wav file
