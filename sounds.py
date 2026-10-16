# sounds.py
# plays sound effects when gestures go off
# uses sounddevice, which already gets installed with mediapipe
# wav files work directly. mp3/m4a get converted to wav with afconvert (comes with every mac)
#
# to get the sounds into a call, start_call_mix() takes your mic, adds the sounds on top, and sends
# that into BlackHole. then you pick BlackHole as the mic in facetime/discord

import os
import subprocess
import wave

import numpy as np

loaded = {}
device = None
CALL_RATE = 48000
call = {"stream": None, "playing": []}  # the mic + sounds mix going into BlackHole


def convert_to_wav(path):
    new_path = os.path.splitext(path)[0] + ".wav"
    if os.path.exists(new_path):
        return new_path
    print("converting", path, "to wav...")
    try:
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", path, new_path], check=True)
        return new_path
    except Exception as e:
        print("couldnt convert it, try making it a wav yourself:", e)
        return None


def read_wav(path):
    with wave.open(path) as f:
        if f.getsampwidth() != 2:
            print(path, "has to be a 16 bit wav")
            return None
        rate = f.getframerate()
        channels = f.getnchannels()
        raw = f.readframes(f.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    return samples.reshape(-1, channels), rate


def load_sounds(sound_files, output_device=None, folder="sounds"):
    global device
    device = output_device
    loaded.clear()
    for name, file in sound_files.items():
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            print("cant find sound", path)
            continue
        if not path.lower().endswith(".wav"):
            path = convert_to_wav(path)
            if path is None:
                continue
        sound = read_wav(path)
        if sound is not None:
            loaded[name] = sound
    if loaded:
        print("loaded sounds for:", list(loaded.keys()))


def for_call(data, rate):
    # the call mix runs at CALL_RATE in stereo, so stretch the sound to match
    if rate != CALL_RATE:
        n = int(len(data) * CALL_RATE / rate)
        old = np.linspace(0, 1, len(data))
        new = np.linspace(0, 1, n)
        data = np.stack([np.interp(new, old, data[:, c]) for c in range(data.shape[1])], 1)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2].astype(np.float32)


def start_call_mix(mic=None, out="BlackHole 2ch"):
    # mic = None uses your normal mic. returns True if it worked
    # two separate streams (mic in, BlackHole out) with a little buffer between them, since
    # macos doesnt like one stream that uses two different devices
    import queue
    import sounddevice as sd

    buffered = queue.Queue(maxsize=20)

    def mic_callback(indata, frames, time, status):
        try:
            buffered.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # call side fell behind, drop a bit of mic instead of lagging more and more

    leftover = {"data": np.zeros(0, np.float32)}

    def out_callback(outdata, frames, time, status):
        # take mic audio from the buffer (silence if there isnt enough yet)
        voice = leftover["data"]
        while len(voice) < frames:
            try:
                voice = np.concatenate([voice, buffered.get_nowait()])
            except queue.Empty:
                break
        part = np.zeros(frames, np.float32)
        part[:min(frames, len(voice))] = voice[:frames]
        leftover["data"] = voice[frames:]
        mix = np.repeat(part[:, None], 2, axis=1)
        # add the sounds on top
        playing = call["playing"]
        for item in playing:
            chunk = item["data"][item["pos"]:item["pos"] + frames]
            mix[:len(chunk)] += chunk
            item["pos"] += frames
        # (filtering instead of .remove() since comparing dicts with arrays in them crashes)
        call["playing"] = [item for item in playing if item["pos"] < len(item["data"])]
        np.clip(mix, -1, 1, out=outdata)

    try:
        mic_stream = sd.InputStream(device=mic, samplerate=CALL_RATE, channels=1, dtype="float32",
                                    blocksize=512, callback=mic_callback)
        out_stream = sd.OutputStream(device=out, samplerate=CALL_RATE, channels=2, dtype="float32",
                                     blocksize=512, callback=out_callback)
        mic_stream.start()
        out_stream.start()
    except Exception as e:
        print("couldnt start sending sounds to the call:", e)
        return False
    call["stream"] = (mic_stream, out_stream)
    mic_name = sd.query_devices(mic_stream.device)["name"]
    print("sending your mic (" + mic_name + ") + sounds to", out, "- pick", out, "as the mic in the call")
    return True


def stop_call_mix():
    if call["stream"] is not None:
        for stream in call["stream"]:
            stream.stop()
            stream.close()
        call["stream"] = None


def play(name, volume=1.0):
    if name not in loaded:
        return
    data, rate = loaded[name]
    try:
        import sounddevice as sd
        # doesnt wait for it to finish. a new sound cuts off the old one (like a soundboard)
        sd.play(data * volume, rate, device=device)
    except Exception as e:
        print("couldnt play sound:", e)
    if call["stream"] is not None:
        call["playing"] = [{"data": for_call(data, rate) * volume, "pos": 0}]
