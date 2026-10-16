# gesture-cam

hand gesture effects for video calls. works with facetime, zoom, discord, google meet, anything that lets you pick a camera

## setup

python 3.12

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python gesture_cam.py
```

keybinds: **q** quit, **d** debug view, **c** clear the drawing, **p** turn the air pen on/off

## using it in facetime

it sends the video through OBS's virtual camera

1. install OBS from obsproject.com
2. open OBS, click **Start Virtual Camera**, then stop it and quit OBS. you only have to do this once, it installs the virtual camera
3. run `python gesture_cam.py`
4. in facetime: **Video** menu at the top > pick **OBS Virtual Camera**

### making the call hear the sounds

the virtual camera is video only. so gesture cam takes your mic, adds the sounds on top, and sends that into BlackHole. then the call uses BlackHole as your mic and hears your voice + the sounds

1. install BlackHole 2ch: `brew install --cask blackhole-2ch`, then restart your mac
2. run `python gesture_cam.py`. it should say "sending your mic (...) + sounds to BlackHole 2ch" (the first time, mac asks to let Terminal use the mic)
3. in the call, pick **BlackHole 2ch** as your microphone
4. turn off the noise filtering or it might cut the sounds out. facetime: Control Center > **Mic Mode** > Standard

## license

MIT, see [LICENSE](LICENSE)
