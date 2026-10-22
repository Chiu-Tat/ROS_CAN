# Camera-side MicroPython. Upload as main.py with mpremote; do not run on the PC.
import sensor
import time
import json
import pyb
import ubinascii
import math

# Tune for the cylinder silhouette and lighting. Exclude unrelated dark objects
# with ROI. The host excludes the fixed magnet in calibrated world coordinates.
DARK_THRESHOLD = (0, 65)
ROI = (0, 0, 320, 240)
MIN_PIXELS = 20
MIN_AREA = 25
MAX_AREA = 3000
MAX_ROUNDNESS = 0.65
EXPOSURE_US = 10000
# Preview shares USB with pose packets. Lower rate/quality if tracking slows.
SEND_PREVIEW = True
PREVIEW_PERIOD_MS = 200
JPEG_QUALITY = 40
DRAW_DETECTIONS = True
MAX_PREVIEW_BYTES = 160 * 1024
USB_PACKET_TIMEOUT_MS = 150

sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_exposure(False, exposure_us=EXPOSURE_US)
usb = pyb.USB_VCP()
clock = time.clock()
sequence = 0
last_preview_ms = time.ticks_ms()


def send_packet(payload):
    packet = memoryview((json.dumps(payload) + '\n').encode())
    offset = 0
    started = time.ticks_ms()
    while offset < len(packet):
        count = usb.send(packet[offset:],timeout=20)
        offset += count or 0
        if time.ticks_diff(time.ticks_ms(),started) > USB_PACKET_TIMEOUT_MS:
            # Terminate a partial line so the host can resynchronize.
            usb.send(b'\n',timeout=5)
            return


while True:
    clock.tick()
    img = sensor.snapshot()
    lines = []
    for blob in img.find_blobs([DARK_THRESHOLD], roi=ROI, pixels_threshold=MIN_PIXELS,
                               area_threshold=MIN_AREA, merge=False):
        if blob.area() <= MAX_AREA and blob.roundness() <= MAX_ROUNDNESS:
            lines.append({'x':blob.cx(), 'y':blob.cy(), 'axis_rad':blob.rotation(),
                          'pixels':blob.pixels(), 'length_px':max(blob.w(),blob.h())})
    # Always send a frame, including an empty list on detection loss.
    payload = {'lines':lines[:12], 'fps':clock.fps(), 'sequence':sequence}
    if SEND_PREVIEW and time.ticks_diff(time.ticks_ms(),last_preview_ms) >= PREVIEW_PERIOD_MS:
        last_preview_ms = time.ticks_ms()
        try:
            # Detection is complete before overlays/compression modify this frame.
            if DRAW_DETECTIONS:
                for line in lines[:12]:
                    x,y,a = line['x'],line['y'],line['axis_rad']
                    dx,dy = math.cos(a)*line['length_px']/2,math.sin(a)*line['length_px']/2
                    img.draw_line(int(x-dx),int(y-dy),int(x+dx),int(y+dy),color=255,thickness=1)
                    img.draw_cross(x,y,color=255)
            encoded = img.compress(quality=JPEG_QUALITY).bytearray()
            if len(encoded) <= MAX_PREVIEW_BYTES:
                payload['jpeg_b64'] = ubinascii.b2a_base64(encoded).decode().strip()
            else:
                payload['preview_error'] = 'JPEG exceeds host preview limit'
        except Exception as exc:
            # An optional preview failure must not suppress the measured pose packet.
            payload['preview_error'] = str(exc)
    try:
        send_packet(payload)
    except OSError:
        pass
    sequence += 1
    time.sleep_ms(5)
