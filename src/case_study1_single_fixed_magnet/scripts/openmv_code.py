# Camera-side MicroPython. Upload as main.py with mpremote; do not run on the PC.
import sensor
import time
import json
import pyb
import ubinascii
import math

# LAB threshold for the physically GREEN coating, before any preview overlays.
# Tune under the experiment lighting; the host also checks the calibrated size.
# Keep dim green edge pixels; overly strict a-channel thresholds fragment the
# cylinder. Merge intersecting blob boxes, then let the host reject wrong sizes.
GREEN_THRESHOLD = (3, 95, -80, -10, -10, 90)
MAX_CANDIDATES = 64
DETECTOR = 'green_cylinder_v1'
ROI = (0, 0, 320, 240)
MIN_PIXELS = 20
MIN_AREA = 25
MAX_AREA = 3000
MAX_ROUNDNESS = 0.60
EXPOSURE_US = 10000
# Preview shares USB with pose packets. Lower rate/quality if tracking slows.
SEND_PREVIEW = True
PREVIEW_PERIOD_MS = 200
JPEG_QUALITY = 40
DRAW_DETECTIONS = True
MAX_PREVIEW_BYTES = 160 * 1024
USB_PACKET_TIMEOUT_MS = 150

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)
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
    for blob in img.find_blobs([GREEN_THRESHOLD], roi=ROI, pixels_threshold=MIN_PIXELS,
                               area_threshold=MIN_AREA, merge=True):
        if blob.area() <= MAX_AREA and blob.roundness() <= MAX_ROUNDNESS:
            corners = blob.min_corners()
            # A tight long edge is stable even when the blob's moment angle
            # occasionally degenerates to zero. Both are undirected axes.
            longest = max(range(4), key=lambda i:
                          (corners[(i+1)%4][0]-corners[i][0])**2
                          + (corners[(i+1)%4][1]-corners[i][1])**2)
            dx = corners[(longest+1)%4][0]-corners[longest][0]
            dy = corners[(longest+1)%4][1]-corners[longest][1]
            lines.append({'x':blob.cx(), 'y':blob.cy(),
                          'axis_rad':math.atan2(dy,dx)%math.pi,
                          'moment_axis_rad':blob.rotation(),
                          'pixels':blob.pixels(), 'length_px':math.sqrt(dx*dx+dy*dy),
                          'kind':DETECTOR, 'corners_px':corners,
                          'roundness':blob.roundness()})
    lines.sort(key=lambda item: item['pixels'], reverse=True)
    # Always send a frame, including an empty list on detection loss.
    payload = {'lines':lines[:MAX_CANDIDATES], 'fps':clock.fps(), 'sequence':sequence,
               'detector':DETECTOR, 'image_size_px':[img.width(),img.height()],
               'capture_ms':time.ticks_ms(), 'detection_overflow':len(lines)>MAX_CANDIDATES}
    if SEND_PREVIEW and time.ticks_diff(time.ticks_ms(),last_preview_ms) >= PREVIEW_PERIOD_MS:
        last_preview_ms = time.ticks_ms()
        try:
            # Detection is complete before overlays/compression modify this frame.
            if DRAW_DETECTIONS:
                for line in lines[:MAX_CANDIDATES]:
                    x,y,a = line['x'],line['y'],line['axis_rad']
                    dx,dy = math.cos(a)*line['length_px']/2,math.sin(a)*line['length_px']/2
                    img.draw_line(int(x-dx),int(y-dy),int(x+dx),int(y+dy),color=(255,170,0),thickness=1)
                    img.draw_cross(x,y,color=(255,170,0))
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
