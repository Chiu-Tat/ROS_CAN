"""Bounded newline framing for pose/JPEG packets on the existing USB link."""
import base64
import binascii
import json


MAX_PACKET_BYTES = 256 * 1024
MAX_JPEG_BYTES = 160 * 1024


class JsonFrameBuffer:
    def __init__(self, skip_invalid_lines=False):
        self.skip_invalid_lines = skip_invalid_lines
        self.rejected_lines = 0
        self.pending = b''

    def clear(self):
        self.pending = b''

    def feed(self, chunk):
        data = self.pending + chunk
        self.pending = b''
        pieces = data.split(b'\n')
        self.pending = pieces.pop()
        if len(self.pending) > MAX_PACKET_BYTES or any(len(part) > MAX_PACKET_BYTES for part in pieces):
            self.clear()
            raise ValueError('Camera USB packet exceeds size limit')
        frames = []
        for piece in pieces:
            if piece.strip():
                try:
                    frame = json.loads(piece.decode('utf-8'))
                    if not isinstance(frame, dict):
                        raise ValueError('Expected camera JSON object')
                except (ValueError, UnicodeError):
                    if not self.skip_invalid_lines:
                        raise
                    # A connection can begin halfway through a camera packet.
                    # Preserve pending bytes of the NEXT line and any valid
                    # frames in this batch; dropping them causes repeated loss.
                    self.rejected_lines += 1
                    continue
                frames.append(frame)
        return frames


def decode_preview(frame):
    encoded = frame.get('jpeg_b64')
    if encoded is None:
        return None
    if not isinstance(encoded,str) or len(encoded) > MAX_JPEG_BYTES*4//3+4:
        raise ValueError('Invalid/oversized camera JPEG encoding')
    try:
        jpeg = base64.b64decode(encoded,validate=True)
    except (binascii.Error,ValueError) as exc:
        raise ValueError('Malformed camera JPEG encoding') from exc
    if len(jpeg) > MAX_JPEG_BYTES or not (jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')):
        raise ValueError('Invalid camera JPEG data')
    return jpeg
