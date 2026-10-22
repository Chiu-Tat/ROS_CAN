"""USB framing and preview transport tests using generated test JPEGs, not hardware."""
import base64
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from PIL import Image
from cs1_experiment.camera_stream import JsonFrameBuffer, MAX_PACKET_BYTES, decode_preview
from cs1_experiment.core import load_config


def test_jpeg():
    stream = io.BytesIO()
    Image.new('RGB',(32,24),(50,100,150)).save(stream,format='JPEG')
    return stream.getvalue()


class CameraStreamTests(unittest.TestCase):
    def test_fragmented_large_packet_and_following_pose(self):
        frame = {'sequence':1,'lines':[], 'jpeg_b64':base64.b64encode(test_jpeg()).decode(),
                 'padding':'x'*25000}
        following = {'sequence':2,'lines':[]}
        packet = (json.dumps(frame)+'\n'+json.dumps(following)+'\n').encode()
        reader = JsonFrameBuffer()
        frames = []
        for offset in range(0,len(packet),137):
            frames.extend(reader.feed(packet[offset:offset+137]))
        self.assertEqual(frames,[frame,following])
        self.assertEqual(reader.pending,b'')
        self.assertEqual(decode_preview(frames[0]),test_jpeg())

    def test_partial_start_or_truncated_video_preserves_following_pose(self):
        reader = JsonFrameBuffer(skip_invalid_lines=True)
        good = {'sequence':42, 'lines':[]}
        raw = json.dumps(good).encode()
        frames = reader.feed(b'fragment of previous packet\n'+raw+b'\n'+raw[:12])
        self.assertEqual(frames, [good])
        self.assertEqual(reader.rejected_lines, 1)
        self.assertEqual(reader.feed(raw[12:]+b'\n'), [good])
        self.assertEqual(reader.feed(b'{"jpeg_b64":"truncated\n'+raw+b'\n'), [good])
        self.assertEqual(reader.rejected_lines, 2)
        with self.assertRaises(ValueError):
            reader.feed(b'x'*(MAX_PACKET_BYTES+1))

    def test_invalid_and_oversized_packets(self):
        for data in [b'not json\n',b'[]\n',b'x'*(MAX_PACKET_BYTES+1)]:
            with self.assertRaises(ValueError):
                JsonFrameBuffer().feed(data)
        for encoded in ['bad-base64',base64.b64encode(b'not jpeg').decode(),42]:
            with self.assertRaises(ValueError):
                decode_preview({'jpeg_b64':encoded})
        self.assertIsNone(decode_preview({'sequence':1}))

    def tracker(self):
        import tempfile
        from tracking_fixture import tracker
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        value = tracker(folder.name)
        self.addCleanup(value.close)
        return value

    def test_image_published_before_heading_initialization_and_without_detection(self):
        import rospy
        tracker = self.tracker()
        frame = {'sequence':9,'lines':[],'jpeg_b64':base64.b64encode(test_jpeg()).decode()}
        with mock.patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(1)):
            tracker.process(frame)
        message = tracker.image_pub.publish.call_args[0][0]
        self.assertEqual(message.data,test_jpeg())
        self.assertEqual(message.header.frame_id,'openmv_camera')
        self.assertEqual(message.format,'jpeg')
        tracker.pub.publish.assert_not_called()
        self.assertIsNone(tracker.heading)

    def test_continuous_usb_arrival_preserves_partial_packet_and_tracking(self):
        import rospy
        tracker = self.tracker()
        first = {'sequence':1,'lines':[]}
        second = {'sequence':2,'lines':[]}
        raw = json.dumps(second).encode()+b'\n'
        serial = mock.Mock()
        serial.in_waiting = 1  # Camera continues producing bytes during each drain.
        serial.read.side_effect = [(json.dumps(first)+'\n').encode()+raw[:10],
                                   b'', b'', b'', raw[10:], b'', b'', b'']
        tracker.ser = serial
        tracker.process = mock.Mock()
        tracker.invalidate = mock.Mock()
        with mock.patch.object(rospy,'is_shutdown',side_effect=[False,False,True]):
            tracker.run()
        self.assertEqual([call.args[0] for call in tracker.process.call_args_list],
                         [first,second])
        serial.reset_input_buffer.assert_not_called()
        tracker.invalidate.assert_not_called()

    def test_bad_preview_does_not_clear_good_heading(self):
        import rospy,time
        tracker = self.tracker()
        tracker.heading = 0
        tracker.previous_xy = __import__('numpy').array([-.03,-.03])
        tracker.last_valid_time = time.monotonic()
        from tracking_fixture import frame as camera_frame
        frame = camera_frame(1, 0)
        frame['jpeg_b64'] = 'invalid'
        with mock.patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(1)), \
             mock.patch.object(rospy,'logwarn_throttle'):
            tracker.process(frame)
        self.assertEqual(tracker.heading,0)
        tracker.pub.publish.assert_called_once()
        tracker.image_pub.publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
