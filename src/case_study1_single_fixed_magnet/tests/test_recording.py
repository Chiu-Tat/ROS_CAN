"""Verify trial boundaries, deletion scope, recovery, and actual video playback."""
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cs1_experiment.recording import TrialStore, UiVideoRecorder, write_json


def controller_sample():
    return {'armed':True,'hardware':False,'input_source':'keyboard','drive':.5,'turn':0,
            'tracking_valid':True,'pose_fresh':True,'pose':[.01,.02,.6],'z_m':.082,
            'heading_unwrapped_rad':6.9,'reason':'drive','currents':[.2]*10,
            'current_feedback':{'values_A':[None]*10,'age_s':[None]*10,'fresh':[False]*10},
            'pose_hold':{'phase':'holding','reference_pose':[.011,.02,.6],
                         'pose_error':[.001,0,0]}}


class TrialRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.store=TrialStore(self.root)

    def tearDown(self):
        if self.store.worker is not None:
            self.store.stop('test cleanup');self.store.worker.join(timeout=3)
        self.temporary.cleanup()

    def finish(self):
        self.store.stop('operator zero / disarm');self.store.worker.join(timeout=3)
        self.assertEqual(self.store.snapshot()['phase'],'pending')

    def test_save_and_delete_are_scoped_to_one_armed_session(self):
        from PIL import Image
        metadata={'hardware':False,'input_source':'keyboard','initial_pose':{'heading':.6}}
        self.store.prepare(metadata,False);self.store.start()
        self.store.record('controller',controller_sample())
        buffer=io.BytesIO();Image.new('RGB',(20,20),'green').save(buffer,format='JPEG')
        self.store.record('/camera',{'sequence':1},jpeg=buffer.getvalue())
        self.store.record('/pose',{'x':.01,'y':.02})
        self.finish()
        saved=self.store.resolve(True,'First forward/backward trial')
        self.assertIn('First forward/backward trial',(saved/'README.md').read_text())
        self.assertEqual(json.loads((saved/'session.json').read_text())['phase'],'saved')
        rows=list(csv.DictReader(io.StringIO((saved/'telemetry.csv').read_text())))
        self.assertEqual(rows[0]['measured_01_A'],'')
        self.assertEqual(float(rows[0]['heading_unwrapped_rad']),6.9)
        self.assertEqual(float(rows[0]['command_10_A']),.2)
        self.assertEqual(rows[0]['hold_phase'],'holding')
        self.assertEqual(float(rows[0]['target_x_m']),.011)
        self.assertEqual(float(rows[0]['error_x_m']),.001)
        events=[json.loads(line) for line in (saved/'events.jsonl').read_text().splitlines()]
        self.assertEqual(len(events),3)
        self.assertTrue(all(event['elapsed_s']>=0 for event in events))
        self.assertTrue((saved/events[1]['payload']['file']).is_file())
        unrelated=self.root/'old_user_data.csv';unrelated.write_text('preserve')
        self.store.prepare(metadata,False);self.store.start();self.store.record('controller',controller_sample())
        pending=self.store.folder;self.finish();self.store.resolve(False)
        self.assertFalse(pending.exists());self.assertTrue(saved.exists())
        self.assertEqual(unrelated.read_text(),'preserve')

    def test_failed_preparation_leaves_no_trial_and_cannot_arm(self):
        with self.assertRaises(ValueError):self.store.prepare({'bad':float('nan')},False)
        self.assertEqual(self.store.snapshot()['phase'],'idle')
        self.assertEqual(list(self.store.pending.glob('trial_*')),[])

    def test_encoder_readiness_and_pending_review_gate_new_arming(self):
        prepared=self.store.prepare({},True)
        with self.assertRaisesRegex(ValueError,'not ready'):self.store.start()
        self.store.video_ready=True;self.store.start()
        self.store.stop('disarmed');self.store.worker.join(timeout=3)
        self.assertEqual(self.store.snapshot()['phase'],'stopping')
        with self.assertRaises(ValueError):self.store.prepare({},True)
        write_json(self.store.folder/'video_complete.json',{'frames':3})
        self.assertEqual(self.store.snapshot()['phase'],'pending')
        with self.assertRaises(ValueError):self.store.cancel_prepared()
        self.store.resolve(False)
        self.assertEqual(self.store.snapshot()['phase'],'idle')
        prepared=self.store.prepare({},True);folder=self.store.folder
        self.store.meta['phase']='cancelled';self.store.cancel_prepared()
        self.assertFalse(folder.exists())

    def test_no_overwrite_and_no_deleting_outside_pending_directory(self):
        self.store.prepare({},False);self.store.start();self.finish()
        folder=self.store.folder
        target=self.root/folder.name;target.mkdir()
        with self.assertRaises(FileExistsError):self.store.resolve(True)
        self.assertEqual(self.store.snapshot()['phase'],'pending')
        target.rmdir()
        with mock.patch('cs1_experiment.recording.os.rename',side_effect=OSError('injected failure')):
            with self.assertRaises(OSError):self.store.resolve(True)
        self.assertEqual(json.loads((folder/'session.json').read_text())['phase'],'pending')
        with self.assertRaises(ValueError):TrialStore.resolve_folder(self.root,self.root,False)
        outside=self.root/'outside';outside.mkdir()
        link=self.store.pending/'trial_symlink';link.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):TrialStore.resolve_folder(self.root,link,False)
        self.assertTrue(outside.exists())

    def test_interrupted_trial_is_recovered_without_automatic_deletion(self):
        self.store.prepare({},False)
        self.store.meta.update(owner_pid=99999999,phase='recording',start_ns=100,start_wall_ns=time.time_ns())
        write_json(self.store.folder/'session.json',self.store.meta)
        marker=self.store.folder/'partial_data.txt';marker.write_text('keep this')
        recovered=TrialStore(self.root)
        self.assertTrue(recovered.snapshot()['recovered'])
        self.assertEqual(recovered.snapshot()['phase'],'pending')
        destination=recovered.resolve(True,'Recovered after interruption')
        self.assertEqual((destination/'partial_data.txt').read_text(),'keep this')

    def test_real_mjpeg_video_has_timed_duplicates_and_decodes(self):
        from PyQt5 import QtGui
        folder=self.root/'video';folder.mkdir()
        video=UiVideoRecorder(folder,(150,100),fps=5)
        self.assertTrue(video.ready.wait(timeout=5));self.assertFalse(video.error)
        start=time.monotonic_ns();video.begin(start)
        red=QtGui.QImage(150,100,QtGui.QImage.Format_RGB32);red.fill(QtGui.QColor('red'))
        green=QtGui.QImage(200,100,QtGui.QImage.Format_RGB32);green.fill(QtGui.QColor('#00ff00'))
        video.submit(red,start);video.submit(green,start+400_000_000)
        video.stop(start+800_000_000)
        self.assertTrue(video.done.wait(timeout=5));self.assertFalse(video.error)
        import cv2
        capture=cv2.VideoCapture(str(folder/'ui.avi'))
        decoded=[]
        while True:
            ok,frame=capture.read()
            if not ok:break
            decoded.append(frame)
        capture.release()
        self.assertEqual(len(decoded),4)
        self.assertGreater(int(decoded[0][50,75,2]),200)
        self.assertGreater(int(decoded[2][50,75,1]),200)
        rows=list(csv.DictReader(io.StringIO((folder/'video_timestamps.csv').read_text())))
        self.assertEqual([row['duplicated'] for row in rows],['False','True','False','True'])
        self.assertEqual(json.loads((folder/'video_complete.json').read_text())['frames'],4)


if __name__=='__main__':unittest.main()
