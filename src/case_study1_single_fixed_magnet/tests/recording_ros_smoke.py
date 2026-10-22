#!/usr/bin/env python3
"""Full Arm/video/Save/Delete/close workflow, isolated ROS + synthetic feedback only."""
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    os.environ['ROS_MASTER_URI']='http://127.0.0.1:%d'%port
    os.environ['ROS_IP']='127.0.0.1';os.environ.pop('ROS_HOSTNAME',None)
    os.environ['QT_QPA_PLATFORM']='offscreen'
    directory=tempfile.TemporaryDirectory(prefix='cs1-recording-flow-')
    output=tempfile.TemporaryFile(mode='w+')
    process=subprocess.Popen(['roslaunch','-p',str(port),'case_study1_single_fixed_magnet','preview.launch',
        'gui:=false','record_ui:=true','gamepad:=false','input_source:=keyboard','post_stop_record_s:=3',
        'results_dir:='+directory.name],
        stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    panel=timer=None
    try:
        import rospy
        from PyQt5 import QtWidgets,QtTest
        from sensor_msgs.msg import CompressedImage
        from PIL import Image,ImageDraw
        from std_srvs.srv import SetBool
        rospy.init_node('cs1_recording_flow',disable_signals=True)
        rospy.wait_for_service('/cs1/arm',timeout=20)
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from control_panel import ControlPanel
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel=ControlPanel(connect_ros=True);panel.show();panel.activateWindow();panel.canvas.setFocus()
        def wait_for(predicate,timeout=15):
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if process.poll() is not None:raise AssertionError('ROS preview exited')
                QtTest.QTest.qWait(25)
                if predicate():return
            raise AssertionError('Timed out: '+str(panel.state)+' recorder '+str(panel.recorder.state))
        image=Image.new('RGB',(320,240),(25,80,45));ImageDraw.Draw(image).text((35,100),'SYNTHETIC CAMERA - NO HARDWARE',fill='white')
        buffer=io.BytesIO();image.save(buffer,format='JPEG')
        pub=rospy.Publisher('/cs1/camera/image/compressed',CompressedImage,queue_size=1)
        timer=rospy.Timer(rospy.Duration(.2),lambda _:pub.publish(CompressedImage(format='jpeg',data=buffer.getvalue())))
        wait_for(lambda:panel.state.get('pose_fresh') and panel.command_pub.get_num_connections()>0)
        for keep in (True,False):
            panel.set_arm(True)
            wait_for(lambda:panel.state.get('armed'))
            identifier=panel.recorder.state['id'];pending=Path(panel.recorder.state['path'])
            assert pending.parent.name=='.pending'
            QtTest.QTest.qWait(1200)
            panel.driver_action('zero_currents')
            # Recording continues after control stops (to capture e.g. the
            # free magnet snapping onto the fixed one).
            wait_for(lambda:not panel.state.get('armed') and panel.recorder.in_post_stop(),timeout=5)
            assert 'Finishing recording' in panel.recorder.label.text(),panel.recorder.label.text()
            assert panel.recorder.stop_now_button.isVisible()
            stopped_at=time.monotonic()
            if not keep:  # end the post-stop recording early
                QtTest.QTest.mouseClick(panel.recorder.stop_now_button,__import__('PyQt5.QtCore',fromlist=['Qt']).Qt.LeftButton)
            wait_for(lambda:panel.recorder.state['phase']=='pending' and panel.recorder.dialog is not None)
            if keep:assert time.monotonic()-stopped_at>2,'post-stop recording ended too early'
            else:assert time.monotonic()-stopped_at<2.5,'Stop recording now did not end it early'
            session=json.loads((pending/'session.json').read_text())
            assert session['control_stop_reason'].startswith('Driver action') and session['stop_reason']==session['control_stop_reason'],session
            assert not panel.state.get('armed')
            video=json.loads((pending/'video_complete.json').read_text())
            assert video['frames']>=2 and not video['error'],video
            assert (pending/'events.jsonl').stat().st_size>0
            assert list((pending/'camera').glob('*.jpg'))
            dialog=panel.recorder.dialog
            if keep:dialog.findChild(QtWidgets.QPlainTextEdit).setPlainText('Synthetic keyboard holding-field recording check')
            button=next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text().startswith('Yes' if keep else 'No'))
            QtTest.QTest.mouseClick(button,__import__('PyQt5.QtCore',fromlist=['Qt']).Qt.LeftButton)
            wait_for(lambda:panel.recorder.state['phase']=='idle' and panel.recorder.dialog is None)
            assert not pending.exists()
            saved=Path(directory.name)/identifier
            if keep:
                assert saved.is_dir() and (saved/'README.md').is_file()
                assert 'Synthetic keyboard' in (saved/'README.md').read_text()
                rows=list(csv.DictReader(io.StringIO((saved/'telemetry.csv').read_text())))
                assert rows and all(row['measured_01_A']=='' for row in rows)
                assert any(row['heading_unwrapped_rad'] for row in rows)
                assert json.loads((saved/'session.json').read_text())['hardware'] is False
                import cv2
                movie=cv2.VideoCapture(str(saved/'ui.avi'));ok,_=movie.read();movie.release()
                assert ok,'Saved UI video cannot be decoded'
                preserved=saved
                print('PASS Arm records full UI + timed telemetry + camera data; Yes saves folder and introduction')
            else:
                assert not saved.exists() and preserved.exists()
                print('PASS No deletes only the second trial, preserving the previously saved trial')
        # Closing the GUI while armed must stop output and still request a decision.
        panel.set_arm(True);wait_for(lambda:panel.state.get('armed'))
        QtTest.QTest.qWait(500);panel.close()
        wait_for(lambda:panel.recorder.dialog is not None and panel.recorder.state['phase']=='pending')
        assert panel.isVisible() and not panel.state.get('armed')
        panel.recorder.resolve(False)
        wait_for(lambda:not panel.isVisible())
        print('PASS window close disarms, finalizes, and requires Save/Delete before closing')
    except Exception:
        output.seek(0);print(output.read());raise
    finally:
        if timer is not None:timer.shutdown()
        if panel is not None:
            panel.recorder.state={'phase':'idle'};panel.recorder.auto_review=False
            if panel.recorder.dialog is not None:panel.recorder.dialog.reject()
            panel.close()
        if 'rospy' in sys.modules:sys.modules['rospy'].signal_shutdown('Recording smoke complete')
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        output.close();directory.cleanup()


if __name__=='__main__':main()
