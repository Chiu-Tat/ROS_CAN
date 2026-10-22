#!/usr/bin/env python3
"""Run an isolated ROS preview and test arming, steering, and command timeout.

Run with the workspace sourced. No camera, CAN interface, or gamepad is opened.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % port
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ.pop('ROS_HOSTNAME',None)
    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    capture = tempfile.TemporaryFile(mode='w+')
    recording_dir = tempfile.TemporaryDirectory(prefix='cs1-smoke-recording-')
    process = subprocess.Popen(['roslaunch','-p',str(port),'case_study1_single_fixed_magnet',
                                'preview.launch','gui:=false','gamepad:=false','input_source:=gamepad','post_stop_record_s:=0','results_dir:='+recording_dir.name],
                               stdout=capture,stderr=subprocess.STDOUT,start_new_session=True)
    timer = None
    keyboard_timer = None
    try:
        rospy.init_node('cs1_smoke',disable_signals=True)
        state = {}
        def receive(msg):
            state.clear();state.update(json.loads(msg.data))
        subscriber = rospy.Subscriber('/cs1/status',String,receive,queue_size=1)
        pub = rospy.Publisher('/cs1/gamepad_command',Twist,queue_size=1)
        command = Twist()
        timer = rospy.Timer(rospy.Duration(.05),lambda _:pub.publish(command))
        def wait_for(predicate, timeout=15):
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError('Preview launch exited unexpectedly')
                if predicate():
                    return
                time.sleep(.05)
            raise AssertionError('Preview wait timed out: '+str(state))
        rospy.wait_for_service('/cs1/arm',timeout=20)
        arm = rospy.ServiceProxy('/cs1/arm',SetBool)
        def discard_pending():
            wait_for(lambda:state.get('recording',{}).get('phase')=='pending')
            response=rospy.ServiceProxy('/cs1/recording/resolve',SetBool)(False)
            assert response.success,response.message
            wait_for(lambda:state.get('recording',{}).get('phase')=='idle')
        wait_for(lambda:state.get('pose_fresh'))
        time.sleep(.3)
        response = arm(True)
        assert response.success,response.message
        wait_for(lambda:state.get('armed'))
        wait_for(lambda:state.get('pose_hold',{}).get('phase')=='holding',timeout=35)
        assert not state['hardware']
        start = list(state['pose'])
        command.linear.x,command.angular.z = .5,.2
        time.sleep(3.0)
        assert state['armed'],state
        pose = state['pose']
        assert abs(pose[0]-start[0])+abs(pose[1]-start[1]) > .0001,(start,pose,state)
        assert abs(pose[2]-start[2]) > .001,(start,pose,state)
        print('PASS preview arming and gamepad-shaped drive/turn commands')
        command.linear.x,command.angular.z = 0,0
        time.sleep(.2)
        keyboard_pub = rospy.Publisher('/cs1/keyboard_command',Twist,queue_size=1)
        keyboard_command = Twist()
        keyboard_timer = rospy.Timer(rospy.Duration(.05),lambda _:keyboard_pub.publish(keyboard_command))
        rospy.set_param('/cs1/input_source','keyboard')
        wait_for(lambda:state.get('input_source')=='keyboard' and not state.get('armed'))
        discard_pending()
        time.sleep(.3)
        response = arm(True)
        assert response.success,response.message
        wait_for(lambda:state.get('armed'))
        wait_for(lambda:state.get('pose_hold',{}).get('phase')=='holding',timeout=35)
        start = list(state['pose'])
        # The inactive gamepad must neither override movement nor keep keyboard input alive.
        command.linear.x,command.angular.z = -1,-1
        keyboard_command.linear.x,keyboard_command.angular.z = .5,.2
        time.sleep(4.0)  # Pose-reference control includes a 1 mm camera-noise deadband.
        pose = state['pose']
        import math
        along = (pose[0]-start[0])*math.cos(start[2])+(pose[1]-start[1])*math.sin(start[2])
        assert state['armed'] and along > .0001 and pose[2]-start[2] > .001,state
        print('PASS input switch disarms; keyboard drives while inactive gamepad is ignored')
        keyboard_timer.shutdown();keyboard_timer=None
        wait_for(lambda:not state.get('armed'),timeout=3)
        assert 'heartbeat' in state['reason'],state
        assert all(i==0 for i in state['currents']),state
        print('PASS command timeout disarms and publishes zero currents')
        discard_pending()
        from std_srvs.srv import Trigger
        feedback = json.loads(rospy.wait_for_message('/cs1/current_feedback',String,timeout=3).data)
        assert not feedback['available'] and feedback['values_A']==[None]*10,feedback
        assert rospy.ServiceProxy('/cs1/zero_currents',Trigger)().success
        for action in ('reset_drivers','prepare_drivers','enable_current_feedback'):
            response = rospy.ServiceProxy('/cs1/'+action,Trigger)()
            assert not response.success and 'CAN disabled' in response.message,response
        print('PASS driver services: preview zero/disarm; hardware actions refused; missing currents stay unavailable')
        # Test the new ROS-image-to-Qt path with a clearly synthetic JPEG.
        # This is transport/render verification, not a physical camera test.
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        import io,sys
        from PIL import Image
        from PyQt5 import QtWidgets
        from sensor_msgs.msg import CompressedImage
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from control_panel import ControlPanel
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        panel = ControlPanel(connect_ros=True)
        panel.timer.stop();panel.command_timer.stop()
        try:
            image_pub = rospy.Publisher('/cs1/camera/image/compressed',CompressedImage,queue_size=1)
            wait_for(lambda:image_pub.get_num_connections()>0)
            stream = io.BytesIO()
            Image.new('RGB',(40,30),(10,120,60)).save(stream,format='JPEG')
            message = CompressedImage(format='jpeg',data=stream.getvalue())
            message.header.seq = 123
            image_pub.publish(message)
            wait_for(lambda:panel.latest_image is not None)
            panel.refresh();app.processEvents()
            assert panel.camera_view.original.width()==40
            assert 'live' in panel.camera_status.text()
            assert 'SIMULATED feedback' in panel.pose_status.text()
            print('PASS synthetic JPEG ROS transport, Qt rendering, and feedback-driven world view')
        finally:
            panel.close()
    except Exception:
        capture.seek(0)
        print(capture.read())
        raise
    finally:
        if timer is not None:
            timer.shutdown()
        if keyboard_timer is not None:
            keyboard_timer.shutdown()
        rospy.signal_shutdown('Smoke complete')
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        capture.close()
        recording_dir.cleanup()


if __name__ == '__main__':
    main()
