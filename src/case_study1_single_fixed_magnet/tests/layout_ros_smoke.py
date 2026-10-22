#!/usr/bin/env python3
"""One layout's complete ROS config/model/recording path, simulation only."""
import argparse
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
    parser = argparse.ArgumentParser();parser.add_argument('layout',choices=['single','gate','triangle'])
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port = sock.getsockname()[1]
    os.environ['ROS_MASTER_URI']='http://127.0.0.1:%d'%port
    os.environ['ROS_IP']='127.0.0.1';os.environ.pop('ROS_HOSTNAME',None)
    with tempfile.TemporaryDirectory(prefix='cs1-'+args.layout+'-smoke-') as folder:
        output = tempfile.TemporaryFile(mode='w+')
        process = subprocess.Popen(['roslaunch','-p',str(port),'case_study1_single_fixed_magnet',
            'preview.launch','layout:='+args.layout,'gui:=false','gamepad:=false',
            'post_stop_record_s:=0','results_dir:='+folder],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        timer = None
        try:
            import rospy
            from std_msgs.msg import String
            from geometry_msgs.msg import Twist
            from std_srvs.srv import SetBool
            rospy.init_node('cs1_layout_smoke',disable_signals=True)
            rospy.wait_for_service('/cs1/arm',timeout=20)
            state = {}
            def receive(msg):state.clear();state.update(json.loads(msg.data))
            subscriber = rospy.Subscriber('/cs1/status',String,receive,queue_size=1)
            publisher = rospy.Publisher('/cs1/keyboard_command',Twist,queue_size=1)
            timer = rospy.Timer(rospy.Duration(.05),lambda _:publisher.publish(Twist()))
            def wait(predicate):
                end = time.monotonic()+15
                while time.monotonic()<end:
                    if process.poll() is not None:raise AssertionError('ROS exited')
                    if predicate():return
                    time.sleep(.03)
                raise AssertionError('Timed out: '+str(state.get('reason')))
            wait(lambda:state.get('pose_fresh') and state.get('layout')==args.layout)
            assert not state['hardware'] and not state['armed']
            controller_config = rospy.get_param('/cs1_controller/config')
            assert rospy.get_param('/cs1_preview_feedback/config')==controller_config
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
            from cs1_experiment.core import config_path,fixed_magnets
            assert Path(controller_config)==config_path(args.layout)
            response = rospy.ServiceProxy('/cs1/arm',SetBool)(True)
            assert response.success,response.message
            wait(lambda:state.get('armed'))
            rospy.ServiceProxy('/cs1/arm',SetBool)(False)
            wait(lambda:state.get('recording',{}).get('phase')=='pending')
            meta = state['recording']
            assert meta['layout']==args.layout
            assert meta['tfw_provenance']['protocol']['layout']==args.layout
            count = len(fixed_magnets(meta['configuration']['model']))
            assert count=={'single':1,'gate':2,'triangle':3}[args.layout]
            response = rospy.ServiceProxy('/cs1/recording/resolve',SetBool)(False)
            assert response.success,response.message
            print('PASS %s: %d fixed magnets; matching ROS config, TFW, simulated arm and trial metadata'%(args.layout,count))
        except Exception:
            output.seek(0);print(output.read()[-5000:]);raise
        finally:
            if timer is not None:timer.shutdown()
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)


if __name__=='__main__':main()
