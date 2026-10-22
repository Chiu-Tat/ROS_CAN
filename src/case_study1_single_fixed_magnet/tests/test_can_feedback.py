"""Exercise the actual driver against an in-memory CAN peer; no hardware opened."""
import math
from pathlib import Path
import sys
import threading
import time
import unittest
import uuid
from unittest import mock

import can
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cs1_experiment.can_driver import CurrentOutput, DEVICE_IDS, FEEDBACK_IDS


class CurrentFeedbackTests(unittest.TestCase):
    def setUp(self):
        channel = 'cs1-test-'+uuid.uuid4().hex
        self.peer = can.interface.Bus(channel=channel,bustype='virtual')
        bus = can.interface.Bus(channel=channel,bustype='virtual')
        with mock.patch.object(can.interface,'Bus',return_value=bus):
            self.driver = CurrentOutput(True,'test-only')

    def tearDown(self):
        self.driver.close(); self.peer.shutdown()

    def drain(self):
        messages = []
        while True:
            value = self.peer.recv(timeout=.01)
            if value is None: return messages
            messages.append(value)

    def test_all_ten_signed_currents_and_per_channel_staleness(self):
        expected = [-15,-12.345,-1,-.001,0,.001,1,6.789,12.345,15]
        self.assertEqual(self.driver.feedback_snapshot()['values_A'],[None]*10)
        for identifier, amps in zip(FEEDBACK_IDS,expected):
            raw = int(round(amps*1000)) & 0xffff
            self.peer.send(can.Message(arbitration_id=identifier,data=[raw>>8,raw&255]+[0]*6,is_extended_id=False))
        end = time.monotonic()+1
        while time.monotonic()<end:
            snapshot = self.driver.feedback_snapshot()
            if all(snapshot['fresh']):break
            self.driver.stop_receiver.wait(.01)
        np.testing.assert_allclose(snapshot['values_A'], expected)
        self.assertTrue(all(snapshot['fresh']))
        stale = self.driver.feedback_snapshot(now=time.monotonic()+2)
        self.assertFalse(any(stale['fresh']))
        np.testing.assert_allclose(stale['values_A'],expected)
        for message in [can.Message(arbitration_id=FEEDBACK_IDS[0],data=[1],is_extended_id=False),
                        can.Message(arbitration_id=FEEDBACK_IDS[0],data=[0,1],is_extended_id=True),
                        can.Message(arbitration_id=0x321,data=[0,1],is_extended_id=False)]:
            self.assertFalse(self.driver.accept_feedback(message))

    def test_zero_before_arm_and_reset_protocol(self):
        self.assertFalse(self.driver.initialized)
        self.driver.zero_all()
        messages = self.drain()
        self.assertEqual([m.arbitration_id for m in messages],[0x003]+DEVICE_IDS)
        self.assertTrue(all(list(m.data[:4])==[0x13,0x88,0,0] for m in messages))
        self.driver.accept_feedback(can.Message(arbitration_id=FEEDBACK_IDS[0],data=[0,123],is_extended_id=False))
        self.driver.reset()
        messages = self.drain()
        self.assertEqual(messages[-1].arbitration_id,0x000)
        self.assertEqual(list(messages[-1].data),[0x55]*8)
        self.assertFalse(self.driver.initialized)
        self.assertEqual(self.driver.feedback_snapshot()['values_A'],[None]*10)

    def test_prepare_and_feedback_use_existing_driver_commands(self):
        self.driver.initialize()
        messages = self.drain()
        self.assertEqual([m.arbitration_id for m in messages],[0x001,0x003])
        self.assertEqual(list(messages[0].data),[0x02]+[0x55]*7)
        self.assertEqual(list(messages[1].data),[0x13,0x88,0,0]+[0x55]*4)
        self.assertTrue(self.driver.initialized)
        self.assertFalse(self.driver.feedback_enabled)
        self.driver.enable_feedback()
        messages = self.drain()
        self.assertEqual([m.arbitration_id for m in messages],[0x00A])
        self.assertEqual(list(messages[-1].data),[0x0A,0]+[0x55]*6)
        self.assertTrue(self.driver.feedback_enabled)
        values = np.linspace(-15,15,10)
        self.driver.write(values)
        messages = self.drain()
        self.assertEqual([m.arbitration_id for m in messages],DEVICE_IDS)
        raw = (messages[0].data[2]<<8)|messages[0].data[3]
        self.assertEqual(raw, (-15000)&0xffff)

    def test_zero_attempts_every_driver_after_a_single_send_failure(self):
        original = self.driver.send
        def fail_one(identifier, data):
            if identifier==DEVICE_IDS[2]:raise RuntimeError('injected transport failure')
            original(identifier,data)
        with mock.patch.object(self.driver,'send',side_effect=fail_one):
            with self.assertRaisesRegex(RuntimeError,'Some zero commands failed'):
                self.driver.zero_all()
        self.assertEqual([m.arbitration_id for m in self.drain()],
                         [identifier for identifier in [0x003]+DEVICE_IDS if identifier!=DEVICE_IDS[2]])

    def controller(self):
        from controller_node import ControllerNode
        node = ControllerNode.__new__(ControllerNode)
        node.lock = threading.RLock()
        node.hardware = True; node.closed = False
        node.armed = node.arming = False; node.arm_generation = 0
        node.previous = np.zeros(10)
        node.output = self.driver
        node.trials = mock.Mock()
        node.trials.snapshot.return_value = {'phase':'idle'}
        return node

    def test_idle_mode_button_matches_legacy_two_message_sequence(self):
        node = self.controller()
        self.assertTrue(node.on_driver_action('prepare_drivers').success)
        self.assertEqual([m.arbitration_id for m in self.drain()],[0x001,0x003])
        self.assertFalse(self.driver.feedback_enabled)
        self.assertFalse(node.armed)

    def test_hardware_arm_requires_preparation_and_all_ten_fresh_reports(self):
        node = self.controller()
        node.config = {'pose_hold':{'hardware_validated':False}}
        self.assertIn('not hardware-validated',node.arm_check())
        node.config['pose_hold']['hardware_validated'] = True
        self.assertIn('Current mode',node.arm_check())
        self.driver.initialized = True
        self.assertIn('all ten',node.arm_check())
        for identifier in FEEDBACK_IDS:
            self.driver.accept_feedback(can.Message(arbitration_id=identifier,
                data=[0,0],is_extended_id=False))
        node.core = mock.Mock()
        node.core.control = {'pose_timeout_s':.3,'command_timeout_s':.3}
        node.pose = np.array([.01,.02,.6])
        node.pose_time = node.command_time = time.monotonic()
        node.drive = node.turn = 0
        node.input_source = 'keyboard'
        self.assertIsNone(node.arm_check())
        self.driver.arrivals[4] = time.monotonic()-2
        self.assertIn('all ten',node.arm_check())

    def test_arm_keeps_prepared_driver_mode_and_retains_run_stop_reason(self):
        from std_srvs.srv import SetBoolRequest
        node = self.controller()
        self.driver.initialized = True
        node.sync_input_source = lambda:None
        node.arm_check = lambda:None
        node.core = mock.Mock()
        node.core.control = {'allocation_tolerance':.02}
        node.core.model.allocate.return_value = (np.ones(10),0.)
        node.config = {'model':{'holding_field_T':.02}}
        node.pose = np.array([.01,.02,.6]);node.heading_unwrapped = .6
        node.video_required = False
        node.trials.snapshot.return_value = {'phase':'prepared'}
        node.trials.video_ready = True
        node.publish_recording = lambda:None
        node.telemetry = lambda:{}
        self.assertTrue(node.on_arm(SetBoolRequest(data=True)).success)
        self.assertEqual([m.arbitration_id for m in self.drain()],DEVICE_IDS)
        node.trials.snapshot.return_value = {'phase':'idle'}
        node.stop('test driver alarm')
        self.assertEqual(node.last_run_stop['reason'],'test driver alarm')
        node.stop('disarmed')
        self.assertEqual(node.last_run_stop['reason'],'test driver alarm')

    def test_tracking_loss_zeros_active_output_once_without_idle_can_flood(self):
        from std_msgs.msg import Bool
        node = self.controller()
        self.driver.initialized = True
        for _ in range(100):node.on_valid(Bool(data=False))
        self.assertEqual(self.drain(),[])
        node.armed = True; node.previous = np.ones(10)
        node.on_valid(Bool(data=False))
        messages = self.drain()
        self.assertEqual([m.arbitration_id for m in messages],DEVICE_IDS)
        self.assertTrue(all(list(m.data[2:4])==[0,0] for m in messages))
        for _ in range(100):node.on_valid(Bool(data=False))
        self.assertEqual(self.drain(),[])
        self.assertFalse(node.armed)

    def test_recording_is_ready_before_any_nonzero_holding_command(self):
        import tempfile
        from controller_node import ControllerNode
        from cs1_experiment.recording import TrialStore
        from std_srvs.srv import SetBoolRequest
        with tempfile.TemporaryDirectory() as folder:
            node=ControllerNode.__new__(ControllerNode)
            node.lock=threading.RLock();node.closed=False;node.armed=node.arming=False
            node.arm_generation=0;node.previous=np.zeros(10)
            node.core=mock.Mock();node.core.control={'allocation_tolerance':.02}
            node.core.model.allocate.return_value=(np.ones(10),0.)
            node.config={'model':{'holding_field_T':.02}}
            node.pose=np.array([.01,.02,.6]);node.heading_unwrapped=.6
            node.sync_input_source=lambda:None;node.arm_check=lambda:None
            node.output=mock.Mock();node.recording_pub=mock.Mock()
            node.trials=TrialStore(folder);node.video_required=True
            node.trials.prepare({},True)
            response=node.on_arm(SetBoolRequest(data=True))
            self.assertFalse(response.success)
            node.output.initialize.assert_not_called();node.output.write.assert_not_called()
            node.trials.cancel_prepared();node.video_required=False;node.trials.prepare({},False)
            with mock.patch.object(node.trials,'start',side_effect=OSError('injected recording failure')):
                response=node.on_arm(SetBoolRequest(data=True))
            self.assertFalse(response.success)
            self.assertFalse(node.armed)
            for call in node.output.write.call_args_list:
                self.assertTrue(np.all(np.asarray(call[0][0])==0))

    def test_reset_cancels_an_in_flight_arming_request(self):
        for action in ('reset_drivers','shutdown'):
            from controller_node import ControllerNode
            from std_srvs.srv import SetBoolRequest
            node = ControllerNode.__new__(ControllerNode)
            node.lock = threading.RLock();node.hardware=True;node.closed=False
            node.armed=node.arming=False;node.arm_generation=0
            node.previous=np.zeros(10);node.drive=node.turn=node.command_time=0
            node.core=mock.Mock();node.config={'model':{'holding_field_T':.02}}
            node.sync_input_source=lambda:None;node.arm_check=lambda:None
            node.output=mock.Mock();node.log=mock.Mock()
            started, release = threading.Event(),threading.Event()
            def initialize():
                started.set();release.wait(timeout=2)
            node.output.initialize.side_effect=initialize
            responses=[]
            worker=threading.Thread(target=lambda:responses.append(node.on_arm(SetBoolRequest(data=True))))
            worker.start();self.assertTrue(started.wait(timeout=1))
            if action == 'reset_drivers':
                self.assertTrue(node.on_driver_action(action).success)
            else:
                node.close()
            release.set();worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertFalse(responses[0].success)
            self.assertFalse(node.armed)
            node.core.model.allocate.assert_not_called()


if __name__=='__main__':unittest.main()
