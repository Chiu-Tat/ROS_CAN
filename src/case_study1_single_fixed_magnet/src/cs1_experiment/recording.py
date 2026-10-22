"""Per-arming recording folders and bounded asynchronous data/video writers."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid


def utc(ns):
    return datetime.fromtimestamp(ns/1e9, timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))


def message_dict(value):
    if hasattr(value, '__slots__'):
        return {name:message_dict(getattr(value, name)) for name in value.__slots__}
    if isinstance(value, (tuple, list)):
        return [message_dict(item) for item in value]
    if isinstance(value, bytes):
        return list(value)
    return value


class TrialStore:
    """One current trial. Files are private to .pending until Save is selected."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pending = self.root/'.pending'
        self.pending.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.folder = None
        self.meta = None
        self.worker = None
        self.events = None
        self.stopping = None
        self.error = ''
        self.video_ready = False
        self.recover()

    def recover(self):
        for folder in sorted(self.pending.glob('trial_*')):
            if folder.is_symlink() or not (folder/'session.json').is_file():
                continue
            meta = json.loads((folder/'session.json').read_text())
            owner = meta.get('owner_pid')
            alive = owner and Path('/proc/%s'%owner).exists()
            if alive and not (folder/'data_complete.json').exists():
                continue
            self.folder, self.meta = folder, meta
            self.meta.update(phase='pending', recovered=True,
                             stop_reason=self.meta.get('stop_reason') or 'Interrupted previous recording')
            if self.meta.get('start_ns') and not self.meta.get('stop_ns'):
                self.meta['stop_ns']=self.meta['start_ns']
            write_json(folder/'session.json', self.meta)
            break

    def snapshot(self):
        with self.lock:
            if self.meta is None:
                return {'phase':'idle'}
            value = dict(self.meta)
            value['path'] = str(self.folder)
            value['error'] = self.error
            if value['phase'] == 'stopping' and (self.folder/'data_complete.json').exists():
                video_done = not value['ui_video_required'] or (self.folder/'video_complete.json').exists()
                if video_done:
                    self.meta['phase'] = value['phase'] = 'pending'
                    write_json(self.folder/'session.json', self.meta)
            return value

    def prepare(self, metadata, video_required=True):
        with self.lock:
            if self.meta is not None:
                raise ValueError('Save or delete the previous trial before arming again')
            identifier = 'trial_'+datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
            self.folder = self.pending/identifier
            self.folder.mkdir()
            self.meta = dict(metadata, id=identifier, phase='prepared', owner_pid=os.getpid(),
                             prepared_utc=utc(time.time_ns()), ui_video_required=bool(video_required),
                             start_ns=None, stop_ns=None, stop_reason=None)
            self.error = ''; self.video_ready = not video_required
            self.events = queue.Queue(maxsize=8192)
            self.stopping = threading.Event()
            self.data_ready = threading.Event()
            try:write_json(self.folder/'session.json', self.meta)
            except Exception:
                shutil.rmtree(self.folder)
                self.folder=self.meta=None
                raise
            return self.snapshot()

    def start(self, arm_metadata=None):
        with self.lock:
            if self.meta is None or self.meta['phase'] != 'prepared' or not self.video_ready:
                raise ValueError('UI/data recording is not ready; use the main UI Arm button')
            starting=dict(self.meta)
            starting.update(arm_metadata or {})
            starting.update(phase='recording',start_ns=time.monotonic_ns(),start_wall_ns=time.time_ns())
            write_json(self.folder/'session.json',starting)
            self.meta=starting
            self.worker = threading.Thread(target=self._write_events, daemon=True)
            self.worker.start()
            if not self.data_ready.wait(2) or self.error:
                raise RuntimeError(self.error or 'Data recorder did not become ready')
            return self.snapshot()

    def record(self, topic, payload, jpeg=None):
        with self.lock:
            if self.meta is None or self.meta['phase'] != 'recording':
                return
            event = {'topic':topic, 'wall_ns':time.time_ns(), 'monotonic_ns':time.monotonic_ns(),
                     'payload':payload}
            if jpeg is not None:
                event['_jpeg'] = bytes(jpeg)
            try:
                self.events.put_nowait(event)
            except queue.Full:
                self.error = 'Data recording queue overflow'

    def mark_control_stop(self, reason, post_stop_s):
        """Control has stopped; keep recording post_stop_s more seconds."""
        with self.lock:
            if self.meta is None or self.meta['phase'] != 'recording':
                return
            now = time.monotonic_ns()
            self.meta.update(control_stop_reason=str(reason), control_stop_ns=now,
                             control_stop_wall_ns=time.time_ns(), post_stop_record_s=float(post_stop_s),
                             post_stop_until_ns=now+int(post_stop_s*1e9))
            write_json(self.folder/'session.json', self.meta)
        self.record('control_stop', {'reason': str(reason), 'post_stop_record_s': float(post_stop_s)})

    def stop(self, reason):
        with self.lock:
            if self.meta is not None and self.meta['phase'] == 'recording':
                self.meta.update(phase='stopping', stop_ns=time.monotonic_ns(),
                                 stop_wall_ns=time.time_ns(), stop_reason=str(reason))
                self.stopping.set()
                write_json(self.folder/'session.json', self.meta)
            return self.snapshot()

    def _write_events(self):
        folder, start = self.folder, self.meta['start_ns']
        names = ['wall_time_utc','unix_time_ns','elapsed_s','armed','hardware','input_source',
                 'drive','turn','tracking_valid','pose_fresh','x_m','y_m','z_m','heading_wrapped_rad',
                 'heading_unwrapped_rad','reason']
        names += ['command_%02d_A'%k for k in range(1,11)]
        names += ['measured_%02d_A'%k for k in range(1,11)]
        names += ['measured_%02d_age_s'%k for k in range(1,11)]
        names += ['measured_%02d_fresh'%k for k in range(1,11)]
        names += ['hold_phase','target_x_m','target_y_m','target_heading_rad',
                  'error_x_m','error_y_m','error_heading_rad']
        count = images = 0
        try:
            (folder/'camera').mkdir(exist_ok=True)
            with (folder/'events.jsonl').open('x') as raw, (folder/'telemetry.csv').open('x',newline='') as flat:
                writer = csv.writer(flat);writer.writerow(names)
                raw.flush();flat.flush();self.data_ready.set()
                while not self.stopping.is_set() or not self.events.empty():
                    try: event = self.events.get(timeout=.05)
                    except queue.Empty: continue
                    event['elapsed_s'] = (event['monotonic_ns']-start)/1e9
                    event['wall_time_utc'] = utc(event['wall_ns'])
                    jpeg = event.pop('_jpeg',None)
                    if jpeg is not None:
                        image_path = 'camera/frame_%07d.jpg'%images
                        (folder/image_path).write_bytes(jpeg)
                        event['payload']['file'] = image_path
                        images += 1
                    raw.write(json.dumps(event, allow_nan=False)+'\n')
                    if event['topic'] == 'controller':
                        state = event['payload'];pose = state.get('pose') or [None]*3
                        feedback = state.get('current_feedback',{})
                        hold = state.get('pose_hold',{})
                        writer.writerow([event['wall_time_utc'],event['wall_ns'],event['elapsed_s'],
                            state.get('armed'),state.get('hardware'),state.get('input_source'),
                            state.get('drive'),state.get('turn'),state.get('tracking_valid'),state.get('pose_fresh'),
                            pose[0],pose[1],state.get('z_m'),pose[2],state.get('heading_unwrapped_rad'),state.get('reason')]
                            +state.get('currents',[None]*10)+feedback.get('values_A',[None]*10)
                            +feedback.get('age_s',[None]*10)+feedback.get('fresh',[False]*10)
                            +[hold.get('phase')]+(hold.get('reference_pose') or [None]*3)
                            +(hold.get('pose_error') or [None]*3))
                    count += 1
                    raw.flush();flat.flush()
        except Exception as exc:
            self.error = 'Data recording failed: '+str(exc)
        finally:
            self.data_ready.set()
            write_json(folder/'data_complete.json', {'events':count,'camera_frames':images,'error':self.error})

    def cancel_prepared(self):
        with self.lock:
            if self.meta is None:return
            if self.meta['phase'] not in ('prepared','cancelled'):
                raise ValueError('An armed trial must be stopped and reviewed, not cancelled')
            shutil.rmtree(self.folder)
            self.folder = self.meta = None
            self.video_ready = False

    def resolve(self, keep, note=''):
        with self.lock:
            if self.snapshot()['phase'] != 'pending':
                raise ValueError('Wait until data/video writers finish before Save/Delete')
            result = self.resolve_folder(self.root, self.folder, keep, note)
            self.folder = self.meta = None
            self.worker = None
            self.video_ready = False
            self.recover()
            return result

    @staticmethod
    def resolve_folder(root, folder, keep, note=''):
        root, folder = Path(root).resolve(), Path(folder)
        if folder.is_symlink() or folder.parent.resolve() != root/'.pending' or not folder.name.startswith('trial_'):
            raise ValueError('Refusing to modify a folder outside this recorder pending directory')
        meta = json.loads((folder/'session.json').read_text())
        if meta['phase'] != 'pending':raise ValueError('Trial is still being written')
        if not keep:
            shutil.rmtree(folder)
            return None
        meta.update(phase='saved', introduction=note.strip(), saved_utc=utc(time.time_ns()))
        duration = ((meta.get('stop_ns') or 0)-(meta.get('start_ns') or 0))/1e9
        video = json.loads((folder/'video_complete.json').read_text()) if (folder/'video_complete.json').exists() else {}
        data = json.loads((folder/'data_complete.json').read_text()) if (folder/'data_complete.json').exists() else {}
        text = '# CS1 — one fixed magnet, manually controlled free magnet\n\n'
        text += (note.strip()+'\n\n') if note.strip() else ''
        text += 'Trial: %s\n\n'%meta['id']
        text += 'Mode: %s. Input: %s.\n\n'%('CAN hardware' if meta.get('hardware') else 'CAN disabled / preview',meta.get('input_source','unknown'))
        text += 'Start: %s. Duration: %.3f s. Stop: %s.\n\n'%(utc(meta['start_wall_ns']) if meta.get('start_wall_ns') else 'not recorded',duration,meta.get('stop_reason'))
        if meta.get('control_stop_ns') and meta.get('start_ns'):
            text += 'Control stopped %.3f s after start (%s); recording continued after the stop.\n\n'%(
                (meta['control_stop_ns']-meta['start_ns'])/1e9, meta.get('control_stop_reason'))
        text += 'Pose at arming [x_m, y_m, heading_rad]: %s.\n\n'%meta.get('pose_at_arm',meta.get('pose_at_prepare'))
        text += 'Initial pose and calibration: see session.json. Position is in EM meters; headings in radians; currents in amperes.\n\n'
        text += 'Files:\n- ui.avi: complete main UI, including camera, TFW, controls and current table (MJPEG).\n- video_timestamps.csv: captured/duplicated frame timing; video_complete.json: frame/drop/error counts.\n- telemetry.csv: controller-rate pose, input, commanded/measured coil currents and freshness.\n- events.jsonl: received pose, continuous heading, tracking status, commands, CAN feedback and image timestamps.\n- camera/: original received JPEG previews.\n- session.json: configuration, calibration, TFW provenance and initial-pose snapshot.\n\n'
        text += 'Counts: %s data events; %s raw camera frames; %s UI video frames.\n\n'%(data.get('events',0),data.get('camera_frames',0),video.get('frames',0))
        text += 'Timing uses host UTC and a common host monotonic clock. Camera/pose/video are not hardware synchronized. Missing current feedback is blank/null, not measured zero.\n'
        if meta.get('recovered'):text += '\nRecovered interrupted session; video/data may be incomplete.\n'
        if data.get('error') or video.get('error'):text += '\nRecording errors: %s %s\n'%(data.get('error',''),video.get('error',''))
        destination = root/folder.name
        if destination.exists():raise FileExistsError(str(destination))
        write_json(folder/'session.json',meta)
        (folder/'README.md').write_text(text)
        try:os.rename(str(folder),str(destination))
        except OSError:
            meta['phase']='pending';write_json(folder/'session.json',meta)
            raise
        return destination


class UiVideoRecorder:
    """Encode QImages on a worker. Never block keyboard/Qt with video I/O."""
    def __init__(self, folder, size, fps=5):
        self.folder = Path(folder)
        self.size = tuple(max(2,int(v)//2*2) for v in size)
        self.fps = int(fps)
        self.frames = queue.Queue(maxsize=3)
        self.ready = threading.Event();self.done = threading.Event();self.finish = threading.Event()
        self.start_ns = self.stop_ns = None
        self.error = '';self.dropped = 0
        self.worker = threading.Thread(target=self._run,daemon=True);self.worker.start()

    def begin(self, start_ns):
        if self.start_ns is None:self.start_ns=start_ns

    def submit(self, image, capture_ns=None):
        if self.start_ns is None or self.finish.is_set():return
        try:self.frames.put_nowait((time.monotonic_ns() if capture_ns is None else capture_ns,image.copy()))
        except queue.Full:self.dropped+=1

    def stop(self, stop_ns=None):
        self.stop_ns = time.monotonic_ns() if stop_ns is None else stop_ns
        self.finish.set()

    def _run(self):
        writer = None;count=captures=0
        try:
            # Some OpenCV distributions alter Qt environment variables on import.
            keys = ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR')
            previous = {key:os.environ.get(key) for key in keys}
            import cv2
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            import numpy as np
            from PyQt5 import QtGui
            writer = cv2.VideoWriter(str(self.folder/'ui.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
            if not writer.isOpened():raise RuntimeError('Cannot open MJPEG video encoder')
            self.ready.set()
            last = None;last_capture = None
            with (self.folder/'video_timestamps.csv').open('x',newline='') as stream:
                index=csv.writer(stream);index.writerow(['frame','video_elapsed_s','capture_elapsed_s','duplicated'])
                def emit(frame, captured, duplicated):
                    nonlocal count
                    writer.write(frame)
                    index.writerow([count,count/self.fps,(captured-self.start_ns)/1e9,duplicated]);count+=1
                while not self.finish.is_set() or not self.frames.empty():
                    try:captured,image=self.frames.get(timeout=.05)
                    except queue.Empty:continue
                    rgb=image.convertToFormat(QtGui.QImage.Format_RGB888)
                    bits=rgb.bits();bits.setsize(rgb.bytesPerLine()*rgb.height())
                    array=np.frombuffer(bits,np.uint8).reshape(rgb.height(),rgb.bytesPerLine())[:,:rgb.width()*3].reshape(rgb.height(),rgb.width(),3)
                    frame=cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
                    scale=min(self.size[0]/rgb.width(),self.size[1]/rgb.height())
                    w,h=max(1,round(rgb.width()*scale)),max(1,round(rgb.height()*scale))
                    fitted=np.zeros((self.size[1],self.size[0],3),np.uint8)
                    x,y=(self.size[0]-w)//2,(self.size[1]-h)//2
                    fitted[y:y+h,x:x+w]=cv2.resize(frame,(w,h))
                    target=max(0,int((captured-self.start_ns)/1e9*self.fps))
                    if target<count:continue
                    while count<target:emit(fitted if last is None else last,captured if last_capture is None else last_capture,True)
                    emit(fitted,captured,False);last,last_capture=fitted,captured;captures+=1
                    stream.flush()
                    if (self.folder/'ui.avi').stat().st_size>1_800_000_000:raise RuntimeError('Video size limit reached')
                if last is not None and self.stop_ns is not None:
                    target=max(count,int((self.stop_ns-self.start_ns)/1e9*self.fps+.999))
                    while count<target:emit(last,last_capture,True)
        except Exception as exc:self.error='UI video recording failed: '+str(exc)
        finally:
            if self.start_ns is not None and not count and not self.error:self.error='No UI frames captured'
            if writer is not None:writer.release()
            write_json(self.folder/'video_complete.json',{'fps':self.fps,'size_px':self.size,'frames':count,
                'captured_frames':captures,'dropped_capture_requests':self.dropped,'error':self.error})
            self.ready.set();self.done.set()
