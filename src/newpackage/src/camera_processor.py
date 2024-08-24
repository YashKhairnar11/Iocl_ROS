#!/usr/bin/python3
import cv2 as cv
import rospy
from cv_bridge import CvBridge
import basler as b
from sensor_msgs.msg import Image
from utils.createConfig import create
import threading as t


class CameraNode(t.Thread):
    def __init__(self, camera_id, cam_config, rate, bridge):
        """
        Initializes a CameraNode thread.
        
        Args:
            thread_name: Name of the cameraNode Thread.
            cameraID: The camera ID as a string (e.g., 'basler_0').
            camConfig_dict: The configuration dictionary for a single camera.
            exit_event: Event to signal thread exit.
        """
        super(CameraNode, self).__init__(daemon=False)
        # self.exit_event = exit_event
        # self.flag = False
        self.camera_id = camera_id
        self.camConfig = cam_config
        self.model_type = self.camConfig['model_type']
        self.name = self.camConfig['name']
        self.topic = f"{self.name}/{self.model_type}"
        self.pub = rospy.Publisher(self.topic, Image, queue_size=10)
        self.bridge = bridge
        self.frame_counter = 0
        self.camera = self._set_camera()
        self.rate = rate

        

    def _set_camera(self):
        """
        Returns a cv2.VideoCapture() or basler.VideoCapture() class based on the camera_type.
        Also sets the exposure and gain value of the camera.
        """
        if self.camConfig['camera_type'] == 'opencv':
            if self.camera_id.startswith('opencv_'):
                camera = cv.VideoCapture(int(self.camera_id[7:]))
            else:
                camera = cv.VideoCapture(self.camera_id)
        else:  # Assuming 'basler' camera type
            if self.camera_id.startswith('basler_d'):
                camera = b.VideoCapture(int(self.camera_id[7:]), self.logger)
            else:
                camera = b.VideoCapture(self.camera_id, self.logger)
            camera.set_exposure(self.camConfig['exposure'])
            camera.set_gain(self.camConfig['gain'])
            camera.set_packet_size(self.camConfig['packet_size'])
        
        return camera
    
    def run(self):
        """
        The run method of the thread that handles frame capture and publishing.
        """
        print(self.camConfig)
        while True : #not self.exit_event.is_set():
            ret, frame = self.camera.read()
            if ret:
                self.frame_counter += 1
                # if self.frame_counter % self.camConfig['frame_skip'] == 0:
                frame = cv.resize(frame, (self.camConfig['resize'], self.camConfig['resize']))
                ros_img = bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                # signal = "START"  # Assuming the proximity signal is always START for simplicity
                # if signal == "START" and not self.flag:
                print(f"[DEBUG] {self.camera_id} RECEIVED START SIGNAL")
                self.pub.publish(ros_img)
                self.rate.sleep()
            else:
                if not self.camera.isOpened():
                    print(f"[ERROR] {self.camera_id} CANNOT OPEN CAMERA {self.camera_id}")  
        
    def start_camera_thread(self):
        """Starts the camera thread."""
        self.start()
        
    def join_camera_thread(self):
        """Joins the camera thread."""
        self.join()

if __name__ == '__main__':
    # Create the config file object
    ConfigPath = '/home/yash/Desktop/iocl/src/newpackage/config/Config.yaml'
    config = create(ConfigPath)  # Dictionary
    cameras = config["cameras"]  # List of  cam_dicts

    allCamDict = {
        camDict['camera_id']:{
            'camera_type': camDict['camera_type'], 
            'model_type': camDict['model_type'], 
            'name' : camDict['name'],
            # 'frame_skip': camDict['frame_skip'], 
            # 'visualize_or_publish': camDict['visualize_or_publish'], 
            # 'distance': camDict['distance'], 
            # 'disappeared':  camDict['disappeared'],
            # 'detection_skip': camDict['detection_skip'], 
            # 'fps': camDict['fps'], 
            # 'ip': camDict['ip'],
            # 'endpoint': camDict['endpoint'],
            # 'port': camDict['port'],
            # 'wsip': camDict['wsip'],
            # 'wsport': camDict['wsport'],
            # 'required_bboxes':camDict['required_bboxes'],
            # 'priority':camDict['priority'],
            'exposure': camDict['exposure'], 
            'gain': camDict['gain'],
            'packet_size':camDict['packet_size'],
            'resize': camDict['resize'],
        } for camDict in cameras
    }
   
    bridge = CvBridge()
    rospy.init_node('CameraProcessorNode',anonymous=True) #start the node
    rate = rospy.Rate(10)
    Camthreads = {}

    try:
        # Create the thread for each of the camera
        for camera_id in allCamDict:
           cam_config = allCamDict[camera_id]
           cam_thread = CameraNode(camera_id, cam_config, rate, bridge)
           Camthreads[camera_id] = cam_thread
           cam_thread.start_camera_thread()
        print('ALL CAMERA THREADS STARTED.')

    except KeyboardInterrupt as e:
        for camera_id, cam_thread in Camthreads.items():
            cam_thread.join_camera_thread()
        rospy.signal_shutdown('KeyboardInterrupt')
        print("Error occurred, exiting main()")
