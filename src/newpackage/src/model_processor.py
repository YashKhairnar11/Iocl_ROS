#!/usr/bin/python3
import cv2 as cv
import rospy
from cv_bridge import CvBridge, CvBridgeError
import basler as b
from sensor_msgs.msg import Image
import yaml
from utils.createConfig import create
import threading as t
import logging

class ModelProcessorNode(t.Thread):
    def __init__(self,camera_id,cam_config,rate,bridge):
        super(ModelProcessorNode,self).__init__(daemon=None)
        self.camera_id = camera_id #the cam associated with this model processor
        self.model_type = cam_config['model_type']

        #pub/sub details
        self.name = cam_config['name']
        self.subTopic = f"{self.name}/{self.model_type}"
        self.pubTopic = f"model/{self.subTopic}"
        #subsribe from the camNode
        self.sub = rospy.Subscriber(self.subTopic,Image,self.callback)
        self.pub = rospy.Publisher(self.pubTopic,Image,queue_size=10)
        self.bridge = bridge
        self.frame_counter = 0

    def callback(self,data):
        rospy.loginfo(f"recieved Data")
        self.pub.publish(data)

    def run(self):
        """
        Description : Used to start the model thread which subscribes to the camera Publisher and takes the frame,
                     then it predicts the result over this frame, and publishes again to the collator
        """
        rospy.spin()

    def start_processing_thread(self):
        """
        DESCRIPTION:
            Calls the start() method of threading.Thread() class
        ARGUMENTS:
        RETURNS:
        """
        self.start()

    def join_model_thread(self):
        """
        DESCRIPTION:
            Calls the join() method of threading.Thread() class
        ARGUMENTS:
        RETURNS:
        """
        self.join()


if __name__=='__main__':
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
            # 'exposure': camDict['exposure'], 
            # 'gain': camDict['gain'],
            # 'packet_size':camDict['packet_size'],
            # 'resize': camDict['resize'],
        } for camDict in cameras
    }
   
    bridge = CvBridge()
    rospy.init_node('ModelProcessorNode',anonymous=True) #start the node
    rate = rospy.Rate(10)
    Modelthreads = {}

    try:
        # Create the thread for each of the model associated with each camera topic
        for camera_id in allCamDict:
           cam_config = allCamDict[camera_id]
           mod_thread = ModelProcessorNode(camera_id, cam_config, rate, bridge)
           Modelthreads[camera_id] = mod_thread
           mod_thread.start_processing_thread()
        print('ALL MODEL THREADS STARTED.')

    except KeyboardInterrupt as e:
        for camera_id, mod_thread in Modelthreads.items():
            mod_thread.join_model_thread()
        rospy.signal_shutdown('KeyboardInterrupt')
        print("Error occurred, exiting main()")
