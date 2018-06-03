#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from keras.models import load_model
from cv_bridge import CvBridge
from math import pow, sqrt
from keras import backend as K
import tf
import cv2
import yaml
import numpy as np

STATE_COUNT_THRESHOLD = 3
SMOOTH = 1

def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

	type = 'simulator'

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

	self.state = TrafficLight.UNKNOWN

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.last_wp = -1
        self.state_count = 0
	self.distance_threshold_max = 40
	self.distance_threshold_min = 20
	self.last_dist = -1

	#setting up classifier
	model = load_model('models/classifier_' + type + '.h5')
	resize_width = 32
	resize_height = 64
	self.light_classifier.setup_classifier(model, resize_width, resize_height)
	self.invalid_state_number = 3

	#setting up detector
	self.detector_model = load_model('models/detector_' + type + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
	self.detector_model._make_predict_function()
	self.resize_width = 128
	self.resize_height = 96

	self.resize_height_ratio = 600/float(self.resize_height)
	self.resize_width_ratio = 800/float(self.resize_width)
	self.middle_col = self.resize_width/2
	self.projection_threshold = 2
	self.projection_min = 200

	#classifier vars
	self.bridge = CvBridge()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def dist_to_point(self, pose, wp_pose):
	x_squared = pow((pose.position.x - wp_pose.position.x), 2)
	y_squared = pow((pose.position.y - wp_pose.position.y), 2)
	dist = sqrt(x_squared + y_squared)
	return dist

    def get_closest_waypoint(self, pose, waypoints):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement

	min_dist = float("inf")
	closest_wp_idx = -1	

        for idx, wp in enumerate(waypoints):
		dist = self.dist_to_point(pose, wp.pose.pose)
		if(dist < min_dist):
			min_dist = dist
			closest_wp_idx = idx
	return closest_wp_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
	return light.state        

	#if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False

       # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
       # return self.light_classifier.get_classification(cv_image)

    def extract_image(self, mask, img):
	if(np.max(mask) < self.projection_min):
		return None


	row_projection = np.sum(mask, axis = 1)
	row_index = np.argmax(row_projection)

	if(np.max(row_projection) < self.projection_threshold):
		return None


	zero_row_indexes = np.argwhere(row_projection <= self.projection_threshold)
	
	#top
	top_part = zero_row_indexes[zero_row_indexes < row_index]
	top = np.max(top_part) if top_part.size > 0 else 0
	
	#bottom
	bottom_part = zero_row_indexes[zero_row_indexes > row_index]
	bottom = np.min(bottom_part) if bottom_part.size > 0 else self.resize_height


	roi = mask[top:bottom,:]
	column_projection = np.sum(roi, axis=0)

	if(np.max(column_projection) < self.projection_min):
		return None


	non_zero_column_index = np.argwhere(column_projection > self.projection_min)

	index_of_column_index = np.argmin(np.abs(non_zero_column_index - self.middle_col))
	column_index = non_zero_column_index[index_of_column_index][0]

	zero_column_indexes = np.argwhere(column_projection == 0)
	
	#left
	left_side = zero_column_indexes[zero_column_indexes < column_index]
	left = np.max(left_side) if left_side.size > 0 else 0
	
	#right
	right_side = zero_column_indexes[zero_column_indexes > column_index]
	right = np.min(right_side) if right_side.size > 0 else self.resize_width

	return img[int(top*self.resize_height_ratio):int(bottom*self.resize_height_ratio), int(left*self.resize_width_ratio):int(right*self.resize_width_ratio)]

    def detect_traffic_light(self, img):
	resize_img = cv2.cvtColor(cv2.resize(img, (self.resize_width, self.resize_height)), cv2.COLOR_RGB2GRAY)
	resize_img = resize_img[..., np.newaxis]
	
	img_mask = self.detector_model.predict(resize_img[None, :, :, :], batch_size=1)[0]
	img_mask = (img_mask[:,:,0]*255).astype(np.uint8)
	return self.extract_image(img_mask, img)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
	line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):

            	car_wp_idx = self.get_closest_waypoint(self.pose.pose, self.lights)
		tl = self.lights[car_wp_idx]

        	#TODO find the closest visible traffic light (if one exists)
		diff = len(self.waypoints)
		for i, light in enumerate(self.lights):
			line = stop_line_positions[i]

			line_pose = Pose()
			line_pose.position.x = line[0]
			line_pose.position.y = line[1]

			temp_wp_idx = self.get_closest_waypoint(line_pose, self.waypoints)

			d = temp_wp_idx - car_wp_idx
			if d >= 0 and d < diff:
				diff = d
				closest_light = light
				line_wp_idx = temp_wp_idx

	car_dist = self.dist_to_point(self.pose.pose, tl.pose.pose)

#	rospy.logwarn("DISTANCE: {}, LAST_DISTANCE: {}".format(car_dist, self.last_dist))

        if closest_light and car_dist<self.distance_threshold_max and car_dist>self.distance_threshold_min and self.last_dist>=car_dist:
		cv_img = self.bridge.imgmsg_to_cv2(self.camera_image, 'rgb8')
		tl_image = self.detect_traffic_light(cv_img)
		if tl_image is not None:
			state = self.light_classifier.get_classification(tl_image)
			state = state if(state != self.invalid_state_number) else TrafficLight.UNKNOWN	
			
			rospy.logwarn("[TL_DETECTOR] State of traffic light is {}".format(state))
		else:
			rospy.logwarn("[TL_DETECTOR] No TL is detected")	
		self.last_dist = car_dist
                return line_wp_idx, state
        
	self.last_dist = car_dist
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
