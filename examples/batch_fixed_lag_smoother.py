#!/usr/bin/env python
''' Simple 2D Fixed-Lag Smoother Class using pyGTSAM and LevenbergMarquardtOptimizer '''

# Author: Nick R. Rypkema (rypkema@mit.edu)
# License: MIT

import numpy as np

from pygtsam import Symbol, extractPose2, extractPose3, extractPoint3, extractKeys
from pygtsam import symbol as _symbol
from pygtsam import Point2, Rot2, Pose2, PriorFactorPose2, PriorFactorPoint2, BetweenFactorPose2, RangeFactorPose2Point2, BearingRangeFactorPose2Point2
from pygtsam import Point3, Rot3, Pose3, PriorFactorPose3, BetweenFactorPose3
from pygtsam import SmartFactor
from pygtsam import Cal3_S2, SimpleCamera, simpleCamera
from pygtsam import StereoPoint2, Cal3_S2Stereo, GenericStereoFactor3D, GenericProjectionFactorPose3Point3Cal3_S2
from pygtsam import NonlinearEqualityPose3
from pygtsam import Isotropic
from pygtsam import Diagonal, Values, Marginals
from pygtsam import ISAM2, NonlinearOptimizer, NonlinearFactorGraph, LevenbergMarquardtOptimizer, DoglegOptimizer, LevenbergMarquardtParams

def symbol(ch, i): 
    return _symbol(ord(ch), i)

def vector(v): 
    return np.float64(v)

def matrix(m): 
    return np.float64(m)

def vec(*args):
    return vector(list(args))

# noise classes to encapsulate pygtsam noise types
class Noise1D(object):
    def __init__(self, x):
        self.noise = Diagonal.Sigmas(vec(x), True)

class Noise2D(object):
    def __init__(self, x, y):
        self.noise = Diagonal.Sigmas(vec(x, y), True)

class Noise3D(object):
    def __init__(self, x, y, z):
        self.noise = Diagonal.Sigmas(vec(x, y, z), True)

# factor classes to encapsulate pygtsam factor types
class priorFactorPoint2(object):
    def __init__(self, x, y, noise):
        if type(noise) is not Noise2D:
            raise TypeError("noise must be of type Noise2D")
        self.point2 = Point2(x, y)
        self.noise = noise.noise

class priorFactorPose2(object):
    def __init__(self, x, y, theta, noise):
        if type(noise) is not Noise3D:
            raise TypeError("noise must be of type Noise3D")
        self.pose2 = Pose2(x, y, theta)
        self.noise = noise.noise

class betweenFactorPose2(object):
    def __init__(self, delta_forward, delta_left, delta_theta, noise):
        if type(noise) is not Noise3D:
            raise TypeError("noise must be of type Noise3D")
        self.pose2 = Pose2(delta_forward, delta_left, delta_theta)
        self.noise = noise.noise

class rangeFactorPose2Point2(object):
    def __init__(self, landmark_id, range_val, noise):
        if type(noise) is not Noise1D:
            raise TypeError("noise must be of type Noise1D")
        self.landmark_id = landmark_id
        self.range_val = range_val
        self.noise = noise.noise

class bearingRangeFactorPose2Point2(object):
    def __init__(self, landmark_id, bearing_val, range_val, noise):
        if type(noise) is not Noise2D:
            raise TypeError("noise must be of type Noise2D")
        self.landmark_id = landmark_id
        self.bearing_val = Rot2(bearing_val)
        self.range_val = range_val
        self.noise = noise.noise

# fixed-lag smoother class using encapsulated noise and factor classes
class fixed_lag_smoother(object):
    def __init__(self, max_poses):
        self.max_poses = max_poses
        self._num_poses = 0
        self._num_landmarks = 0
        self._full = False
        self._pose_to_odometry_or_prior = {}
        self._pose_to_measurements = {}
        self._landmark_to_prior = {}
        self._free_factors = []
        self._pose_ids = []
        self._max_factor_id = 0
        self._factor_graph = NonlinearFactorGraph()
        self._values = Values()
        self._head_pose_id = None
        self._tail_pose_id_idx = None

    def add_landmark(self, x, y, prior_factor=None):
        if prior_factor is not None and type(prior_factor) is not priorFactorPoint2:
            raise TypeError("prior_factor must be of type priorFactorPoint2")
        l_pos = Point2(x, y)
        l_id = symbol('l', self._num_landmarks)
        self._values.insert(l_id, l_pos)
        self._num_landmarks += 1
        self._landmark_to_prior[l_id] = []
        if prior_factor is not None:
            if not self._free_factors:
                self._factor_graph.add(PriorFactorPoint2(l_id, prior_factor.point2, prior_factor.noise))
                self._landmark_to_prior[l_id].append(self._max_factor_id)
                self._max_factor_id += 1
            else:
                f_id = self._free_factors.pop()
                self._factor_graph.replace(f_id, PriorFactorPoint2(l_id, prior_factor.point2, prior_factor.noise))
                self._landmark_to_prior[l_id].append(f_id)           
        return l_id

    def add_pose(self, replace_noise, x=None, y=None, theta=None, prior_factor=None, odometry_factor=None):
        if type(replace_noise) is not Noise3D:
            raise TypeError("replace_noise must be of type Noise3D")

        if x is not None and y is not None and theta is not None and prior_factor is None:
            sel = 'no_factor'
        elif x is not None and y is not None and theta is not None and prior_factor is not None:
            sel = 'prior'
        elif odometry_factor is not None:
            sel = 'odometry'
        else:
            raise ValueError("three choices:\n  - x, y, theta not None\n  - x, y, theta, prior_factor not None\n  - odometry_factor not None")

        if not self._full:
            if sel == 'no_factor':
                p_pos = Pose2(x, y, theta)
                p_id = symbol('x', self._num_poses)
                self._values.insert(p_id, p_pos)
                self._num_poses += 1
                self._pose_to_odometry_or_prior[p_id] = []
                self._pose_to_measurements[p_id] = []
            elif sel == 'prior':
                if type(prior_factor) is not priorFactorPose2:
                    raise TypeError("prior_factor must be of type priorFactorPose2")
                p_pos = Pose2(x, y, theta)
                p_id = symbol('x', self._num_poses)
                self._values.insert(p_id, p_pos)
                self._num_poses += 1
                self._pose_to_odometry_or_prior[p_id] = []
                self._pose_to_measurements[p_id] = []
                if not self._free_factors:
                    self._factor_graph.add(PriorFactorPose2(p_id, prior_factor.pose2, prior_factor.noise))
                    self._pose_to_odometry_or_prior[p_id].append(self._max_factor_id)
                    self._max_factor_id += 1
                else:
                    f_id = self._free_factors.pop()
                    self._factor_graph.replace(f_id, PriorFactorPose2(p_id, prior_factor.pose2, prior_factor.noise))
                    self._pose_to_odometry_or_prior[p_id].append(f_id)
            elif sel == 'odometry':
                if self._head_pose_id is None:
                    raise Exception("cannot add odometry-based pose because no prior poses exist")
                if type(odometry_factor) is not betweenFactorPose2:
                    raise TypeError("odometry_factor must be of type betweenFactorPose2")
                p_id = symbol('x', self._num_poses)
                factor = BetweenFactorPose2(self._head_pose_id, p_id, odometry_factor.pose2, odometry_factor.noise)
                p_pos = self._values.atPose2(self._head_pose_id).compose(odometry_factor.pose2)
                self._values.insert(p_id, p_pos)
                self._num_poses += 1
                self._pose_to_odometry_or_prior[p_id] = []
                self._pose_to_measurements[p_id] = []
                if not self._free_factors:
                    self._factor_graph.add(factor)
                    self._pose_to_odometry_or_prior[p_id].append(self._max_factor_id)
                    self._max_factor_id += 1
                else:
                    f_id = self._free_factors.pop()
                    self._factor_graph.replace(f_id, factor)
                    self._pose_to_odometry_or_prior[p_id].append(f_id)
            
            self._pose_ids.append(p_id)
            self._head_pose_id = p_id

            if self._tail_pose_id_idx is None:
                self._tail_pose_id_idx = 0
            
            if self._num_poses >= self.max_poses:
                self._full = True

        else:
            if sel == 'no_factor':
                p_pos = Pose2(x, y, theta)
                p_id = self._pose_ids[self._tail_pose_id_idx]
                self.remove_factors(p_id)
                self._values.update(p_id, p_pos)
            elif sel == 'prior':
                if type(prior_factor) is not priorFactorPose2:
                    raise TypeError("prior_factor must be of type priorFactorPose2")
                p_pos = Pose2(x, y, theta)
                p_id = self._pose_ids[self._tail_pose_id_idx]
                self.remove_factors(p_id)
                self._values.update(p_id, p_pos)
                if not self._free_factors:
                    self._factor_graph.add(PriorFactorPose2(p_id, prior_factor.pose2, prior_factor.noise))
                    self._pose_to_odometry_or_prior[p_id].append(self._max_factor_id)
                    self._max_factor_id += 1
                else:
                    f_id = self._free_factors.pop()
                    self._factor_graph.replace(f_id, PriorFactorPose2(p_id, prior_factor.pose2, prior_factor.noise))
                    self._pose_to_odometry_or_prior[p_id].append(f_id)
            elif sel == 'odometry':
                if self._head_pose_id is None:
                    raise Exception("cannot add odometry-based pose because no prior poses exist")
                if type(odometry_factor) is not betweenFactorPose2:
                    raise TypeError("odometry_factor must be of type betweenFactorPose2")
                p_id = self._pose_ids[self._tail_pose_id_idx]
                factor = BetweenFactorPose2(self._head_pose_id, p_id, odometry_factor.pose2, odometry_factor.noise)
                p_pos = self._values.atPose2(self._head_pose_id).compose(odometry_factor.pose2)
                self.remove_factors(p_id)
                self._values.update(p_id, p_pos) 
                if not self._free_factors:
                    self._factor_graph.add(factor)
                    self._pose_to_odometry_or_prior[p_id].append(self._max_factor_id)
                    self._max_factor_id += 1
                else:
                    f_id = self._free_factors.pop()
                    self._factor_graph.replace(f_id, factor)
                    self._pose_to_odometry_or_prior[p_id].append(f_id)
            
            self._head_pose_id = p_id

            self._tail_pose_id_idx += 1
            if self._tail_pose_id_idx == len(self._pose_ids):
                self._tail_pose_id_idx = 0

            t_id = self._pose_ids[self._tail_pose_id_idx]
            t_pos = self._values.atPose2(t_id)
            self.remove_factors(t_id)
            if not self._free_factors:
                self._factor_graph.add(PriorFactorPose2(t_id, t_pos, replace_noise.noise))
                self._pose_to_odometry_or_prior[p_id].append(self._max_factor_id)
                self._max_factor_id += 1
            else:
                f_id = self._free_factors.pop()
                self._factor_graph.replace(f_id, PriorFactorPose2(t_id, t_pos, replace_noise.noise))
                self._pose_to_odometry_or_prior[t_id].append(f_id)

            return p_id

    def add_measurement(self, measurement_factor, pose2_id=None):
        if pose2_id is None:
            p_id = self._head_pose_id
        else:
            p_id = pose2_id
        if type(measurement_factor) is rangeFactorPose2Point2:
            if not self._free_factors:
                self._factor_graph.add(RangeFactorPose2Point2(p_id, measurement_factor.landmark_id, measurement_factor.range_val, measurement_factor.noise))
                self._pose_to_measurements[p_id].append(self._max_factor_id)
                self._max_factor_id += 1
            else:
                f_id = self._free_factors.pop()
                self._factor_graph.replace(f_id, RangeFactorPose2Point2(p_id, measurement_factor.landmark_id, measurement_factor.range_val, measurement_factor.noise))
                self._pose_to_measurements[p_id].append(f_id)
        elif type(measurement_factor) is bearingRangeFactorPose2Point2:
            if not self._free_factors:
                self._factor_graph.add(BearingRangeFactorPose2Point2(p_id, measurement_factor.landmark_id, measurement_factor.bearing_val, measurement_factor.range_val, measurement_factor.noise))
                self._pose_to_measurements[p_id].append(self._max_factor_id)
                self._max_factor_id += 1
            else:
                f_id = self._free_factors.pop()
                self._factor_graph.replace(f_id, BearingRangeFactorPose2Point2(p_id, measurement_factor.landmark_id, measurement_factor.bearing_val, measurement_factor.range_val, measurement_factor.noise))
                self._pose_to_measurements[p_id].append(f_id)
        else:
            raise TypeError("measurement_factor must be of type rangeFactorPose2Point2 or of type bearingRangeFactorPose2Point2")


    def remove_factors(self, pose2_id):
        for f in self._pose_to_odometry_or_prior[pose2_id]:
            self._factor_graph.replace(f, None)
            self._free_factors.append(f)
        self._pose_to_odometry_or_prior[pose2_id] = []
        for f in self._pose_to_measurements[pose2_id]:
            self._factor_graph.replace(f, None)
            self._free_factors.append(f)
        self._pose_to_measurements[pose2_id] = []

    def update(self):
        batchOptimizer = LevenbergMarquardtOptimizer(self._factor_graph, self._values, LevenbergMarquardtParams())
        self._values = batchOptimizer.optimize()
