#!/usr/bin/env python

import math
import numpy as np

from isam_fixed_lag_smoother import Noise1D, Noise2D, Noise3D
from isam_fixed_lag_smoother import priorFactorPoint2, priorFactorPose2, betweenFactorPose2, rangeFactorPose2Point2, bearingRangeFactorPose2Point2
from isam_fixed_lag_smoother import fixed_lag_smoother

# create noise values for priors, odometry, and measurements 
priorNoise = Noise3D(0.2, 0.2, 0.05)
landmarkPriorNoise = Noise2D(0.2, 0.2)
odometryNoise = Noise3D(0.1, 0.05, 0.01)
rangeNoise = Noise1D(0.25)
bearingRangeNoise = Noise2D(0.1, 0.25)
# special noise value for 'pinning' down the tail to its current value when we exceed the length of the fixed lag
replacementPriorNoise = Noise3D(0.02, 0.02, 0.005)

# create the fixed lag smoother, with example size 4
fls = fixed_lag_smoother(4)

# add a landmark to our fixed lag smoother at (0,0) (note that there is no cap on the number of landmarks allowed)
lm = fls.add_landmark(0, 0, priorFactorPoint2(0, 0, landmarkPriorNoise))

# add an initial pose at (1,1,-pi/2) with prior factor at this position
fls.add_pose(replacementPriorNoise, 1, 1, -math.pi/2, priorFactorPose2(1, 1, -math.pi/2, priorNoise), None)

# lets create an odometry factor of (delta_forward,delta_left,delta_theta)=(2,0,-pi/2) to continuously drive in a square of size 2x2 with some random Gaussian noise
odometry = betweenFactorPose2(2+np.random.normal(0, 0.1), 0+np.random.normal(0, 0.05), -math.pi/2+np.random.normal(0, 0.01), odometryNoise)

# add a second pose using our odometry factor, with a bearing/range measurement to the landmark with some random Gaussian noise
fls.add_pose(replacementPriorNoise, None, None, None, None, odometry)
fls.add_measurement(bearingRangeFactorPose2Point2(lm, -math.pi/4+np.random.normal(0, 0.1), math.sqrt(2.0)+np.random.normal(0, 0.25), bearingRangeNoise))

# add a third pose using our odometry factor with some new random Gaussian noise
odometry = betweenFactorPose2(2+np.random.normal(0, 0.1), 0+np.random.normal(0, 0.05), -math.pi/2+np.random.normal(0, 0.01), odometryNoise)
fls.add_pose(replacementPriorNoise, None, None, None, None, odometry)

# print out the initial factor graph and values before optimization
print 'initial factor graph and values:'
fls._factor_graph.printf()
fls._values.printf()
print ''

# optimize the factor graph - the first update calls a batch optimizer, all subsequent updates use iterative smoothing and mapping (ISAM2)
fls.update()

# print out the factor graph and values after optimization
print 'factor graph optimized!!!'
fls._isam.printFactors()
fls._isam_result.printf()
print ''

# continuously drive in a square of size 2x2 for numreps
numreps = 100
for i in xrange(numreps):
	# add a new pose using our odometry factor with some new random Gaussian noise
	odometry = betweenFactorPose2(2+np.random.normal(0, 0.1), 0+np.random.normal(0, 0.05), -math.pi/2+np.random.normal(0, 0.01), odometryNoise)
	fls.add_pose(replacementPriorNoise, None, None, None, None, odometry)
	factor_roll = np.random.uniform(0,4)
	if factor_roll < 1:		# 25% chance of adding a range measurement factor
		fls.add_measurement(rangeFactorPose2Point2(lm, math.sqrt(2.0), rangeNoise))
	if factor_roll >= 3:	# 25% chance of adding a bearing/range measurement factor
		fls.add_measurement(bearingRangeFactorPose2Point2(lm, -math.pi/4, math.sqrt(2.0), bearingRangeNoise))
	# iteratively optimize the factor graph and print it out along with the latest result
	fls.update()
	fls._isam.printFactors()
	fls._isam_result.printf()
	print ''

# print out the (private) pose ids in our fixed lag smoother queue
print fls._pose_ids
print ''