""" This file defines utilities for the ROS agents. """



baxter_joint_names = ['_s0','_s1','_e0','_e1','_w0','_w1','_w2']

class GPSBaxterInterface(object):
    def __init__(self):
        self.right_arm = Limb.limb('right')
        self.left_arm = Limb.limb('left')
        self.limbs = {'right': self.right_arm, 'left': self.left_arm}

    def set_baxter_joint_angles(self, limb, joint_angles):
        """
        limb: String 'left' or 'right'
        joint_angles: dict of joint_name:angle
        """
        assert limb in ['left', 'right']
        self.limbs[limb].move_to_joint_positions(joint_angles)

    def set_baxter_joint_efforts(self, limb, joint_efforts):
        """
        limb: String 'left' or 'right'
        joint_efforts: dict of joint_name:torques
        """
        assert limb in ['left', 'right']
        self.limbs[limb].set_joint_torques(joint_efforts)

    def get_joint_angles(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns dict of joint_name:angle
        """
        return self.limbs[limb].joint_angles()

    def get_joint_efforts(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns dict of joint_name:torque
        """
        return self.limbs[limb].joint_efforts()

    def get_ee_pose(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns the cartesian endpoint pose (x, y, z)
        """
        return self.limbs[limb].endpoint_pose()['position']

    def get_ee_quaternion(self, limb):
        """
        limb: String 'left' or 'right'
        ------
        Returns the cartesian endpoint quaternion (x, y, z, w)
        """
        return self.limbs[limb].endpoint_pose()['orientation']

    def map_values_to_joints(self, limb, values):
        """
        limb: String 'left' or 'right'
        values: List of numerical values to map to joints using above ordering
        -----
        Returns a dictionary with joint_name:value
        """
        assert len(values) == 7
        mapping = {}
        for i in range(len(baxter_joint_names)):
            mapping[limb+baxter_joint_names[i]] = values[i]

        return mapping
