import numpy as np

class Config(object): 
    def __init__(self, INTENSITIES, TYPE=0, NUM_HEART_BEATS=2, image_resolution=128):
        '''
        Define the Environment parameters of CT setup
        '''
     
        # Perform some sanity checks
        
        assert isinstance(INTENSITIES, np.ndarray), 'INTENSITIES must be a Nx1 numpy array' # each one in N corresponds to an intensity class (which is one organ in the image)
        assert len(INTENSITIES.shape) == 2, 'INTENSITIES must be a Nx1 numpy array'
        assert isinstance(TYPE, int) and TYPE in [0,1,2], 'TYPE must be either 0, 1 or 2'
        assert isinstance(NUM_HEART_BEATS, float) and NUM_HEART_BEATS >= 0 and NUM_HEART_BEATS < 10, 'NUM_HEART_BEATS must be a float between 1 and 10'
#         assert isinstance(NUM_SDFS, int) and NUM_SDFS > 0 and NUM_SDFS < 5, 'NUM_SDFs should be positive integer not more than 5' 
        
        
        self.IMAGE_RESOLUTION = image_resolution             # Resolution of the CT image
        self.GANTRY_VIEWS_PER_ROTATION = 720     # Number of views that the gantry clicks in a single 360 degree rotation
        self.HEART_BEAT_PERIOD = 1000            # Time (ms) it takes the heart to beat once
        self.GANTRY_ROTATION_PERIOD = 275        # Time (ms) it takes for the gantry to complete a single 360 degree rotation
        self.NUM_HEART_BEATS = NUM_HEART_BEATS   # Number of heart beats during the time HEART_BEAT_PERIOD
        self.INTENSITIES = INTENSITIES
        self.TYPE = TYPE
        '''
        NOTE: In the current setup, all of motion happens within the period HEART_BEAT_PERIOD. In case there are N hearbeats, then the time period of each heart beat is taken as HEART_BEAT_PERIOD/N. 
        '''
        
        '''
        Parameters for defining experimental setup
        '''
        if self.TYPE==0:
        # To run gantry for a single 360 degree rotation
            # explanation of math:
            # self.GANTRY2HEART_SCALE = (self.NUM_HEART_BEATS/self.THETA_MAX)*(self.GANTRY_ROTATION_PERIOD/self.HEART_BEAT_PERIOD)
            # phase_in_heart_motion = self.GANTRY2HEART_SCALE * t

            # so:
            # phase_in_heart_motion = (t / theta_max) * (gantry_rotation_period/heart_beat_period) * number_heart_beats
            # assume t = 180, theta_max = 360, it means now we have half gantry rotation.
            # then we need to convert gantry period to heart period, so we have gantry_rotation_period/heart_beat_period, if it's 500/1000,
            # then we know each gantry rotation = half heart beat period, so half gantry rotation = 1/4 heart beat peirod
            # In each heart period we have two heatbeats
            # so 1/4 heart beat period = 1/2 heartbeat = phase 1/2
            
            self.TOTAL_CLICKS = self.GANTRY_VIEWS_PER_ROTATION
            self.THETA_MAX = 360
            self.GANTRY2HEART_SCALE = (self.NUM_HEART_BEATS/self.THETA_MAX)*(self.GANTRY_ROTATION_PERIOD/self.HEART_BEAT_PERIOD) 

        elif self.TYPE==1:
        # Otherwise, to run gantry for a single heart beat
            # heart_beat_period/num_heart_beat = time_for_one_heart_beat,
            # time_for_one_heart/gantry_rotation_period = how_many_gantry_rotation_needed
            # how_many_gantry_rotation_needed * gantry_views_per_rotation = total_clicks
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS) 
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD)/self.NUM_HEART_BEATS)
            self.GANTRY2HEART_SCALE = 1/(self.THETA_MAX)   
        
        elif self.TYPE==2:
        # Lastly, if you wish to run gantry to capture N heart beats then
            self.TOTAL_CLICKS = int(self.GANTRY_VIEWS_PER_ROTATION * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.THETA_MAX = int(360 * (self.HEART_BEAT_PERIOD/self.GANTRY_ROTATION_PERIOD))
            self.GANTRY2HEART_SCALE = self.NUM_HEART_BEATS/(self.THETA_MAX)
        
        '''
        NeuralCT Hyper parameters
        '''
        self.SDF_SCALING = self.IMAGE_RESOLUTION/1.414  # Factor to scale NeuralCT's output to match G.T. SDF range of values
        self.BATCH_SIZE=25                       # Number of projections used in a single training iterations
        self.NUM_SDFS = self.INTENSITIES.shape[1]