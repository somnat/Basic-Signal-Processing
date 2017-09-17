import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
# This function computes the number of samples in a wave file of given size in milliseconds.
def calSample(size,samp_fre):
   
   print "Size means the len of wav file in milliseconds" 
   print "Sampling frequency"
   num_sample=(size*samp_fre)/1000
   return num_sample

#This function computes the number of frames in a signal where the frame size and frame shift is given
def calNumFrame(frame_size,frame_shift):
   print "frame_size means the frame size on for which you want to compute the features, generally we keep frame size between 20-30 ms"
   print "frame shift generally 5-10 ms"
   N=calSample(50,16000)
   t=frame_size*16
   m=frame_shift*16
   Numframe=np.ceil((N-t)/m)
   return Numframe

print calNumFrame(30,10)

   
   



