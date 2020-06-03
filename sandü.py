import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import matplotlib
# matplotlib.rcParams['image.cmap'] = 'gray'


dict_data = {}
center_dict = {}

# load the pictures
print("load data from dictionary")
with open("dict_data.bin", "rb") as bin_file:
    dict_data = pickle.load(bin_file)

# load the coordinates of the centers (spiral as well as electodes)
print("load data from dictionary")
with open("dict_results.bin", "rb") as bin_file:
  center_dict = pickle.load(bin_file)
  

for patientID in center_dict.keys():
  print("patient:", patientID)
  
  electrodes = np.array(center_dict[patientID][1])
  n_el = len(electrodes)
  center = np.array(center_dict[patientID][0][0][0][:-1])
  
##################################################################
# delete outermost electrodes if there are > 12 (locally)
##################################################################

  # crop the electrodes that are most distant to the cochlear center if the
  # number of electrodes is bigger than 12

  while n_el > 12:
    print("I'm cropping")
    tmp_dists = []
    
    # calculate the distances of all electrodes to the center
    for i in range(n_el):
      tmp_dists.append(np.linalg.norm(np.array(center-electrodes[i])))
    
    # take index of most distant electrode and delete the electrode
    my_index = np.where(tmp_dists == np.amax(tmp_dists))[0][0]
    electrodes = np.delete(electrodes, my_index, axis=0)
    
    # update the number of electrodes so that the loop will eventually terminate
    n_el = len(electrodes)

##################################################################
# get the order of the electrodes regarding the center distance
##################################################################
  
  # if patientID == "17":
    
  # calculate the distances of all electrodes to the center
  tmp_dists = []
  for i in range(n_el):
    tmp_dists.append(np.linalg.norm(np.array(center-electrodes[i])))
    
  ordered_idx_of_electrodes = []
  for i in range(n_el):
    my_index = np.where(tmp_dists == np.amax(tmp_dists))[0][0]
    ordered_idx_of_electrodes.append(my_index)
    tmp_dists[my_index] = 0
    
  # get the index of the outermost electrode
  # my_index = np.where(tmp_dists == np.amax(tmp_dists))[0][0]
  # print("index", electrodes[my_index])
  print("\nelectrode order:")
  print(ordered_idx_of_electrodes)
  print()


##################################################################
# plot the indexed images
##################################################################
  
  radius_dot = 7
  
  pic_to_plot = dict_data[patientID][1]
  
  pic_to_plot = cv2.cvtColor(pic_to_plot.astype("uint8"), cv2.COLOR_GRAY2BGR)
  # show the cochlear center
  cv2.circle(pic_to_plot, tuple(center), radius_dot, (255,0,0), -1)
  # show the electrode centers
  for e_nbr, electrode in enumerate(electrodes):
    # draw electrodes
    cv2.circle(pic_to_plot, tuple(electrode), radius_dot, (255,255,0), -1)
    cv2.circle(pic_to_plot, tuple(electrode), radius_dot+2, (0,0,255), 2)
    
    # label the electrodes
    plt.text(electrodes[ordered_idx_of_electrodes[e_nbr]][0],
             electrodes[ordered_idx_of_electrodes[e_nbr]][1],
             str(e_nbr), c=(1, 0, 0))
    
  plt.title("Patient: " + patientID)
  plt.imshow(pic_to_plot)
  plt.show()