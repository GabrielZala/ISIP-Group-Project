import cv2
import tools.data_manager as data_manager
import tools.image_utils as img
from matplotlib import pyplot as plt
import numpy as np



dict_data = data_manager.read_pictures()

print("processing images")
lst_binary_preprocessed=img.calculate_binaries(dict_data)
lst_cropped_binaries = img.crop_binaries(lst_binary_preprocessed,270)
lst_individual_erosion = [img.individual_erosion(i) for i in lst_cropped_binaries]
dict_of_electrode_centers = img.get_center_of_electrodes(lst_individual_erosion)
plot_electrode_centers = True
if plot_electrode_centers:
    counter = 1
    for patient in dict_data:
        coords = dict_of_electrode_centers[counter]
        counter += 1
        final_map = np.zeros((723,1129))
        for coordinate in coords:
            x = coordinate[0]
            y = coordinate[1]
            cv2.circle(dict_data[patient][1],(x,y),7,(0,0,255),-1)
        #plt.imshow(dict_data[patient][1])
        #plt.show()
            

#cropp more away from the lst_cropped_binaries and after that try to fit circle with houghtransform
# to get the center of the cochlea.
test_lst=img.crop_binaries(lst_binary_preprocessed, 430)
dict_binaries = {}
for patient in enumerate(dict_data):
    dict_binaries[patient[1]]=test_lst[patient[0]]
    
hough_circle_detection = True
if hough_circle_detection:
    spiral_centres = {}
    
    for patient in dict_binaries:
        
        image = dict_binaries[patient].astype("uint8")
        image = cv2.GaussianBlur(image, (5,5), 10)
        try:
            circles_image = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 3, 100000, 
                                                       param1=50, param2=30, minRadius=80, 
                                                       maxRadius=180)
            spiral_centres[patient] = circles_image
            
        except:
            print("no circle found")
            pass

    # for i in spiral_centres:  # use arrowkeys to go through the images
    #     img.circles_show(dict_binaries[i], spiral_centres[i])

# merge both dictionaries to have one with both spiral centres and electrode centres for each patient
lst_electrodes = []
for i in dict_of_electrode_centers:
    lst_electrodes.append(dict_of_electrode_centers[i])
    
center_dict = {}
for i in enumerate(spiral_centres):
    center_dict[i[1]]=[spiral_centres[i[1]],lst_electrodes[i[0]]]
        

##################################################################
# start the loop over all patients
##################################################################

for patientID in center_dict.keys():
  print("\n### patient:", patientID, "###")
  
  electrodes = np.array(center_dict[patientID][1])
  n_el = len(electrodes)
  center = np.array(center_dict[patientID][0][0][0][:-1])
  
##################################################################
# delete outermost electrodes if there are > 12
##################################################################

  # crop the electrodes that are most distant to the cochlear center if the
  # number of electrodes is bigger than 12
  while n_el > 12:
    #print("I'm cropping")
    tmp_dists = []
    
    # calculate the distances of all electrodes to the center
    for i in range(n_el):
      tmp_dists.append(np.linalg.norm(np.array(center-electrodes[i])))
    
    # take index of most distant electrode and delete the electrode
    my_index = np.where(tmp_dists == np.amax(tmp_dists))[0][0]
    electrodes = np.delete(electrodes, my_index, axis=0)
    
    # update the number of electrodes so that the loop will eventually terminate
    n_el = len(electrodes)

  # update the center_dict to hold only the 12 remaining electrodes
  center_dict[patientID][1] = electrodes.tolist()


##################################################################
# get the order of the electrodes regarding the center distance
##################################################################

  # calculate the distances of all electrodes to the center
  tmp_dists = []
  for i in range(n_el):
    tmp_dists.append(np.linalg.norm(np.array(center-electrodes[i])))
    
  ordered_idx_of_electrodes = []
  for i in range(n_el):
    my_index = np.where(tmp_dists == np.amax(tmp_dists))[0][0]
    ordered_idx_of_electrodes.append(my_index)
    tmp_dists[my_index] = 0
    
  #print("electrode order:", ordered_idx_of_electrodes)


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
  #plt.savefig(patientID+'jpg')
  plt.show()
   

##################################################################
# calculate the angles by the given order of the electrodes
##################################################################
  
  # store the cummulated angles in a list (going from outside to inside)
  angle_lst = []
  print("############################################")
  print("center:", center)
  print("x,    y,    angle")
  for i in range(n_el):
      if i == 0:
          print(electrodes[0], "0")
      elif i == 1: # store the first angle as it is
          angle = img.get_angle(center, (electrodes[i-1], electrodes[i]))
          angle_lst.append(angle)
          print(electrodes[i], int(round(angle, 0)))
      else: # store the following angles as sums
          angle = angle_lst[-1] + img.get_angle(center, (electrodes[i-1], electrodes[i]))
          angle_lst.append(angle)
          print(electrodes[i], int(round(angle, 0)))
  print("############################################")


