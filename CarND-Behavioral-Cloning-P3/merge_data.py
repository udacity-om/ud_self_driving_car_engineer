import numpy as np
import pandas as pd
import os
import shutil

data_input_folder = 'my_training_data'
scenarios = ['Given_Data', 'Track1_Four_Laps_One_Direction', 'Track1_Two_Laps_Other_Direction', 'Track1_Two_Laps_Other_Direction_Recover_To_Center']
data_output_folder = 'data'
img_folder_name = 'IMG'
csv_file_name = 'driving_log.csv'

# Read in data
dataframe_list = []
for scenario in range(len(scenarios)):
   data_folder_path = data_input_folder + '/' + scenarios[scenario] + '/'
   full_path = data_folder_path + csv_file_name
   print(full_path)
   df = pd.read_csv(full_path)
   #print(df.head())
   dataframe_list.append(df.as_matrix())
   # img_folder_path = data_folder_path + img_folder_name + '/'
   # print(img_folder_path)
   # source = os.listdir(img_folder_path)
   # destination = data_output_folder + '/' + img_folder_name + '/'
   # print(destination)
   # for files in source:
      # #print(files)
      # file_path = img_folder_path + files
      # shutil.copy(file_path,destination)
   
combined_dataframes_list = np.vstack(dataframe_list)
final_dataframe = pd.DataFrame(combined_dataframes_list)
final_dataframe.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

# for path_name_index in final_dataframe['center']:
   # print(path_name_index)
   # path, file_name = os.path.split(path_name_index)
   # dir_name = os.path.split(path)[-1]
   # path_name_index = dir_name + '/' + file_name
   
# first_path = final_dataframe['center'][0]
# print(first_path)
# dir_name = os.path.split(path1)[-1]
# req_file_path = dir_name + '/' + file_name
# print(req_file_path)
   
num_of_images = final_dataframe.shape[0]
#print('Shape of Final Dataframe:', final_dataframe.shape)
#print('Number of Images in Final Dataframe:', num_of_images)
#print(final_dataframe.tail())

final_dataframe.to_csv(path_or_buf = data_output_folder + '/' + csv_file_name, index = False)
