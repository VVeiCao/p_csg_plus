import os
import shutil

def get_directory_names(directory):
    directories = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            directories.append(item)
    return directories

# Example usage:
directories = ['carla_data/clear-weather/data', 'carla_data/weather-mix/data']
for directory in directories:
    all_sub_directory = get_directory_names(directory)
    for d in all_sub_directory:
        split_parts = d.split('_')
        town_name = split_parts[1]
        town_name[0] = 'T'
        d_name = town_name + '_' + split_parts[2]
        destination = os.path.join(directory, d_name)
        if not os.path.exists(destination):
            os.mkdir(destination)
        shutil.move(os.path.join(directory, d), destination)