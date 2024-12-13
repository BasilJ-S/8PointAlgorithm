'''
from the console enter this to find the pip install target--target
target = bpy.utils.user_resource("SCRIPTS", path="modules")
'''
#import sys, subprocess
#subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas', '-t', 'C:\\Users\\Joey\\AppData\\Roaming\\Blender Foundation\\Blender\\4.3\\scripts\\modules'])
#import sys
#sys.path.append(target)



from pathlib import Path
import bpy
import pandas as pd
import os


# Get scene
scene = bpy.context.scene    
# Get the active object whose' transform you want to export
ob = bpy.context.active_object


# === Initialize file variables ===
filepath = bpy.data.filepath
directory = os.path.dirname(filepath)

csv_filename = "Comp558 simple scene translation_Camera_loc_and_rot.csv"

# Can change this to any file
input_csv_file = os.path.join(directory , csv_filename)

# Read specific columns from CSV file
header_columns = ['frame','x', 'y', 'z', 'x_rot', 'y_rot', 'z_rot']  # Replace with your column names
df = pd.read_csv(input_csv_file, usecols=header_columns)


for row in df.itertuples():

    # set location
    loc = [row.x, row.y, row.z]
    ob.matrix_world.translation = loc # global
    
    # set rotation
    rot = [row.x_rot, row.y_rot, row.z_rot]
    ob.rotation_euler = rot
    
    frame = row.frame
    # insert all location keyframes  
    ob.keyframe_insert("location", frame=frame)
    ob.keyframe_insert("rotation_euler", frame=frame)