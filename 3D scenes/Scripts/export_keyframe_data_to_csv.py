'''Creates a csv for the active object '''
import bpy
import csv
from pathlib import Path
import os


# === Initialize blender variables ===
# Get scene
scene = bpy.context.scene    
# Get the active object whose' transform you want to export
ob = bpy.context.active_object

# === Initialize file variables ===
filepath = bpy.data.filepath
directory = os.path.dirname(filepath)

filename_stem = Path(filepath).stem
csv_filename = f"{filename_stem}_{ob.name}_loc_and_rot.csv"

output_csv_file = os.path.join(directory , csv_filename)

header_columns = ['frame','x', 'y' 'z', 'x_rot', 'y_rot', 'z_rot']




with open(output_csv_file,'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header_columns)
    
    # Save the current frame to return to after running the script
    saved_frame = scene.frame_current

    for frame in range(scene.frame_start, scene.frame_end + 1):
        # Set the active frame to the desired one
        scene.frame_set(frame)
        
        # get location
        loc = ob.matrix_world.to_translation()

        # get euler rotation of the form (x,y,z)
        rot = ob.matrix_world.to_euler('XYZ')
        
        row = [frame, loc.x, loc.y, loc.z, rot.x, rot.y, rot.z]
        print(row)
        csv_writer.writerow(row)
    
    # Use the saved frame to go back to the frame before the script was run
    scene.frame_set(saved_frame)
        
   