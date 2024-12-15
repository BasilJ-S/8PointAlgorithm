import pandas as pd
def process_blender_sequence(data: pd.DataFrame, keep_x_rot = False):
    # assumeing the first frame is the first row
    # subtract the initial rotation and translation
    # x,y,z,x_rot,y_rot,z_rot
    initial_position = data.loc[0, ["x","y","z"]]
    initial_rotation = data.loc[0, ["x_rot","y_rot","z_rot"]]
    print(initial_position)
    if keep_x_rot:
        adjusted_data = data.sub([0, initial_position[0], initial_position[1], initial_position[2], 0,initial_rotation[1],initial_rotation[2]])
    else: 
        adjusted_data = data.sub([0, initial_position[0], initial_position[1], initial_position[2], initial_rotation[0], initial_rotation[1], initial_rotation[2]])

    return adjusted_data