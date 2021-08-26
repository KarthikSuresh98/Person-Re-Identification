import numpy as np
import pandas as pd
import cv2
import os

df = pd.read_csv('detected_boxes_details_with_id.csv' , converters = {'bounding_boxes' : eval})

frames_loc = 'video-frames/frames/'
frames_with_id_loc = 'video-frames/frames-with-id/'
images = os.listdir(frames_loc)

for filename in images:

    frame_name = frames_loc + filename
    print(filename)
    frame = cv2.imread(frame_name)
    sub_df = df[df['frame_names'] == frame_name]
    bounding_boxes = list(sub_df['bounding_boxes'])
    person_id = list(sub_df['person_id'])
   
    for i in range(len(person_id)):
        bounding_box = bounding_boxes[i]
        x_start = bounding_box[0]
        x_end = bounding_box[1]
        y_start = bounding_box[2]
        y_end = bounding_box[3]

        cv2.rectangle(frame, (y_start, x_start), (y_end, x_end), (255, 255, 255), 2)
        cv2.putText(frame, "ID: " + str(person_id[i]), (y_start, x_start), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)

    frame_id_name = frames_with_id_loc + filename 
    cv2.imwrite(frame_id_name , frame)

