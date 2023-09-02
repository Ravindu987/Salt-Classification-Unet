import cv2
import os


def get_frame_from_gif(gifpath):
    cap = cv2.VideoCapture(gifpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    res, frame = cap.read()
    return frame


directory = "./Car/train_masks"
save_directory = "./Car/train_masks_images"

for img_name in os.listdir(directory):
    img_path = os.path.join(directory, img_name)
    frame = get_frame_from_gif(img_path)
    name = img_name.split(".")[0]
    dest_path = os.path.join(save_directory, name + ".jpg")
    print(dest_path)
    cv2.imwrite(dest_path, frame)
