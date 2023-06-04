import cv2

def append_slash_to_dirpath_if_not_present(dir_path):
    # make sure output_dir endswith "/"
    if not dir_path.endswith("/"):
        dir_path += "/"
    return dir_path

def resize(img, new_sidedims):
    new_sidedims = (new_sidedims[1], new_sidedims[0])
    return cv2.resize(img, new_sidedims)