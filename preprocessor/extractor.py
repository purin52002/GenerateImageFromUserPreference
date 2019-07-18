from face_recognition import face_landmarks, face_locations
import cv2
import numpy as np


def _normalize_landmrk(landmark: dict):
    left_eye_list = landmark['left_eye']
    left_eye_length = len(left_eye_list)
    e0_x = sum([eye[0] for eye in left_eye_list])/left_eye_length
    e0_y = sum([eye[1] for eye in left_eye_list])/left_eye_length

    right_eye_list = landmark['right_eye']
    right_eye_length = len(right_eye_list)
    e1_x = sum([eye[0] for eye in right_eye_list])/right_eye_length
    e1_y = sum([eye[1] for eye in right_eye_list])/right_eye_length

    e0 = np.array([e0_x, e0_y])
    e1 = np.array([e1_x, e1_y])

    bottom_lip_list = landmark['bottom_lip']
    top_lip_list = landmark['top_lip']

    left_bottom_lip = min(bottom_lip_list, key=lambda x: x[0])
    right_bottom_lip = max(bottom_lip_list, key=lambda x: x[0])

    left_top_lip = min(top_lip_list, key=lambda x: x[0])
    right_top_lip = max(top_lip_list, key=lambda x: x[0])

    left_lip = np.array(
        min([left_bottom_lip, left_top_lip], key=lambda x: x[0]))
    right_lip = np.array(
        max([right_bottom_lip, right_top_lip], key=lambda x: x[0]))

    x_ = e1-e0

    eye_mean = 0.5*(e0+e1)
    lip_mean = 0.5*(left_lip+right_lip)

    y_ = eye_mean - lip_mean

    c = eye_mean - 0.1*y_

    amped_x_ = 4*np.abs(x_)
    amped_y_ = 3.6*np.abs(y_)
    s = np.maximum(amped_x_, amped_y_)

    rot90 = np.array([[0, -1], [1, 0]])

    rot_y_ = np.dot(rot90, y_)

    x_dest = x_ - rot_y_
    norm = np.linalg.norm(x_dest)
    x = x_dest/norm

    y = np.dot(rot90, x)

    return c, s, x, y


def extract_face(image):
    location = face_locations(image)
    landmark_list = face_landmarks(image, location)

    extract_list = []

    for landmark in landmark_list:
        c, s, x, y = _normalize_landmrk(landmark)

        top_left = (c - 0.5*s).astype(int)
        top_left = np.maximum(top_left, 0)

        bottom_right = (c + 0.5*s).astype(int)

        extract_list.append(
            image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])

    return extract_list


def _extract_test():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()

    parser.add_argument('-i', '--image_path', required=True)
    parser.add_argument('-o', '--extract_dir_path', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError

    dest_path = Path(args.extract_dir_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    extract_list = extract_face(image)
    print(f'extract: {image_path}')

    for i, extract in enumerate(extract_list):
        cv2.imwrite(
            str(dest_path/(('%i.' % (i+1))+image_path.name)), extract)


if __name__ == "__main__":
    _extract_test()
