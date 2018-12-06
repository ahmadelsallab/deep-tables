import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd


def validate(img):
    img = img - np.min(img)  # zero image
    img = img / np.max(img)  # 0.0 ~ 1.0
    return (img * 255).astype(np.uint8)


def preprocess_image(img, is_validate):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)

    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))
    if is_validate:
        return validate(transformed_image)
    else:
        return transformed_image


if __name__ == "main":

    params_parser = argparse.ArgumentParser(description='Data preprocessing')
    params_parser.add_argument('--validate', action="store_true")

    params = params_parser.parse_args()
    print("Validate:", params.validate)

    root_dir = './'
    file_list = ['train.csv', 'val.csv']
    image_source_dir = os.path.join(root_dir, 'data/images/')
    data_root = os.path.join(root_dir, 'data')

    logs = open("logs.txt", "w")

    #######################################################################
    import multiprocessing
    from multiprocessing import Pool

    print("=" * 50, file=logs, flush=True)
    start = time.time()
    for file in file_list:
        image_target_dir = os.path.join(data_root, file.split(".")[0])  # + "-multi"
        os.mkdir(image_target_dir)
        print(image_target_dir, "directory created.", file=logs, flush=True)
        # read list of image files to process from file
        image_list = pd.read_csv(os.path.join(data_root, file), header=None)[0]


        def worker(image):  # all params
            global image_source_dir
            global logs
            global image_target_dir
            """thread worker function"""
            # open image file
            img = cv2.imread(os.path.join(image_source_dir, image))

            transformed_image = preprocess_image(img, validate=params.validate)
            target_file = os.path.join(image_target_dir, image)
            print("Writing target file {}".format(target_file), file=logs, flush=True)
            cv2.imwrite(target_file, transformed_image)


        print("Start preprocessing images", file=logs, flush=True)
        workers = Pool(multiprocessing.cpu_count())
        print("Number images", image_target_dir, len(image_list[1:]))
        workers.imap_unordered(worker, image_list[1:], chunksize=multiprocessing.cpu_count())
        workers.close()
        workers.join()

    print("multi", time.time() - start, file=logs, flush=True)
    print("multi", time.time() - start)

    ######################################################################
    # print("=" * 50, file=logs, flush=True)
    # start = time.time()
    # for file in file_list:
    #
    #     image_target_dir = os.path.join(data_root, file.split(".")[0])
    #     os.mkdir(image_target_dir)
    #     print(image_target_dir, "directory created.", file=logs, flush=True)
    #     # read list of image files to process from file
    #     image_list = pd.read_csv(os.path.join(data_root, file), header=None)[0]
    #
    #     print("Start preprocessing images", file=logs, flush=True)
    #     for image in image_list[1:]:
    #         # open image file
    #         print(os.path.join(image_source_dir, image))
    #         img = cv2.imread(os.path.join(image_source_dir, image))
    #         transformed_image = preprocess_image(img)
    #         target_file = os.path.join(image_target_dir, image)
    #         print("Writing target file {}".format(target_file), file=logs, flush=True)
    #         cv2.imwrite(target_file, transformed_image)
    #
    # print("Sequential", time.time() - start, file=logs, flush=True)
    # print("Sequential", time.time() - start)
