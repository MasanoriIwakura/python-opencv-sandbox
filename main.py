import cv2
import os

IMAGE_SIZE = (200, 200)


def init_image(path):
    detector = cv2.AKAZE_create()
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMAGE_SIZE)
    return detector.detectAndCompute(image, None)


def compare(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    return [m.distance for m in matches]


if __name__ == "__main__":

    # オリジナルファイルの情報を取得
    original_kp, original_des = init_image("./images/cat-original.jpg")
    ret = compare(original_des, original_des)
    print("オリジナル：{0}".format(sum(ret) / len(ret)))

    # オリジナルとサイズ違いの比較
    size_kp, size_des = init_image("./images/cat-small.jpg")
    ret = compare(original_des, size_des)
    print("サイズ違い：{0}".format(sum(ret) / len(ret)))

    # オリジナルと色違いの比較
    color_kp, color_des = init_image("./images/cat-color.jpg")
    ret = compare(original_des, color_des)
    print("色違い：{0}".format(sum(ret) / len(ret)))

    # オリジナルと画質違いの比較
    quality_kp, quality_des = init_image("./images/cat-quality.jpg")
    ret = compare(original_des, quality_des)
    print("画質違い：{0}".format(sum(ret) / len(ret)))

    # 全く異なる画像との比較
    different_kp, different_des = init_image("./images/cat-different.jpg")
    ret = compare(original_des, different_des)
    print("全く異なる画像：{0}".format(sum(ret) / len(ret)))
