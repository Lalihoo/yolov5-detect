import cv2
from utils.detector import Detector

detector = Detector("weights", "best", 0.45)


def predict(name):
    """
    图片检测
    """
    img = cv2.imread(f'./images/{name}')
    boxes = detector.detect(img)
    img = detector.plot_boxes(img, boxes)
    cv2.imwrite(f'./images/predict/{name}', img)
    return boxes


def camera():
    """
    摄像头检测
    """
    # video = cv2.VideoCapture(0)
    path = './video/2.mp4'
    video = cv2.VideoCapture(path)
    while video.isOpened():
        res, frame = video.read()
        if frame is None:
            break
        if res:
            boxes = detector.detect(frame)
            img = detector.plot_boxes(frame, boxes)
            cv2.imshow("result", img)
            res = []
            for box in boxes:
                res.append(f'{box["name"]}-{box["confidence"]}')
            print(res)
            if cv2.waitKey(10) & 0xFF == 27:
                break

    video.release()
    cv2.destroyAllWindows()


# predict('000000000049.jpg')
camera()
