import torch
import cv2
import json
import random


class Detector():
    def __init__(self, repo_or_dir='weights', weights='coco128', conf_thres=0.55):
        """
        初始化

        Parameters
        ----------
        repo_or_dir: 本地模型路径
        weights: 模型名称，无需后缀
        """
        self.model = torch.hub.load(source="local", repo_or_dir=repo_or_dir, model=weights, force_reload=True)
        self.conf_thres = conf_thres
        self.names = self.model.model.names if hasattr(
            self.model, 'module'
        ) else self.model.names
        self.colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in self.names
        ]

    def detect(self, image, size=640):
        """
        图片检测

        Parameters
        ----------
        image: 图片
        size: 图片大小，640/1280
        """
        pred = self.model(image)
        boxes = pred.pandas().xyxy[0].to_json(orient="records")
        boxes = json.loads(boxes)
        boxes = [box for box in boxes if box['confidence'] > self.conf_thres]
        return boxes

    def plot_boxes(self, image, boxes):
        """
        画框

        Parameters
        ----------
        image: 图片
        boxes: 框集合
        """
        thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        for item in boxes:
            c1 = (int(item['xmin']), int(item['ymin']))
            c2 = (int(item['xmax']), int(item['ymax']))
            confidence, class_id, name = item['confidence'], item['class'], item['name']
            color = self.colors[class_id]
            cv2.rectangle(image, c1, c2, color, thickness=thickness, lineType=cv2.LINE_AA)
            tf = max(thickness - 1, 1)
            t_size = cv2.getTextSize(name, 0, fontScale=thickness / 3, thickness=tf)[0]

            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(image, '{}-{:.2f}'.format(name, confidence), (c1[0], c1[1] - 2), 0, thickness / 3,
                        [255, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)

        return image
