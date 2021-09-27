# 模型

> coco128 模型名称

+ 存放位置 `weights\coco128.pt`
+ `weights\hubconf.py`中新增对应的方法
    ```python
    def coco128(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
        return _create('coco128', pretrained, channels, classes, autoshape, verbose, device)
    ```

# 调用

```python
import cv2
from utils.detector import Detector

detector = Detector("weights", "coco128", 0.45)

img = cv2.imread(f'./images/1.png')
# 检测
boxes = detector.detect(img)
# 画框
img = detector.plot_boxes(img, boxes)
```