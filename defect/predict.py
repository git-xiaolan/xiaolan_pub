from pathlib import Path
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8.yaml')  # build a new model from YAML
# model = YOLO(r'D:\CodeWorkSpace\Defect-Detection\ultralytics-main2\yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights
model = YOLO(r'D:\CodeWorkSpace\Defect-Detection\ultralytics-main2\weights\train2_345all_50epoch_weights\best.pt')  # load a pretrained model (recommended for training)

# # 测试图片文件路径
# folder_path = r'D:\CodeWorkSpace\Defect-Detection\ultralytics-main2\runs\演示\FLIR0254.jpg'
#
# # 获取该文件夹下测试图片路径（jpg格式）
# image_files = [str(f) for f in Path(folder_path).glob('**/*.jpg')]  # 也可以包含其他格式，如 .png 等
image_files = r'D:\CodeWorkSpace\Defect-Detection\ultralytics-main2\photo_examples\FLIR0254.jpg'  # 也可以包含其他格式，如 .png 等

# Train the model
if __name__ == '__main__':

    # 模型预测，save=True 的时候表示直接保存yolov8的预测结果
    model.predict(image_files, save=True)



