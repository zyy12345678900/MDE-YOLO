from ultralytics import YOLO  as YOLO
import warnings
warnings.filterwarnings('ignore')

model_yaml_path = r'ultralytics/cfg/models/v11/mde-yolo.yaml'

data_yaml_path = r'pothole.yaml'
if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=200,
                          batch=16,
                          workers=8,
                          amp=False,  
                          project='runs/train',
                          name='exp',
                          )
