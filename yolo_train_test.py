import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Cracks-main/output/model/data.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--weights', default='yolov8n.pt')
    parser.add_argument('--project', default='Cracks-main/output/yolo')
    parser.add_argument('--name', default='train')
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)

    # Validate
    model.val(data=args.data, imgsz=args.imgsz, project=args.project, name=args.name+'_val')

    # Predict on validation images
    val_dir = Path('Cracks-main/output/dataset/images/val')
    if val_dir.exists():
        model.predict(source=str(val_dir), save=True, save_txt=True, conf=0.25, imgsz=args.imgsz, project=args.project, name=args.name+'_preds')


if __name__ == '__main__':
    main()
