from ultralytics import YOLO

model = YOLO("best.pt")

source = "checking"

# Run inference on the source
results = model(source,save=True)  # list of Results objects