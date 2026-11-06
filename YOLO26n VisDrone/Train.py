from ultralytics import YOLO

def main():
    # Load a YOLO26n model. (Original is coco trained)
    model = YOLO("yolo26n.pt")

    # Source Objects365 v2 name (lowercased) -> target COCO name
    ALIASES = {
        "people": "person",

    }
    # model.model.names = {i: ALIASES.get(name.lower(), name) for i, name in model.model.names.items()}
    # # Train the model on the datases6t for 100 epochs
    results = model.train(
        data="VisDrone.yaml",  # Path to your dataset configuration
        epochs=100,                        # Number of training epochs
        imgsz=800,                         # Input image size
        batch=16,                          # Batch size, defaults to 16 in CPU training
        device=0,                      # GPU device ID (use 'cpu' for CPU training)
        name="LapTopComputated",            # Name for the results folder
        patience=5,                         # if no improvement happens in 5 epochs. stops the training
        cache='true',                       # uses ram to store data for faster processing
        scale=0.85,                           # simulates distance by scaling the image randomly 50%-150% of its size.
        optimizer='auto',   
        max_det=300,                        # default is 300 and it complains about it when run, not sure  
    )

if __name__ == "__main__":
    main()