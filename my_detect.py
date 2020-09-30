from utils.datasets import letterbox
from utils.utils import *

cap = cv2.VideoCapture(0)
# cap.set(3, 800)  
# cap.set(4, 600)

with torch.no_grad(): 
    device = torch_utils.select_device('cpu') # cpu, 0
    torch.backends.cudnn.benchmark = True 

    # Load model
    weights = './weights/best5s.pt' # link to  model in ./weights
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()
    names = model.names if hasattr(model, 'names') else model.modules.names #['OK', 'NG_pallet', 'NG_wood','CCL1'] 
    red,green = [0,0,255],[0,255,0]
    colors = [green,red,red,green,green]

    while(True):
        # time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(800,600))
            # Cropped processed frame
            process_frame = frame[200:500,20:780]
            
            # Preprocess
            img = letterbox(process_frame, new_shape=640)[0]
            img = preprocess(img,device)

            # Predict
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.8, 0.5,
                                   fast=True, classes=None, agnostic=False)

            draw_border(frame,(20,200),(780,500),30,th=4)
            # print(pred)
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], process_frame.shape).round()
                    for *xyxy, conf, cls_ in det:
                        index_result = int(cls_)
                        label = '%s %.2f' % (names[index_result], conf)  # Pallet type
                        xyxy = [int(xyxy[0]+20), int(xyxy[1]+200), int(xyxy[2]+20), int(xyxy[3]+200)]
                        distance = plot_one_box(xyxy, frame, label=label, color=colors[index_result], line_thickness=2) 
                        print(label, xyxy, distance)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            cv2.imshow('frame',frame)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()