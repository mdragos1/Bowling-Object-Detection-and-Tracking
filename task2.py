import cv2
import numpy as np
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import math
    
def get_video_files_from_folder(folder_path, extensions=None):
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    all_files = os.listdir(folder_path)
    video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in extensions]
    video_files = [os.path.join(folder_path, f) for f in video_files]
    
    return video_files

def get_txt_files_from_folder(folder_path):
    all_files = os.listdir(folder_path)
    
    txt_files = [f for f in all_files if f.endswith('.txt')]
    txt_files = [os.path.join(folder_path, f) for f in txt_files]
    
    return txt_files

def read_input_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        second_line = lines[1].strip().split()
        numbers = [int(l) for l in second_line[1:]]
        
        return lines[0], tuple(numbers)

def get_rectangle_area(coords):
    x1, y1, x2, y2 = coords
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height

def find_median_area(area1, area2):
    return (area1 + area2) / 2

def calculate_third_rectangle(rect1, rect2):
    area1 = get_rectangle_area(rect1)
    area2 = get_rectangle_area(rect2)
    
    median_area = find_median_area(area1, area2)
    
    center1 = ((rect1[0] + rect1[2]) / 2, (rect1[1] + rect1[3]) / 2)
    center2 = ((rect2[0] + rect2[2]) / 2, (rect2[1] + rect2[3]) / 2)
    
    # Calculate the direction vector from rect1 to rect2
    direction_vector = np.array([center2[0] - center1[0], center2[1] - center1[1]])
    distance = np.linalg.norm(direction_vector)
    if distance==0:
        distance = 1e-6
    direction_vector /= distance  # Normalize direction vector
    
    center3 = (center2[0] + direction_vector[0] * distance, center2[1] + direction_vector[1] * distance)
    
    width1 = abs(rect1[2] - rect1[0])
    height1 = abs(rect1[3] - rect1[1])
    
    aspect_ratio = width1 / height1
    width3 = np.sqrt(abs(median_area * aspect_ratio))
    height3 = median_area / width3
    try:
        rect3 = (
        int(center3[0] - width3 / 2), int(center3[1] - height3 / 2),
        int(center3[0] + width3 / 2), int(center3[1] + height3 / 2)
        )
    except:
        rect3 = rect2
    
    
    return rect3

def predict(chosen_model, img, classes=[], conf=0.5, iou=0.7):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, iou= iou)
    else:
        results = chosen_model.predict(img, conf=conf, iou= iou)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, iou=0.7):
    results = predict(chosen_model, img, classes, conf=conf, iou= iou)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

def intersection_over_union(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area

    return iou

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        extracted_pos = []
        for line in lines[1:]:
            components = line.strip().split()
            last_four = components[-4:]
            extracted_pos.append(tuple([int(num) for num in last_four]))
        
        return extracted_pos

class PredictionBox:
    def __init__(self, bbox, confidence, cls, iou_with_ancestor):
        self.bbox = bbox
        self.confidence = confidence
        self.cls = cls
        self.iou_with_ancestor = iou_with_ancestor

def track_videos_from_path(path="train/Task2", model = "yolov8x.pt", verbose=0):
    model = YOLO(model)
    videos = get_video_files_from_folder(path)
    start_positions_txt = get_txt_files_from_folder(path)
    i = 1
    for video_path, st_pos in tqdm(zip(videos, start_positions_txt)):
        capture = cv2.VideoCapture(video_path)
        line_0, initial_bbox = read_input_txt(st_pos)
        frames_pred_list = [initial_bbox]

        #Discard the first frame because we already have it 
        ok, frame = capture.read()
        if not ok:
            print('Cannot read the first frame')
            exit()
        consecutive_not_found = 0
        while capture.isOpened():
            success, frame = capture.read()
            pb_list = []
            if success:
                results = predict(model, frame, conf=0.005, iou=0.5, classes=[])
                
                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        bbox = int(x1), int(y1), int(x2), int(y2)
                        confidence = float(box.conf[0])
                        cls = r.names[int(box.cls[0])]
                        iou_with_ancestor = intersection_over_union(frames_pred_list[-1], bbox)

                        pbox = PredictionBox(bbox, confidence, cls, iou_with_ancestor)
                        pb_list.append(pbox)
                
                if consecutive_not_found < 6:
                    pb_list = [pb for pb in pb_list if pb.iou_with_ancestor >= 0.2]
                    pb_list = sorted(pb_list, key= lambda x: (x.iou_with_ancestor), reverse=True)
                else:
                    classes = ['handbag', 'suitcase', 'frisbee', 'sports ball', 'baseball bat', 'baseball glove','skateboard', 'bowl', 'apple', 'donut', 'clock', 'vase']
                    pb_list = [pb for pb in pb_list if pb.cls in classes]
                    pb_list = sorted(pb_list, key= lambda x: (x.confidence), reverse=True)

                if len(pb_list) == 0 and len(frames_pred_list)>1:
                    consecutive_not_found += 1
                    pos_1 = frames_pred_list[-2]
                    pos_2 = frames_pred_list[-1]
                    actual_pos = calculate_third_rectangle(pos_1, pos_2)
                    frames_pred_list.append(actual_pos)
                elif len(pb_list) == 0 and len(frames_pred_list) == 1:
                    frames_pred_list.append(frames_pred_list[-1])
                    consecutive_not_found += 1
                else:
                    frames_pred_list.append(pb_list[0].bbox)
                    consecutive_not_found = 0

                if verbose == 1:
                    for pb in pb_list:
                        cv2.rectangle(frame, (pb.bbox[0], pb.bbox[1]),
                                    (pb.bbox[2], pb.bbox[3]), (255, 0, 0), 2)
                        cv2.putText(frame, f"{pb.cls}",
                                    (pb.bbox[0], pb.bbox[1] - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    cv2.rectangle(frame, (frames_pred_list[-1][0], frames_pred_list[-1][1]),
                            (frames_pred_list[-1][2], frames_pred_list[-1][3]), (35, 178, 200), 2)
                    cv2.imshow("Image", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        capture.release()
        cv2.destroyAllWindows()

        if i<10:
            file_path = os.path.join(f"{os.getcwd()}\\train\\Task2\\predicted-truth", f"0{i}_pt.txt")
        else:
            file_path = os.path.join(f"{os.getcwd()}\\train\\Task2\\predicted-truth", f"{i}_pt.txt")

        ensure_directory_exists(os.path.dirname(file_path))

        with open(file_path, 'w') as file:
            j=1
            file.write(line_0)
            for pos in frames_pred_list:
                file.write(f"{j} {pos[0]} {pos[1]} {pos[2]} {pos[3]}\n")
                j+=1

        i += 1

track_videos_from_path(path='test/Task2',verbose=1)
def get_accuracy(pt, gt):
    total=0
    good=0
    for i in range(len(gt)):
        if intersection_over_union(gt[i], pt[i]) >= 0.3:
            good+=1
        total+=1
    return good/total

predicted = get_txt_files_from_folder("train/Task2/predicted-truth")
ground = get_txt_files_from_folder("train/Task2/ground-truth")
pred_results = []
ground_results = []
acc_list=[]

for filepath in predicted:
    pred_results.append(extract_results(filepath))
for filepath in ground:
    ground_results.append(extract_results(filepath))

for pt,gt in zip(pred_results, ground_results):
    acc = get_accuracy(pt, gt)
    acc_list.append(acc)

print(len([a for a in acc_list if a >= 0.8]))