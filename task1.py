import cv2
import os
import cv2
from ultralytics import YOLO

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    return images

def load_queries(directory):
    queries = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                # Read all lines and strip newline characters
                query = [int(line.strip()) for line in file]
            queries.append((query[0],query[1:]))
    return queries

def load_results(directory):
    queries = []
    for filename in os.listdir(directory):
        if filename.endswith("t.txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                query = [line.strip() for line in file]
            queries.append(" | ".join(query[1:]))
    return queries

def load_lane_pins(directory):
    pins_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                pins = [line.strip().split(',') for line in file]
                pins = [tuple([int(num) for num in sublist]) for sublist in pins]
            pins_list.append(pins)
    return pins_list

def display_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

def get_sorted_pins_position(model, image, classes=[], conf=0.5, verbose=0):
    pin_pos = []
    if verbose == 1:
        img,results = predict_and_detect(model, image, classes, conf)
        display_image(img)
    else:
        results = predict(model, image, classes, conf)
    for result in results:
        for box in result.boxes:
            left, top, right, buttom = box.xyxy[0]
            pin_pos.append((int(left), int(top), int(right), int(buttom)))
    return sorted(pin_pos, key=lambda x: (x[3], x[0]), reverse=True)

def template_matching(image, template):
    # Apply template matching
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    # Get the best match position
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc

def classify_lane(image, lane1_template, lane2_template, lane3_template, lane4_template):
    max_val1, max_loc1 = template_matching(image, lane1_template)
    max_val2, max_loc2 = template_matching(image, lane2_template)
    max_val3, max_loc3 = template_matching(image, lane3_template)
    max_val4, max_loc4 = template_matching(image, lane4_template)

    # print(f"Matching score for lane 1: {max_val1}")
    # print(f"Matching score for lane 2: {max_val2}")
    # print(f"Matching score for lane 3: {max_val3}")
    # print(f"Matching score for lane 4: {max_val4}")

    max_val = max(max_val1, max_val2, max_val3, max_val4)

    # Determine which lane the photo belongs to
    if max_val1 == max_val:
        return 1, max_val1, max_loc1
    elif max_val2 == max_val:
        return 2, max_val2, max_loc2
    elif max_val3 == max_val:
        return 3, max_val3, max_loc3
    elif max_val4 == max_val:
        return 4, max_val4, max_loc4
    
def rectangle_area(rect):
    xmin, ymin, xmax, ymax = rect
    return max(0, xmax - xmin) * max(0, ymax - ymin)

def intersection_area(rect1, rect2):
    xmin1, ymin1, xmax1, ymax1 = rect1
    xmin2, ymin2, xmax2, ymax2 = rect2

    # Calculate the coordinates of the intersection rectangle
    ixmin = max(xmin1, xmin2)
    iymin = max(ymin1, ymin2)
    ixmax = min(xmax1, xmax2)
    iymax = min(ymax1, ymax2)

    # Compute the width and height of the intersection rectangle
    iw = max(0, ixmax - ixmin)
    ih = max(0, iymax - iymin)

    # Return the area of the intersection rectangle
    return iw * ih

def find_best_matching_rectangle(target_rect, rectangles):
    max_intersection_area = 0
    best_match = (0,0,0,0)

    for rect in rectangles:
        area = intersection_area(target_rect, rect)
        if area > max_intersection_area:
            max_intersection_area = area
            best_match = rect

    target_area = rectangle_area(target_rect)

    if target_area == 0:
        return best_match, max_intersection_area, 0

    shared_percentage = (max_intersection_area / target_area) * 100

    lambda_buttom = abs(target_rect[3] - best_match[3])
    return best_match, max_intersection_area, shared_percentage, lambda_buttom

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_accuracy(predicted_truths, ground_truths):
    sum = 0
    for prediction,truth in zip(predicted_truths, ground_truths):
        if prediction == truth:
            sum += 1
    return sum / len(predicted_truths)





train_images = load_images('test/Task1')
train_queries = load_queries('test/Task1')
lane_images = load_images('test/Task1/full-configuration-templates')
lane_pins = load_lane_pins('test/Task1/full-configuration-templates')

model = YOLO("yolov8x.pt")

i = 1
for image, query in zip(train_images, train_queries):
    #Find from which lane was the image taken from.
    lane, _, _ = classify_lane(image, lane_images[0], lane_images[1], lane_images[2], lane_images[3])

    #Get the full pins positions for the specific lane.
    full_pins_pos = lane_pins[lane-1]

    #Detect the pins for the current image.
    pins_pos = get_sorted_pins_position(model, image, classes=[39,75], conf=0.02, verbose=0)

    detection_list = []
    for input_pin in query[1]:
        #For each original input pin positon in the lane find the one in the image with the best_match over surface and position.
        input_pin_pos = full_pins_pos[input_pin-1]
        best_match, max_intersection_area, shared_percentage, lambda_buttom = find_best_matching_rectangle(input_pin_pos, pins_pos)
        
        #For a detected pin to be valid it needs to share 25-30% of the same surface as the original
        #and the offset between buttom should be smaller then 10 pixels.
        if shared_percentage > 27:  #and lambda_buttom < 10:
            detection_list.append(1)
        else:
            detection_list.append(0)

    if i<10:
        file_path = os.path.join(f"{os.getcwd()}\\test\\Task1\\predicted-truth", f"0{i}_pt.txt")
    else:
        file_path = os.path.join(f"{os.getcwd()}\\test\\Task1\\predicted-truth", f"{i}_pt.txt")

    ensure_directory_exists(os.path.dirname(file_path))

    with open(file_path, 'w') as file:
        file.write(f"{query[0]}\n")
        for q,response in zip(query[1], detection_list):
            file.write(f"{q} {response}\n")

    i+=1

predicted_truths = load_results("test/Task1/predicted-truth")
ground_truths = load_results("test/Task1/ground-truth")

acc = calculate_accuracy(predicted_truths, ground_truths)
print(acc)