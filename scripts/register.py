#!/usr/bin/env python3
import cv2
import imutils
import numpy as np
import os
import random

# --------------------------------
# Configurable Parameters
# --------------------------------
BASE_DIR                = "patterns"
SHAPES                  = ["circle", "square", "triangle"]
FIXED_SIZE              = (200, 200)   # resize all template ROIs to 200×200

# Random cropping settings
NUM_RANDOM_CROPS        = 5            # number of random crops per ROI
CROP_SIZE               = (100, 100)   # size of each random crop

CLAHE_CLIP              = 2.0
CLAHE_GRID              = (8, 8)
BILATERAL_PARAMS        = (5, 75, 75)
NLM_PARAMS              = {"h": 10, "templateWindowSize":7, "searchWindowSize":21}

CONTOUR_PARAMS = {
    'min_area': 10,
    'approx_poly_epsilon': 0.2,
    'square_aspect_ratio_min': 0.95,
    'square_aspect_ratio_max': 1.05,
}
DISPLAY_PARAMS = {
    'contour_color': (0, 255, 0),
    'contour_thickness': 2,
    'text_color': (255, 255, 255),
    'text_thickness': 2,
    'text_scale': 0.5,
    'status_color': (0, 0, 255),
    'status_scale': 0.8,
}
# --------------------------------

def get_next_pattern_number():
    max_num = 0
    for shape in SHAPES:
        base = os.path.join(BASE_DIR, shape)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.startswith("pat_") and os.path.isdir(os.path.join(base, name)):
                try:
                    num = int(name.split("_")[1])
                    max_num = max(max_num, num)
                except:
                    pass
    return max_num + 1

def detect_shape(c):
    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, CONTOUR_PARAMS['approx_poly_epsilon'] * peri, True)
    v = len(approx)
    if v == 3:
        return "triangle"
    if v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w/float(h)
        if CONTOUR_PARAMS['square_aspect_ratio_min'] <= ar <= CONTOUR_PARAMS['square_aspect_ratio_max']:
            return "square"
        else:
            return "rectangle"
    return "circle"

def extract_shape_region(frame, contour):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    rgba = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    rgba[...,3] = mask_roi
    return rgba

def denoise_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    d, sc, ss = BILATERAL_PARAMS
    s = cv2.bilateralFilter(s, d, sc, ss)
    v = cv2.fastNlMeansDenoising(v, None, **NLM_PARAMS)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

def on_change(x): pass

def main():
    # ensure folder structure
    os.makedirs(BASE_DIR, exist_ok=True)
    for s in SHAPES:
        os.makedirs(os.path.join(BASE_DIR, s), exist_ok=True)

    # start pattern counter
    current_pat = get_next_pattern_number()
    print(f"→ Starting with pattern #{current_pat:02d}")
    print("   (Press 'c' to capture, 'n' for next pattern, 'q' to quit.)")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Blur k", "Controls", 5, 51, on_change)
    cv2.createTrackbar("Canny Th1", "Controls", 65, 255, on_change)
    cv2.createTrackbar("Canny Th2", "Controls", 150, 255, on_change)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # read controls
        k = cv2.getTrackbarPos("Blur k", "Controls")
        if k < 1: k = 1
        if k % 2 == 0: k += 1
        th1 = cv2.getTrackbarPos("Canny Th1", "Controls")
        th2 = cv2.getTrackbarPos("Canny Th2", "Controls")

        disp = imutils.resize(frame, width=600)
        orig = disp.copy()

        gray    = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        edges   = cv2.Canny(blurred, th1, th2)
        cnts    = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # draw detected shapes
        for c in cnts:
            if cv2.contourArea(c) < CONTOUR_PARAMS['min_area']:
                continue
            shape = detect_shape(c)
            if shape in SHAPES:
                M = cv2.moments(c)
                cx = int(M["m10"]/M["m00"]) if M["m00"]!=0 else 0
                cy = int(M["m01"]/M["m00"]) if M["m00"]!=0 else 0
                cv2.drawContours(disp, [c], -1,
                                 DISPLAY_PARAMS['contour_color'],
                                 DISPLAY_PARAMS['contour_thickness'])
                cv2.putText(disp, shape, (cx-20, cy-20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            DISPLAY_PARAMS['text_scale'],
                            DISPLAY_PARAMS['text_color'],
                            DISPLAY_PARAMS['text_thickness'])

        cv2.putText(disp,
                    f"Pattern #{current_pat:02d}  |  Blur={k} Th1={th1} Th2={th2}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_PARAMS['status_scale'],
                    DISPLAY_PARAMS['status_color'],
                    DISPLAY_PARAMS['text_thickness'])
        cv2.putText(disp, "Press 'c' to capture, 'n' next, 'q' quit",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                    DISPLAY_PARAMS['status_scale'],
                    DISPLAY_PARAMS['status_color'],
                    DISPLAY_PARAMS['text_thickness'])

        cv2.imshow("Shape Detection", disp)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # capture using current_pat
            print(f"\nCapturing pattern #{current_pat:02d}")
            pat_dirs = {}
            for s in SHAPES:
                sub = os.path.join(BASE_DIR, s, f"pat_{current_pat:02d}")
                os.makedirs(sub, exist_ok=True)
                pat_dirs[s] = sub

            counters = {s:1 for s in SHAPES}
            for c in cnts:
                if cv2.contourArea(c) < CONTOUR_PARAMS['min_area']:
                    continue
                shape = detect_shape(c)
                if shape in SHAPES:
                    roi_rgba = extract_shape_region(orig, c)
                    roi_bgr  = cv2.cvtColor(roi_rgba, cv2.COLOR_BGRA2BGR)
                    roi_dn   = denoise_hsv(roi_bgr)
                    roi_cl   = apply_clahe(roi_dn)
                    roi_proc = cv2.resize(roi_cl, FIXED_SIZE)

                    # save full ROI
                    idx      = counters[shape]
                    letter   = shape[0]
                    base_fn  = f"{idx:02d}_{letter}"
                    full_fn  = os.path.join(pat_dirs[shape], base_fn + ".png")
                    cv2.imwrite(full_fn, roi_proc)
                    print(f"  • Saved full ROI: {full_fn}")

                    # random crops
                    h_proc, w_proc = FIXED_SIZE[1], FIXED_SIZE[0]
                    ch, cw         = CROP_SIZE[1], CROP_SIZE[0]
                    for j in range(NUM_RANDOM_CROPS):
                        if w_proc <= cw or h_proc <= ch:
                            break
                        x_off = random.randint(0, w_proc - cw)
                        y_off = random.randint(0, h_proc - ch)
                        crop = roi_proc[y_off:y_off+ch, x_off:x_off+cw]
                        crop_fn = os.path.join(pat_dirs[shape], f"{base_fn}_crop{j}.png")
                        cv2.imwrite(crop_fn, crop)
                        print(f"    • Saved crop: {crop_fn}")

                    counters[shape] += 1
            print("Done.")

        elif key == ord('n'):
            # move to next pattern number
            current_pat += 1
            print(f"\n→ Switched to pattern #{current_pat:02d}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()