import cv2
import numpy as np
import os
import time
import math
import csv
import psutil

# === CONFIGURATION ===
TEMPLATE_DIR      = "../patterns/circle"
FRAME_WIDTH       = 960
MATCH_THRESHOLD   = 5        # total good matches (across all 3 channels) needed
RATIO_THRESH      = 0.85     # Lowe’s ratio test
FLANN_TREES       = 5
FLANN_CHECKS      = 1500
SIFT_MAX_FEATURES = 2000
HOLD_FRAMES_MAX   = 5        # how many frames to keep “detection” state alive
UNLOCK_DELAY      = 2.0      # seconds the correct pattern must persist

# --- Pre-processing parameters
FIXED_SIZE        = (200, 200)
BILATERAL_PARAMS  = (5, 75, 75)
NLM_PARAMS        = {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21}
CLAHE_CLIP        = 2.0
CLAHE_GRID        = (8, 8)

# --- Contour filtering
MIN_AREA           = 500     # px²
CIRCULARITY_THRESH = 0.7     # 4π·A / P²

# === OpenCV & SIFT/FLANN setup ===
cv2.setUseOptimized(True)
cv2.setNumThreads(4)
sift  = cv2.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=FLANN_TREES),
    dict(checks=FLANN_CHECKS)
)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def preprocess(img):
    """Denoise, CLAHE enhance, resize & return processed color + gray."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.bilateralFilter(s, *BILATERAL_PARAMS)
    v = cv2.fastNlMeansDenoising(v, None, **NLM_PARAMS)
    den = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(den, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

    proc = cv2.resize(enhanced, FIXED_SIZE)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    return proc, gray

def is_circle(cnt):
    """Filter non-circular contours by area & circularity."""
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    circ = 4 * math.pi * area / (peri * peri)
    return circ >= CIRCULARITY_THRESH

def select_pattern_cv(original_map, thumb_size=(150,150), cols=4):
    """Shows all patterns in one OpenCV window. Click on a thumbnail to pick it."""
    items = list(original_map.items())
    n = len(items)
    rows = (n + cols - 1) // cols
    w, h = thumb_size

    canvas = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
    for idx, (pid, img) in enumerate(items):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(img, thumb_size)
        y, x = r*h, c*w
        canvas[y:y+h, x:x+w] = thumb
        cv2.rectangle(canvas, (x,y), (x+w-1,y+h-1), (255,255,255), 2)
        cv2.putText(canvas, pid, (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    selected = {'pid': None}
    win = "Select Pattern"
    cv2.namedWindow(win)
    cv2.imshow(win, canvas)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            col = mx // w
            row = my // h
            idx = row*cols + col
            if 0 <= idx < n:
                selected['pid'] = items[idx][0]
                cv2.destroyWindow(win)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if selected['pid'] is not None:
            return selected['pid']
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            return None

def detect_pattern(desc2_list):
    """Returns (pattern_id, total_good_matches) or (None,0)."""
    counts = {}
    for tmpl in templates:
        total_good = 0
        for desc_t, desc2 in zip(tmpl['desc_ch'], desc2_list):
            if desc_t is None or desc2 is None or len(desc_t) < 2 or len(desc2) < 2:
                continue
            matches = flann.knnMatch(desc_t, desc2, k=2)
            good = [m for m,n in matches if m.distance < RATIO_THRESH * n.distance]
            total_good += len(good)
        pid = tmpl['pattern_id']
        counts[pid] = max(counts.get(pid, 0), total_good)

    if not counts:
        return None, 0
    best_pid = max(counts, key=counts.get)
    best_cnt = counts[best_pid]
    return (best_pid, best_cnt) if best_cnt >= MATCH_THRESHOLD else (None, 0)

def play_unlock_animation(frame, template):
    """Show an animated 'UNLOCKED' overlay."""
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    base = frame.copy()
    for i in range(20):
        alpha = i/19.0
        overlay = base.copy()
        cv2.putText(overlay, 'UNLOCKED', (center[0]-140, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 4, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)
        cv2.imshow('Match', blended)
        cv2.imshow('Template', template)
        cv2.waitKey(50)
    final = base.copy()
    cv2.rectangle(final, (0,0), (w,h), (0,255,0), 8)
    cv2.putText(final, 'UNLOCKED', (center[0]-140, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0,255,0), 6, cv2.LINE_AA)
    cv2.imshow('Match', final)
    cv2.imshow('Template', template)
    cv2.waitKey(0)

# --------------------------------------------------------------------------
# Load templates (compute SIFT per channel)
# --------------------------------------------------------------------------
original_map = {}  
templates    = []

for d in sorted(os.listdir(TEMPLATE_DIR)):
    sub = os.path.join(TEMPLATE_DIR, d)
    if not os.path.isdir(sub) or not d.startswith('pat_'):
        continue
    pid = d.split('_',1)[1]

    for fn in sorted(os.listdir(sub)):
        if fn.endswith('.png') and '_crop' not in fn:
            img = cv2.imread(os.path.join(sub, fn))
            if img is not None:
                original_map[pid] = preprocess(img)[0]
                break

    for fn in sorted(os.listdir(sub)):
        img = cv2.imread(os.path.join(sub, fn))
        if img is None:
            continue
        proc, _ = preprocess(img)
        chans = cv2.split(proc)
        desc_ch = []
        for ch in chans:
            kp, desc = sift.detectAndCompute(ch, None)
            desc_ch.append(desc if desc is not None else np.zeros((0,128),dtype=np.float32))
        templates.append({'pattern_id': pid, 'desc_ch': desc_ch})

if not templates:
    print("No valid templates found! Check TEMPLATE_DIR.")
    exit(1)

cv2.namedWindow('Match')
cv2.namedWindow('Template')

# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
while True:
    selected = select_pattern_cv(original_map)
    if selected is None:
        print("No selection made, exiting.")
        break

    # prompt for recording
    record = input("Record performance data to CSV? (y/n): ").strip().lower().startswith('y')
    if record:
        total_detects   = 0
        correct_detects = 0
        latencies       = []
        cpu_usages      = []
        mem_usages      = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        break

    prev_time       = time.time()
    hold            = 0
    detection_start = None
    exit_program    = False
    print("Press 'r' to re-select pattern, 'q' to quit.")

    while True:
        loop_t = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # resize
        h0, w0 = frame.shape[:2]
        new_h  = int(h0 * FRAME_WIDTH / w0)
        frm    = cv2.resize(frame, (FRAME_WIDTH, new_h))
        disp   = frm.copy()

        # contour detection
        t0      = time.time()
        gray_f  = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray_f, (7,7), 0)
        edges   = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_cont  = (time.time() - t0)*1000

        detections = []
        t_match     = 0

        for c in cnts:
            if not is_circle(c):
                continue
            x,y,wc,hc = cv2.boundingRect(c)
            roi       = frm[y:y+hc, x:x+wc]
            proc_roi, _ = preprocess(roi)
            chans2      = cv2.split(proc_roi)

            desc2_list = []
            for ch in chans2:
                kp2, desc2 = sift.detectAndCompute(ch, None)
                desc2_list.append(desc2 if desc2 is not None else np.zeros((0,128),dtype=np.float32))

            t1           = time.time()
            pid, count  = detect_pattern(desc2_list)
            t_match     += (time.time() - t1)*1000
            if pid:
                detections.append((x,y,wc,hc,pid,count))

        # record stats
        if record and detections:
            total_detects += 1
            if any(d[4]==selected for d in detections):
                correct_detects += 1
                total_latency = (time.time()-loop_t)*1000
                latencies.append(total_latency)
                cpu_usages.append(psutil.cpu_percent(interval=None))
                mem_usages.append(psutil.virtual_memory().percent)

        # sustained detection
        seen = any(d[4]==selected for d in detections)
        if seen:
            if detection_start is None:
                detection_start = time.time()
            hold = HOLD_FRAMES_MAX
        else:
            hold = max(hold-1,0)
            if hold==0:
                detection_start = None

        if detection_start and (time.time()-detection_start)>=UNLOCK_DELAY:
            play_unlock_animation(disp, original_map[selected])
            break

        # draw boxes
        for x,y,wc,hc,pid,count in detections:
            color = (0,255,0) if pid==selected else (0,0,255)
            cv2.rectangle(disp, (x,y), (x+wc, y+hc), color, 2)
            cv2.putText(disp, f"{pid}:{count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # overlays
        total_latency = (time.time()-loop_t)*1000
        fps           = 1.0/(time.time()-prev_time)
        prev_time     = time.time()
        cv2.putText(disp,
            f"Contour:{t_cont:.1f}ms Match:{t_match:.1f}ms Total:{total_latency:.1f}ms",
            (10,new_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),1)
        cv2.putText(disp, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255),2)

        cv2.imshow('Match', disp)
        if detections:
            cv2.imshow('Template', original_map[selected])
        else:
            cv2.imshow('Template', np.zeros_like(original_map[selected]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            break
        elif key == ord('q'):
            exit_program = True
            break

    cap.release()
    cv2.destroyAllWindows()

    # write CSV if recording
    if record:
        filename = "performance.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Pattern", "Picture", "Total Frames", "Correct Frames",
                    "Accuracy", "Avg Latency (ms)", "Avg CPU (%)", "Avg Mem (%)"
                ])
            accuracy    = correct_detects/total_detects if total_detects>0 else 0
            avg_latency = sum(latencies)/len(latencies) if latencies else 0
            avg_cpu     = sum(cpu_usages)/len(cpu_usages) if cpu_usages else 0
            avg_mem     = sum(mem_usages)/len(mem_usages) if mem_usages else 0
            writer.writerow([
                selected, "",
                total_detects, correct_detects,
                accuracy, avg_latency, avg_cpu, avg_mem
            ])

    if exit_program:
        break