"""
Connections game score detection from bar chart images.
"""

import os
import cv2
import numpy as np


def compute_score_from_bar_image(image_path, save_debug=True, debug_dir='debug'):
    """
    Detect the Connections game score from a screenshot image.
    Returns (score, status) where status is 'success' or 'failed:<reason>'
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, "failed:cannot_read_image"

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find white content card
    _, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_white_area = 0
    content_region = None

    for c in contours_white:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > (width * height * 0.1) and w > 200 and h > 300:
            if area > largest_white_area:
                largest_white_area = area
                content_region = (x, y, w, h)

    if content_region is None:
        content_x, content_y, content_w, content_h = 0, 0, width, height
        cropped_img = img
    else:
        content_x, content_y, content_w, content_h = content_region
        cropped_img = img[content_y:content_y+content_h, content_x:content_x+content_w]
        height, width = cropped_img.shape[:2]

    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_green1 = np.array([25, 12, 60])
    upper_green1 = np.array([105, 255, 255])
    lower_green2 = np.array([25, 5, 130])
    upper_green2 = np.array([105, 90, 255])
    lower_grey = np.array([0, 0, 150])
    upper_grey = np.array([180, 35, 250])
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    green_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_green1, upper_green1),
        cv2.inRange(hsv, lower_green2, upper_green2)
    )
    grey_mask = cv2.inRange(hsv, lower_grey, upper_grey)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    combined_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, grey_mask), red_mask)
    # Morphological opening to break thin pixel bridges (anti-aliasing/compression artifacts)
    # that can merge separate bars into one giant contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find score bars
    bars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        in_content_v = height * 0.15 < y < height * 0.95
        in_content_h = x > width * 0.05

        if in_content_v and in_content_h and 20 < h < 80 and 10 < w < 400 and area > 200:
            # Fix 1: Rectangularity filter - reject annotations (circles, etc.)
            contour_area = cv2.contourArea(c)
            rect_fill_ratio = contour_area / area if area > 0 else 0
            if rect_fill_ratio < 0.55:
                continue

            bar_region_hsv = hsv[y:y+h, x:x+w]

            green_pixels = cv2.bitwise_or(
                cv2.inRange(bar_region_hsv, lower_green1, upper_green1),
                cv2.inRange(bar_region_hsv, lower_green2, upper_green2)
            )
            red_pixels = cv2.bitwise_or(
                cv2.inRange(bar_region_hsv, lower_red1, upper_red1),
                cv2.inRange(bar_region_hsv, lower_red2, upper_red2)
            )

            green_ratio = np.count_nonzero(green_pixels) / area
            red_ratio = np.count_nonzero(red_pixels) / area

            is_green = green_ratio > 0.02 and green_ratio > red_ratio
            is_red = red_ratio > 0.15

            # Fix 2: Relaxed dark_ratio - skip check for colored bars
            bar_region_gray = cv2.cvtColor(cropped_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            dark_pixels = np.sum(bar_region_gray < 220)
            dark_ratio = dark_pixels / area

            has_color_signal = green_ratio > 0.01 or red_ratio > 0.10
            if not has_color_signal:
                # For grey bars: try relaxed threshold as fallback
                if dark_ratio < 0.40:
                    dark_pixels_relaxed = np.sum(bar_region_gray < 240)
                    dark_ratio_relaxed = dark_pixels_relaxed / area
                    if dark_ratio_relaxed < 0.30:
                        continue

            bars.append({
                'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
                'is_green': is_green, 'is_red': is_red,
                'center_y': y + h // 2,
                'green_ratio': green_ratio, 'red_ratio': red_ratio,
                'dark_ratio': dark_ratio
            })

    # Deduplicate
    unique_bars = []
    bars_sorted = sorted(bars, key=lambda b: (b['center_y'], -b['w']))
    for bar in bars_sorted:
        is_duplicate = False
        for existing in unique_bars:
            if abs(existing['center_y'] - bar['center_y']) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_bars.append(bar)

    bars = sorted(unique_bars, key=lambda b: b['center_y'])

    # Fix 3: Restrict bars to chart area - find best 8-bar cluster first,
    # then filter out bars outside its vertical range
    if len(bars) > 8:
        # First try alignment groups to narrow down
        alignment_groups = []
        tolerance = 50
        for bar in bars:
            bar_right = bar['x'] + bar['w']
            found_group = False
            for group in alignment_groups:
                group_avg_right = np.mean([b['x'] + b['w'] for b in group])
                if abs(bar_right - group_avg_right) < tolerance:
                    group.append(bar)
                    found_group = True
                    break
            if not found_group:
                alignment_groups.append([bar])

        if alignment_groups:
            largest_group = max(alignment_groups, key=len)
            if len(largest_group) >= 8:
                bars = sorted(largest_group, key=lambda b: b['center_y'])

    # If still >8, find best 8-bar cluster by spacing+alignment and restrict vertical range
    if len(bars) > 8:
        best_cluster = None
        best_quality = -1000
        for start_idx in range(len(bars) - 7):
            cluster = bars[start_idx:start_idx + 8]
            spacings = [cluster[i+1]['center_y'] - cluster[i]['center_y'] for i in range(7)]
            avg_spacing = np.mean(spacings)
            if avg_spacing <= 0:
                continue
            spacing_cv = np.std(spacings) / avg_spacing
            right_edges = [b['x'] + b['w'] for b in cluster]
            right_edge_cv = np.std(right_edges) / np.mean(right_edges) if np.mean(right_edges) > 0 else 1.0
            quality = 500 - right_edge_cv * 1000 - spacing_cv * 400
            if quality > best_quality:
                best_quality = quality
                best_cluster = cluster
        if best_cluster is not None:
            # Filter bars to only those within the cluster's vertical range (with margin)
            cluster_top = best_cluster[0]['center_y']
            cluster_bottom = best_cluster[-1]['center_y']
            v_range = cluster_bottom - cluster_top
            margin = v_range * 0.10
            bars = [b for b in bars if cluster_top - margin <= b['center_y'] <= cluster_bottom + margin]
            bars = sorted(bars, key=lambda b: b['center_y'])

    # Find best cluster of 8 bars
    if len(bars) < 8:
        # Fix 4: Non-game image detection
        if len(bars) < 4 and content_region is None:
            return None, "failed:not_a_game"
        if 4 <= len(bars) < 8 and content_region is None:
            # Check height consistency - real bars have similar heights
            if len(bars) >= 2:
                heights = [b['h'] for b in bars]
                height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1.0
                if height_cv > 0.3:
                    return None, "failed:not_a_game"
        return None, f"failed:only_{len(bars)}_bars"
    elif len(bars) >= 8:
        best_cluster = None
        best_score_quality = -1000

        for start_idx in range(len(bars) - 7):
            cluster = bars[start_idx:start_idx + 8]
            spacings = []
            for i in range(len(cluster) - 1):
                spacing = cluster[i+1]['center_y'] - cluster[i]['center_y']
                spacings.append(spacing)

            if spacings:
                avg_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)
                spacing_cv = std_spacing / avg_spacing if avg_spacing > 0 else 1.0

                heights = [b['h'] for b in cluster]
                std_height = np.std(heights)

                right_edges = [b['x'] + b['w'] for b in cluster]
                std_right_edge = np.std(right_edges)
                right_edge_cv = std_right_edge / np.mean(right_edges) if np.mean(right_edges) > 0 else 1.0

                quality = 500
                quality -= right_edge_cv * 1000
                quality -= spacing_cv * 400
                quality -= std_height * 2

                if cluster[0]['center_y'] > height * 0.2:
                    quality += 50
                if 35 < avg_spacing < 95:
                    quality += 50
                if right_edge_cv > 0.08:
                    quality -= 500

                if quality > best_score_quality:
                    best_score_quality = quality
                    best_cluster = cluster

        if best_cluster is not None:
            bars = best_cluster
        else:
            bars = bars[:8] if len(bars) >= 8 else bars

    # Validation
    green_count = sum(1 for b in bars if b['is_green'])
    red_count = sum(1 for b in bars if b.get('is_red', False))
    grey_count = sum(1 for b in bars if not b['is_green'] and not b.get('is_red', False))

    # Check validation rules
    if green_count == 0 and red_count == 0:
        return None, "failed:no_green_or_red"
    elif green_count > 1:
        return None, f"failed:{green_count}_green_bars"
    elif red_count > 1:
        return None, f"failed:{red_count}_red_bars"
    elif green_count >= 1 and red_count >= 1:
        return None, "failed:both_green_and_red"

    if grey_count != 7:
        return None, f"failed:{grey_count}_grey_bars"

    # Spacing validation
    if len(bars) >= 2:
        spacings = []
        for i in range(len(bars) - 1):
            spacing = bars[i+1]['center_y'] - bars[i]['center_y']
            spacings.append(spacing)

        if spacings:
            avg_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            spacing_cv = (std_spacing / avg_spacing) if avg_spacing > 0 else 0

            if spacing_cv > 0.18:
                return None, f"failed:spacing_cv_{spacing_cv:.1%}"

    # Alignment validation
    if len(bars) >= 2:
        right_edges = [b['x'] + b['w'] for b in bars]
        avg_right_edge = np.mean(right_edges)
        std_right_edge = np.std(right_edges)
        right_edge_cv = (std_right_edge / avg_right_edge) if avg_right_edge > 0 else 0

        if right_edge_cv > 0.08:
            return None, f"failed:alignment_cv_{right_edge_cv:.1%}"

    # Draw debug image
    debug_img = img.copy()
    if content_region is not None:
        cv2.rectangle(debug_img, (content_x, content_y),
                     (content_x+content_w, content_y+content_h), (255, 255, 0), 2)

    green_bar = None
    red_bar = None

    for i, bar in enumerate(bars):
        x_orig = bar['x'] + content_x
        y_orig = bar['y'] + content_y
        w, h = bar['w'], bar['h']

        if bar['is_green']:
            color = (0, 255, 0)
            thickness = 3
            label = f"GREEN #{i+1}"
            green_bar = bar
            green_bar['position'] = i + 1
        elif bar.get('is_red', False):
            color = (0, 0, 255)
            thickness = 3
            label = f"RED #{i+1}"
            red_bar = bar
            red_bar['position'] = i + 1
        else:
            color = (255, 0, 0)
            thickness = 2
            label = f"Grey #{i+1}"

        cv2.rectangle(debug_img, (x_orig, y_orig), (x_orig+w, y_orig+h), color, thickness)
        cv2.putText(debug_img, label, (x_orig, max(10, y_orig-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate score
    if green_bar is not None:
        score = 8 - green_bar['position'] + 1
    elif red_bar is not None and red_bar['position'] == len(bars):
        score = 1
    else:
        return None, "failed:unexpected_state"

    # Add score to debug image
    score_text = f"SCORE: {score}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    padding = 10
    cv2.rectangle(debug_img, (10, 10),
                  (10 + text_width + padding * 2, 10 + text_height + padding * 2 + baseline),
                  (100, 100, 100), -1)
    cv2.putText(debug_img, score_text, (10 + padding, 10 + text_height + padding),
                font, font_scale, (0, 255, 255), font_thickness)

    # Save debug image
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_path = os.path.join(debug_dir, f'{base_name}_debug.png')
        cv2.imwrite(debug_path, debug_img)

    return score, "success"
