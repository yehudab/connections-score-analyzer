#!/usr/bin/env python3
"""
Connections Score Analyzer
Downloads WhatsApp images from a date range and analyzes scores.
"""

import argparse
import os
import csv
import hashlib
from datetime import datetime, date
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import cv2
import numpy as np

# Import the scoring function from image-debug.py
import sys
sys.path.insert(0, os.path.dirname(__file__))

YOU = "Yehuda"


def parse_date(date_str):
    """Parse ISO-like date format YYYY-MM-DD"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_whatsapp_date(date_text):
    """
    Parse WhatsApp date format to a date object.
    Examples: "12/27/25, 10:30 PM" or "12/25/2025 at 16:16" or "Yesterday" or "Today"
    """
    date_text = date_text.strip()

    # Handle "Today" and "Yesterday"
    if date_text.lower().startswith("today"):
        return date.today()
    elif date_text.lower().startswith("yesterday"):
        from datetime import timedelta
        return date.today() - timedelta(days=1)

    # Parse MM/DD/YY or MM/DD/YYYY format
    try:
        # Extract just the date part (before comma or "at")
        if ',' in date_text:
            date_part = date_text.split(',')[0].strip()
        elif ' at ' in date_text.lower():
            date_part = date_text.lower().split(' at ')[0].strip()
        else:
            date_part = date_text.strip()

        # Try 4-digit year first, then 2-digit year
        try:
            dt = datetime.strptime(date_part, "%m/%d/%Y")
        except ValueError:
            dt = datetime.strptime(date_part, "%m/%d/%y")

        return dt.date()
    except (ValueError, IndexError):
        print(f"Warning: Could not parse date '{date_text}', skipping")
        return None


def download_images(from_date, to_date, output_dir):
    """
    Download images from WhatsApp Web for the specified date range.
    Returns list of (filename, sender, date_text, date_obj, uuid)
    """
    os.makedirs(output_dir, exist_ok=True)

    images_metadata = []
    seen_hashes = set()

    print(f"\nDownloading images from {from_date} to {to_date}...")
    print("Opening WhatsApp Web in browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir="whatsapp-session",
            headless=False
        )
        page = browser.new_page()
        page.goto("https://web.whatsapp.com")
        page.wait_for_timeout(5000)

        print("\nMake sure the media viewer is open on the LATEST image.")
        print("Navigate to the group's media, then open the most recent image.")
        input("Press Enter when ready...")

        should_stop = False

        while True:
            # Get image
            img = page.locator("img[src^='blob:']").first
            if img.count() == 0:
                print("No image found, stopping.")
                break

            blob_url = img.get_attribute("src")

            image_bytes = page.evaluate("""
            async (url) => {
                const response = await fetch(url);
                const buffer = await response.arrayBuffer();
                return Array.from(new Uint8Array(buffer));
            }
            """, blob_url)
            image_bytes = bytes(image_bytes)

            # Dedup by hash
            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash in seen_hashes:
                print("Duplicate image, skipping.")
            else:
                seen_hashes.add(img_hash)

                # Get metadata
                user_span = page.locator("div[role=gridcell] span[dir=auto]").first
                sender = user_span.inner_text().strip()
                if sender == "You":
                    sender = YOU

                date_div = user_span.locator("xpath=../../following-sibling::div/div").first
                date_text = date_div.inner_text().strip()

                # Parse date and check if in range
                image_date = parse_whatsapp_date(date_text)

                if image_date is None:
                    print(f"Skipping image with unparseable date: {date_text}")
                elif image_date < from_date:
                    print(f"Reached images before {from_date} ({date_text}), stopping.")
                    should_stop = True
                elif image_date > to_date:
                    print(f"Image from {date_text} is after {to_date}, skipping...")
                else:
                    # Image is in range, save it
                    uuid = urlparse(blob_url).path.split("/")[-1]
                    filename = f"{uuid}_{sender}.png"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_bytes)

                    images_metadata.append({
                        'filename': filename,
                        'sender': sender,
                        'date_text': date_text,
                        'date': image_date,
                        'uuid': uuid,
                        'filepath': filepath
                    })

                    print(f"Downloaded: {filename} ({date_text})")

            if should_stop:
                break

            # Click previous button
            prev_btn = page.locator('button[aria-label="Previous"]')
            if prev_btn.count() == 0:
                print("No previous button, finished.")
                break
            prev_btn.click()
            page.wait_for_timeout(1200)

        browser.close()

    return images_metadata


def compute_score_from_bar_image(image_path, save_debug=True, debug_dir='debug'):
    """
    Simplified version of the score detection from image-debug.py
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
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find score bars
    bars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        in_content_v = height * 0.15 < y < height * 0.95
        in_content_h = x > width * 0.05

        if in_content_v and in_content_h and 20 < h < 80 and 15 < w < 400 and area > 400:
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

            # Color density check
            bar_region_gray = cv2.cvtColor(cropped_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            dark_pixels = np.sum(bar_region_gray < 220)
            dark_ratio = dark_pixels / area

            if dark_ratio < 0.40:
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

    # Pre-filter by alignment groups
    if len(bars) > 8:
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

    # Find best cluster of 8 bars
    if len(bars) < 8:
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
                if right_edge_cv > 0.05:
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

            if spacing_cv > 0.10:
                return None, f"failed:spacing_cv_{spacing_cv:.1%}"

    # Alignment validation
    if len(bars) >= 2:
        right_edges = [b['x'] + b['w'] for b in bars]
        avg_right_edge = np.mean(right_edges)
        std_right_edge = np.std(right_edges)
        right_edge_cv = (std_right_edge / avg_right_edge) if avg_right_edge > 0 else 0

        if right_edge_cv > 0.05:
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


def main():
    parser = argparse.ArgumentParser(description='Analyze Connections game scores from WhatsApp images')
    parser.add_argument('--from', dest='from_date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--to', dest='to_date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')

    args = parser.parse_args()

    from_date = parse_date(args.from_date)
    to_date = parse_date(args.to_date)

    if from_date > to_date:
        print(f"Error: from date ({from_date}) is after to date ({to_date})")
        return 1

    # Create directory structure
    date_range_str = f"{from_date}_{to_date}"
    base_dir = os.path.join("images", date_range_str)
    src_dir = os.path.join(base_dir, "src")
    debug_dir = os.path.join(base_dir, "debug")
    csv_path = os.path.join(base_dir, "scores.csv")

    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Connections Score Analyzer")
    print(f"Date Range: {from_date} to {to_date}")
    print(f"Output Directory: {base_dir}")
    print(f"{'='*70}")

    # Download images
    images_metadata = download_images(from_date, to_date, src_dir)

    if not images_metadata:
        print("\nNo images downloaded.")
        return 0

    print(f"\n{'='*70}")
    print(f"Downloaded {len(images_metadata)} images")
    print(f"{'='*70}")

    # Create CSV and process images
    print("\nProcessing images and detecting scores...")

    results = []
    success_count = 0
    failure_count = 0

    for i, meta in enumerate(images_metadata, 1):
        print(f"\n[{i}/{len(images_metadata)}] Processing {meta['filename']}...", end=" ")

        score, status = compute_score_from_bar_image(
            meta['filepath'],
            save_debug=True,
            debug_dir=debug_dir
        )

        if status == "success":
            success_count += 1
            print(f"✓ Score: {score}")
        else:
            failure_count += 1
            print(f"✗ {status}")

        results.append({
            'date': meta['date'].isoformat(),
            'time': meta['date_text'],
            'user_name': meta['sender'],
            'image_uuid': meta['uuid'],
            'score_status': status,
            'score': score if score is not None else ''
        })

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'time', 'user_name', 'image_uuid', 'score_status', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total photos: {len(images_metadata)}")
    print(f"Successfully scored: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"\nCSV saved to: {csv_path}")

    # Calculate per-user totals
    user_scores = {}
    for result in results:
        if result['score'] != '':
            user = result['user_name']
            score = result['score']
            if user not in user_scores:
                user_scores[user] = {'total': 0, 'count': 0}
            user_scores[user]['total'] += score
            user_scores[user]['count'] += 1

    if user_scores:
        print(f"\n{'='*70}")
        print(f"SCORES BY USER (sorted by total score)")
        print(f"{'='*70}")

        # Sort by total score descending
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1]['total'], reverse=True)

        for user, data in sorted_users:
            print(f"{user:20s} Total: {data['total']:3d}  (from {data['count']} images)")

    print(f"\n{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
