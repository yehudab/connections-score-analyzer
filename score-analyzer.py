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

from scorer import compute_score_from_bar_image

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


def main():
    parser = argparse.ArgumentParser(description='Analyze Connections game scores from WhatsApp images')
    parser.add_argument('--from', dest='from_date', type=str, required=True,
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--to', dest='to_date', type=str, required=True,
                        help='End date in YYYY-MM-DD format')
    parser.add_argument('--rescore', action='store_true',
                        help='Re-run scoring on already-downloaded images without downloading')

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

    if args.rescore:
        debug_dir = os.path.join(base_dir, "debug-v2")
        csv_path = os.path.join(base_dir, "scores-v2.csv")
    else:
        debug_dir = os.path.join(base_dir, "debug")
        csv_path = os.path.join(base_dir, "scores.csv")

    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Connections Score Analyzer")
    print(f"Date Range: {from_date} to {to_date}")
    print(f"Output Directory: {base_dir}")
    if args.rescore:
        print(f"Mode: RESCORE (debug-v2/, scores-v2.csv)")
    print(f"{'='*70}")

    if args.rescore:
        # Read existing images from src/ directory
        if not os.path.isdir(src_dir):
            print(f"\nError: Source directory not found: {src_dir}")
            return 1

        # Load original CSV to get date/time metadata and row order
        original_csv_path = os.path.join(base_dir, "scores.csv")
        orig_rows = []
        if os.path.isfile(original_csv_path):
            with open(original_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_rows.append(row)
            print(f"Loaded {len(orig_rows)} entries from original scores.csv")

        # Build set of available image files by UUID
        import glob as glob_mod
        available_files = {}
        for filepath in glob_mod.glob(os.path.join(src_dir, "*.png")):
            filename = os.path.basename(filepath)
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split('_', 1)
            uuid = parts[0]
            sender = parts[1] if len(parts) == 2 else "Unknown"
            available_files[uuid] = {'filepath': filepath, 'filename': filename, 'sender': sender}

        # Build metadata list in original CSV order, then append any files not in CSV
        images_metadata = []
        seen_uuids = set()
        for row in orig_rows:
            uuid = row['image_uuid']
            if uuid in available_files:
                seen_uuids.add(uuid)
                date_str = row.get('date', '')
                try:
                    img_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else from_date
                except ValueError:
                    img_date = from_date
                images_metadata.append({
                    'filename': available_files[uuid]['filename'],
                    'sender': row.get('user_name', '') or available_files[uuid]['sender'],
                    'date_text': row.get('time', ''),
                    'date': img_date,
                    'uuid': uuid,
                    'filepath': available_files[uuid]['filepath']
                })

        # Append any images not in the original CSV (sorted by filename)
        for uuid in sorted(available_files, key=lambda u: available_files[u]['filename']):
            if uuid not in seen_uuids:
                images_metadata.append({
                    'filename': available_files[uuid]['filename'],
                    'sender': available_files[uuid]['sender'],
                    'date_text': '',
                    'date': from_date,
                    'uuid': uuid,
                    'filepath': available_files[uuid]['filepath']
                })

        if not images_metadata:
            print(f"\nNo .png files found in {src_dir}")
            return 0

        print(f"\n{'='*70}")
        print(f"Found {len(images_metadata)} images for rescoring")
        print(f"{'='*70}")
    else:
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
