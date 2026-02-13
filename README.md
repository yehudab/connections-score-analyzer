# Connections Score Analyzer

Automated tool to download WhatsApp Connections game screenshots and analyze scores using computer vision.

## Features

- Downloads images from WhatsApp Web for a specified date range
- Detects and scores Connections game results using CV (94.1% accuracy)
- Validates detections with strict quality checks
- Generates debug visualizations for all images
- Produces CSV report with all results
- Calculates per-user score totals

## Initial Setup (one time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Usage

```bash
source .venv/bin/activate
python score-analyzer.py --from=2025-12-12 --to=2025-12-25
deactivate  # when done
```

### What It Does

1. **Opens WhatsApp Web** in a browser (persistent session saved)
2. **Prompts you** to navigate to the group's media and open the latest image
3. **Downloads images** from the date range, going backwards from latest
4. **Analyzes each image** to detect score bars and calculate the score
5. **Saves everything**:
   - Original images: `images/<from-date>_<to-date>/src/`
   - Debug images: `images/<from-date>_<to-date>/debug/`
   - CSV report: `images/<from-date>_<to-date>/scores.csv`

### Output

Terminal output shows:
- Number of photos found
- Number of successful/failed detections
- Per-user score totals (sorted highest first)

Example:
```
======================================================================
SUMMARY
======================================================================
Total photos: 25
Successfully scored: 24
Failed: 1

CSV saved to: images/2025-12-12_2025-12-25/scores.csv

======================================================================
SCORES BY USER (sorted by total score)
======================================================================
Alice                Total: 142  (from 8 images)
Bob                  Total: 135  (from 7 images)
Charlie              Total: 98   (from 9 images)
```

## CSV Format

The generated CSV contains:
- `date`: ISO format date (YYYY-MM-DD)
- `time`: Original WhatsApp timestamp
- `user_name`: Sender name
- `image_uuid`: Unique identifier from blob URL
- `score_status`: "success" or "failed:<reason>"
- `score`: Score value 1-8 (empty if failed)

## Score Detection

The algorithm detects:
- **Green bar** at any position → Score = 8 - position + 1
- **Red bar** at bottom position → Score = 1

Validation ensures:
- Exactly 1 green OR 1 red bar (not both)
- Exactly 7 grey bars
- Consistent vertical spacing (SD < 10% of average)
- Consistent horizontal alignment (SD < 5% of right edge)
- Sufficient color density (40%+ filled area)

## Debug Images

Debug images show:
- Yellow border: Detected content card
- Green boxes: Green bar with position
- Red boxes: Red bar
- Blue boxes: Grey bars
- Grey box with yellow text: Detected score

Failed images are saved with `_debug.png` suffix and show which bars were detected.

## Troubleshooting

**"No image found"**: Make sure you're in the media viewer with an image open

**"Duplicate image"**: The script automatically skips duplicates by hash

**Browser doesn't open**: Check Playwright installation with `playwright install chromium`

**Wrong scores**: Check debug images in the `debug/` folder to see what was detected

## Accuracy

Current validation-checked accuracy: **94.1%** (111/118 test images)

Failures are typically due to:
- Very light pastel green shades
- Poor lighting/image quality
- Non-standard screenshot formats
