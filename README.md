# Connections Score Analyzer

Automated tool to download WhatsApp Connections game screenshots and analyze scores using computer vision.

## Features

- Downloads images from WhatsApp Web for a specified date range
- Detects and scores Connections game results using CV (99%+ accuracy)
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

## PicoClaw Bot Integration

This project runs as a sidecar HTTP service alongside the [PicoClaw](https://github.com/sipeed/picoclaw) WhatsApp bot. The bot receives images from the WhatsApp group, calls the scorer API, and posts the results back.

### Architecture

```
WhatsApp group → picoclaw-gateway (Go bot)
                      ↓  POST /score (multipart image)
                 connections-scorer (Flask, this repo)
                      ↓
                 SQLite DB (/data/scores.db)
```

Both containers share a Docker network (`botnet`). The bot reaches the scorer at `http://scorer:5000`.

### API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/score` | Submit image for scoring. Fields: `image` (file), `user_id`, `user_name` |
| `GET` | `/leaderboard?sprint=current` | Rankings for a sprint |
| `GET` | `/stats?user_id=X&sprint=current` | Personal stats |
| `GET` | `/summary?sprint=current` | Formatted sprint summary text |
| `GET` | `/missing` | Members who haven't submitted today (Israel time) |
| `GET` | `/sprint` | Current sprint ID, dates, days remaining |
| `GET` | `/config/sprint_epoch` | View sprint start date |
| `PUT` | `/config/sprint_epoch` | Update sprint start date |
| `GET` | `/health` | Health check |

### Curling the API

Use the picoclaw-gateway container (which has `curl`) to call the scorer.
This also verifies the Docker network between the two containers is working:

```bash
sudo docker exec -it picoclaw-gateway sh

# Health check
curl http://scorer:5000/health

# Current sprint info
curl http://scorer:5000/sprint

# Who hasn't submitted today?
curl http://scorer:5000/missing

# Leaderboard
curl http://scorer:5000/leaderboard?sprint=current

# Sprint summary
curl http://scorer:5000/summary?sprint=current

# Update sprint epoch
curl -X PUT http://scorer:5000/config/sprint_epoch \
  -H "Content-Type: application/json" \
  -d '{"date": "2025-12-12"}'

# Score an image manually (image must exist inside the picoclaw-gateway container)
curl -X POST http://scorer:5000/score \
  -F "image=@/home/picoclaw/.picoclaw/workspace/media/some-image.jpg" \
  -F "user_id=972500000000" \
  -F "user_name=Alice"
```

### Sprint Configuration

Sprints are 14-day periods starting from the epoch date (stored in the DB).
To update after deploy:

```bash
sudo docker exec -it connections-scorer sh
curl -X PUT http://scorer:5000/config/sprint_epoch \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-02-06"}'
```

### Daily Reminder Cron

The bot is configured to call `GET /missing` at 21:00 Israel time every day
and send a reminder to the group for members who haven't submitted yet.
The cron job is managed by the PicoClaw cron tool inside the bot container.

### Viewing Container Logs

Logs persist as long as the container exists (even when stopped), but are lost on `docker compose down`.

```bash
# Check if container still exists
sudo docker ps -a | grep picoclaw-gateway

# View logs (works even when container is stopped)
sudo docker logs picoclaw-gateway

# Last 200 lines with timestamps
sudo docker logs picoclaw-gateway --timestamps --tail 200

# Follow live logs
sudo docker logs picoclaw-gateway --follow
```

The `logging` driver in `docker-compose.yml` keeps up to 50MB of history (5 × 10MB files),
so logs survive container restarts.

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
