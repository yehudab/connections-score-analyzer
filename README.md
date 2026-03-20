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
| `POST` | `/score` | Submit image for scoring. Fields: `image` (file), `user_id`, `user_name`, `chat_id` |
| `POST` | `/score/correct` | Manually correct a failed submission. Body: `{user_id, score, chat_id}` |
| `GET` | `/leaderboard?sprint=current\|previous\|<n>&chat_id=X` | Rankings for a sprint |
| `GET` | `/stats?user_id=X&sprint=current\|previous\|<n>&chat_id=X` | Personal stats |
| `GET` | `/summary?sprint=current\|previous\|<n>&chat_id=X` | Formatted sprint summary |
| `GET` | `/sprint/end_report?chat_id=X` | Previous sprint results; `should_post` true only on sprint transition day |
| `GET` | `/missing?chat_id=X` | Members who haven't submitted today (Israel time) |
| `GET` | `/sprint` | Current sprint ID, dates, days remaining |
| `GET` | `/config/sprint_epoch` | View sprint epoch date |
| `PUT` | `/config/sprint_epoch` | Update sprint epoch. Body: `{"date": "YYYY-MM-DD"}` |
| `GET` | `/health` | Health check |

### Local Development

```bash
# First time
python3 -m venv .venv
pip install -r requirements.txt

# Every time
source .venv/bin/activate
set -a && source .env && set +a   # export vars from .env (copy from .env.example)

# Init DB and start server (use a local DB path)
DB_PATH=./dev.db python app.py
```

Test with a fake group ID (WhatsApp format required for chat_id validation):

```bash
CHAT="120363000000000001@g.us"

# Submit a screenshot
curl -X POST http://localhost:5000/score \
  -F "image=@/path/to/screenshot.jpg" \
  -F "user_id=972501234567@s.whatsapp.net" \
  -F "user_name=Alice" \
  -F "chat_id=$CHAT"

# Manually correct a failed scan
curl -X POST http://localhost:5000/score/correct \
  -H "Content-Type: application/json" \
  -d '{"user_id":"972501234567@s.whatsapp.net","score":5,"chat_id":"'"$CHAT"'"}'

# Leaderboard (current / previous sprint)
curl -sG "http://localhost:5000/leaderboard?sprint=current" --data-urlencode "chat_id=$CHAT"
curl -sG "http://localhost:5000/leaderboard?sprint=previous" --data-urlencode "chat_id=$CHAT"

# Personal stats
curl -sG "http://localhost:5000/stats?sprint=current" \
  --data-urlencode "user_id=972501234567@s.whatsapp.net" \
  --data-urlencode "chat_id=$CHAT"

# Sprint summary
curl -sG "http://localhost:5000/summary?sprint=current" --data-urlencode "chat_id=$CHAT"
curl -sG "http://localhost:5000/summary?sprint=previous" --data-urlencode "chat_id=$CHAT"

# Sprint end report (returns should_post=true only on sprint transition days)
curl -sG "http://localhost:5000/sprint/end_report" --data-urlencode "chat_id=$CHAT"

# Who hasn't submitted today?
curl -sG "http://localhost:5000/missing" --data-urlencode "chat_id=$CHAT"

# Sprint info
curl -s http://localhost:5000/sprint
```

### Curling the API (Production)

Use the picoclaw-gateway container (which has `curl`) to call the scorer.
This also verifies the Docker network between the two containers is working:

```bash
sudo docker exec -it picoclaw-gateway sh

CHAT="120363xxxxxxxxxxxxxxxxx@g.us"

# Health check
curl http://scorer:5000/health

# Current sprint info
curl http://scorer:5000/sprint

# Who hasn't submitted today?
curl -sG "http://scorer:5000/missing" --data-urlencode "chat_id=$CHAT"

# Leaderboard / summary
curl -sG "http://scorer:5000/leaderboard?sprint=current" --data-urlencode "chat_id=$CHAT"
curl -sG "http://scorer:5000/summary?sprint=previous" --data-urlencode "chat_id=$CHAT"

# Sprint end report
curl -sG "http://scorer:5000/sprint/end_report" --data-urlencode "chat_id=$CHAT"

# Update sprint epoch
curl -X PUT http://scorer:5000/config/sprint_epoch \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-02-06"}'

# Score an image manually (image must be inside the picoclaw-gateway container)
curl -X POST http://scorer:5000/score \
  -F "image=@/home/picoclaw/.picoclaw/workspace/media/some-image.jpg" \
  -F "user_id=972501234567@s.whatsapp.net" \
  -F "user_name=Alice" \
  -F "chat_id=$CHAT"
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
