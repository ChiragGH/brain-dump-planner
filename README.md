# Brain Dump Planner

A small Flask prototype that turns an unstructured daily brain dump into a structured task plan with DeepSeek and can send tasks to Google Tasks.

## Setup

Install `requirements.txt`, set `DEEPSEEK_API_KEY` in your own environment, and run `python app.py`. Google Tasks integration requires your own OAuth `credentials.json` in the project directory. The app creates `token.json` locally after authorization.

No API key, OAuth file, or personal task data is included in this release. This is a local prototype; do not expose its Flask server to the internet.
