# Local Persistent Test Deployment

This project includes small macOS launchd helpers for running a persistent local
Gradio test instance from a dedicated checkout.

## Recommended Setup

Use a separate folder from your active development checkout:

```bash
git clone --branch main --single-branch https://github.com/chunhualiao/co-scientist-loop \
  ~/workspace/open-ai-co-scientist-local-deploy
cd ~/workspace/open-ai-co-scientist-local-deploy
```

Start LM Studio's local server and load a model before launching the app. The
defaults use `http://127.0.0.1:1234/v1`; copy `.env.example` to `.env` when the
server address or model ID differs. A symlink to an existing gitignored `.env`
is also fine:

```bash
ln -s ~/workspace/open-ai-co-scientist/.env .env
```

Install and start the persistent service:

```bash
./local-deploy/install-launchagent.sh
```

Open:

```text
http://127.0.0.1:7860
```

## Refreshing The Local Site

After changes are pushed to `origin/main`, or after a manual `git pull`, restart
the service from the deployment checkout:

```bash
./local-deploy/restart.sh
```

This also works:

```bash
./local-deploy/refresh.sh
```

On startup, the service fetches `origin/main`, pulls it when the checkout is on
`main`, installs current requirements into `venv`, sources `.env`, and runs
`python app.py`.

## Useful Maintenance Commands

Check service state:

```bash
launchctl print gui/$(id -u)/com.liao.open-ai-co-scientist.local
```

Stop the service:

```bash
launchctl bootout gui/$(id -u) \
  ~/Library/LaunchAgents/com.liao.open-ai-co-scientist.local.plist
```

Follow app logs:

```bash
tail -f logs/local-deploy.log
```
