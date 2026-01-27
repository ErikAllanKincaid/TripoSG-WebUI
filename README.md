# TripoSG-WebUI
### Enhancement to https://github.com/VAST-AI-Research/TripoSG to add a WebUI.

Clone this repo to get the puthon script for the WebUI.

Clone TripoSG, add these files,
`cp app.py pyproject.toml requirement.txt ~/code/TripoSG/`
Get it running in commandline, then run `uv run python app.py` to start the WebUI.

## Get TripoSG running
## TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models
```bash
mkdir -p ~/code/
cd code
git clone https://github.com/erikallankincaid/
cp app.py pyproject.toml requirement.txt ~/code/TripoSG/
git clone https://github.com/VAST-AI-Research/TripoSG.git
cd TripoSG
uv init
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-build-isolation diso
uv add -r requirements.txt
uv pip install numpy
uv pip install huggingface-hub
uv pip install -r requirements.txt
```
## Add an image to try.
`scp ./fist.jpg ops@192.168.1.43:/home/ops/code/TripoSG/`
## Run the generation model.
`IMAGE=fist.jpg ; uv run python -m scripts.inference_triposg --image-input /home/erik/code/TripoSG/$IMAGE`

## Start the WebUI
`cd /home/$USER/code/TripoSG/ && uv run python app.py`

## Make a service
`sudo cp triposg-webui.service /etc/systemd/system/` 
#### Copy service file to the service directory.
`sudo cp /home/$USER/code/TripoSG/triposg-webui.service  /etc/systemd/system/`
#### Reload systemd: Notify the service manager of the new file.
`sudo systemctl daemon-reload`
#### Start the service: Manually start the service for the first time.
`sudo systemctl start webui.service`
#### Check the service status: Verify that it is running correctly.
`sudo systemctl status webui.service`
#### Enable the service (optional): Configure the service to start automatically every time your system boots.
`sudo systemctl enable webui.service`
### Troubleshooting
If the service fails to start, use journalctl to view the logs and identify the error: 
`sudo journalctl -u webui.service -f`

