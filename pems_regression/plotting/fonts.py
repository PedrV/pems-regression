import os
import tempfile

import matplotlib.font_manager
import matplotlib.pyplot as plt
import requests


def setup_monospace_font():
    """
    Downloads and sets up the Courier Prime font for matplotlib.
    """

    font_url = "https://github.com/google/fonts/raw/main/ofl/courierprime/CourierPrime-Regular.ttf"
    with tempfile.TemporaryDirectory() as tmpdir:
        font_path = os.path.join(tmpdir, "CourierPrime-Regular.ttf")

        # Step 1: Download font to temp directory
        response = requests.get(font_url, stream=True)
        if response.status_code == 200:
            with open(font_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Font download failed with status {response.status_code}")

        # Step 2: Register font with matplotlib
        matplotlib.font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = ["monospace"]
        plt.rcParams["font.monospace"] = ["Courier Prime"]
        plt.rcParams["figure.dpi"] = 200
