"""
Simple timer from https://github.com/jannerm/diffuser/blob/main/diffuser/utils/timer.py

Credits to
@inproceedings{dppo2024,
    title={Diffusion Policy Policy Optimization},
    author={Ren, Allen Z. and Lidard, Justin and Ankile, Lars L. and Simeonov, Anthony and Agrawal, Pulkit and Majumdar, Anirudha and Burchfiel, Benjamin and Dai, Hongkai and Simchowitz, Max},
    booktitle={arXiv preprint arXiv:2409.00588},
    year={2024}
}
"""

import time


class Timer:

    def __init__(self):
        self._start = time.time()

    def __call__(self, reset=True):
        now = time.time()
        diff = now - self._start
        if reset:
            self._start = now
        return diff