import torch
import numpy as np
import webrtcvad
from typing import Tuple
import time

from src.saletech.utils.logger import get_logger
from config.settings import AppSettings

logger = get_logger("saletech.vad.model")

