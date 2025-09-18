import os
import time
import asyncio
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random

import pytest
import requests

import joblib
import numpy as np
import tempfile
import xgboost
from sklearn.pipeline import Pipeline

# Base URLs for the dual-server architecture
PREDICTION_URL = os.getenv("PREDICTION_SERVER_URL", "http://0.0.0.0:8001")  # Update this
TRAINING_URL = os.getenv("TRAINING_SERVER_URL", "http://0.0.0.0:8000")  # Update this

# Helper to wait until the servers are ready
def wait_for_ready(url: str, timeout: float = 30.0, interval: float = 1.0):
    start = time.time()
    while True:
        try:
            r = requests.get(f"{url}/readyz", timeout=2.0)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.time() - start > timeout:
            pytest.skip(f"Server at {url} did not become ready in time")
        time.sleep(interval)

@pytest.fixture(scope="module", autouse=True)
def ensure_servers_ready():
    """Wait for both servers to be ready before running tests."""
    print("Waiting for training server...")
    wait_for_ready(TRAINING_URL)



def test_add_training_data_to_training_server():
    """Send training data to the training server."""
    entries = []
    
    # Generate 50 training samples with varied patterns for quantile learning
    for i in range(1, 2000):
        kv = random.uniform(0.1, 0.9)
        inp_len = random.randint(50, 500)
        waiting = random.randint(0, 10)
        running = random.randint(1, 5)
        tokens = random.randint(5, 50)
        prefix_cache = random.uniform(0.0, 1.0)
        
        # Generate synthetic latency data with realistic distributions
        # Higher variability to test quantile learning
        base_ttft = inp_len * 0.5 + waiting * 10 + running * 5 + kv * 20 + prefix_cache * 15 + 50
        base_tpot = kv * 50 + inp_len * 0.1 + tokens * 0.8 + running * 3 + 5
        
        # Add realistic noise (log-normal-ish distribution for latencies)
        noise_factor_ttft = random.lognormvariate(0, 0.3)  # Realistic latency noise
        noise_factor_tpot = random.lognormvariate(0, 0.2)
        
        entries.append({
            "kv_cache_percentage": kv,
            "input_token_length": inp_len,
            "num_request_waiting": waiting,
            "num_request_running": running,
            "actual_ttft_ms": max(1.0, base_ttft * noise_factor_ttft),
            "actual_tpot_ms": max(1.0, base_tpot * noise_factor_tpot),
            "num_tokens_generated": tokens,
            "prefix_cache_score": prefix_cache,
        })

    payload = {"entries": entries}
    print(len(entries))
    r = requests.post(f"{TRAINING_URL}/add_training_data_bulk", json=payload)
    assert r.status_code == 202, f"Expected 202, got {r.status_code}"
    print("Successfully sent realistic training data to training server")

test_add_training_data_to_training_server()