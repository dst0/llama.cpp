import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_predict = 96
    server.cache_prompt = True
    server.cache_ram = 256
    server.cache_reuse = 16
    server.ctx_checkpoints = 8
    server.checkpoint_min_spacing = 16
    server.temperature = 0.0


def test_generation_checkpoints_do_not_destroy_reusable_prompt_prefix():
    global server
    server.start()

    common = (
        "You are testing a persistent prompt prefix. "
        "Keep this exact instruction and all numbered facts unchanged. "
        + " ".join(f"fact-{index}=value-{index}" for index in range(80))
    )
    prompt_a = f"{common}\nUser turn: summarize fact 3."
    prompt_b = f"{common}\nUser turn: summarize fact 41."
    prompt_c = f"{common}\nUser turn: summarize fact 99."

    first = server.make_request("POST", "/v1/completions", data={
        "prompt": prompt_a,
        "cache_prompt": True,
        "id_slot": 0,
        "n_predict": 96,
    })
    assert first.status_code == 200
    first_prompt_n = first.body["timings"]["prompt_n"]
    assert first_prompt_n > 80

    for prompt in (prompt_b, prompt_c):
        response = server.make_request("POST", "/v1/completions", data={
            "prompt": prompt,
            "cache_prompt": True,
            "id_slot": 0,
            "n_predict": 96,
        })
        assert response.status_code == 200
        assert response.body["timings"]["prompt_n"] < first_prompt_n // 2
        health = server.make_request("GET", "/health")
        assert health.status_code == 200
