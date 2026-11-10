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


def test_identical_prompt_resubmission_evaluates_last_token():
    # resubmitting a fully cached prompt must still evaluate one prompt token [TAG_PROMPT_LOGITS]
    # instead of keeping the whole common prefix and aborting on an empty batch
    global server
    server.start()

    prompt = "Once upon a time there was a little girl who liked to keep every prompt token."
    responses = []
    for _ in range(2):
        response = server.make_request("POST", "/v1/completions", data={
            "prompt": prompt,
            "cache_prompt": True,
            "id_slot": 0,
            "n_predict": 16,
        })
        assert response.status_code == 200
        responses.append(response.body)

    first, second = responses
    n_prompt = first["timings"]["prompt_n"]
    assert first["timings"]["cache_n"] == 0
    assert second["timings"]["prompt_n"] == 1
    assert second["timings"]["cache_n"] == n_prompt - 1
    health = server.make_request("GET", "/health")
    assert health.status_code == 200


def test_hybrid_prompt_checkpoint_is_reused_across_turns():
    # a hybrid recurrent model cannot roll its state back, so the follow-up turn must restore
    # the prompt checkpoint instead of re-processing the whole prompt, and match an uncached run
    global server
    server = ServerPreset.tinylfm3omni()
    server.n_ctx = 2048
    server.n_slots = 1
    server.cache_ram = 0
    server.ctx_checkpoints = 8
    server.checkpoint_min_spacing = 16
    server.temperature = 0.0
    server.start()

    common = (
        "You are testing a persistent prompt prefix. "
        "Keep this exact instruction and all numbered facts unchanged. "
        + " ".join(f"fact-{index}=value-{index}" for index in range(80))
    )
    prompt_a = f"{common}\nUser turn: summarize fact 3."
    prompt_b = f"{common}\nUser turn: summarize fact 41."

    def complete(prompt, cache_prompt, n_predict, n_probs=0):
        response = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "cache_prompt": cache_prompt,
            "id_slot": 0,
            "n_predict": n_predict,
            "n_probs": n_probs,
        })
        assert response.status_code == 200
        return response.body

    first = complete(prompt_a, True, 16)
    first_prompt_n = first["timings"]["prompt_n"]
    assert first_prompt_n > 80

    cached = complete(prompt_b, True, 1, n_probs=5)
    assert cached["timings"]["prompt_n"] < first_prompt_n // 2

    fresh = complete(prompt_b, False, 1, n_probs=5)
    assert fresh["timings"]["cache_n"] == 0

    top_cached = cached["completion_probabilities"][0]["top_logprobs"]
    top_fresh = fresh["completion_probabilities"][0]["top_logprobs"]
    assert [p["id"] for p in top_cached] == [p["id"] for p in top_fresh]
    for p_cached, p_fresh in zip(top_cached, top_fresh):
        assert p_cached["logprob"] == pytest.approx(p_fresh["logprob"], abs=1e-3)
