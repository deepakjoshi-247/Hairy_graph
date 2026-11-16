import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from hairy_graph import config
from hairy_graph.database import (
    get_all_active_bubbles,
    get_bubble_by_id,
    get_connection,
    insert_bubble_link,
)
from hairy_graph.graph_engine import (
    compute_similarity,
    estimate_tokens,
    extract_extractive_summary,
    extract_keywords,
    fit_shared_vectorizer,
    get_bubble_search_text,
)
from hairy_graph.main import app


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Ensure every test runs against a clean, isolated temporary SQLite database."""
    db_file = tmp_path / "test_hairy_graph.db"
    monkeypatch.setenv("MEMORY_DB_PATH", str(db_file))
    # Unset auth by default so tests run open unless explicitly enabled
    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    return str(db_file)


# 1. Cold start with < 10 active bubbles
def test_cold_start_under_ten_bubbles(temp_db):
    with TestClient(app) as client:
        # Save 3 distinct memories (distinct vocabularies so no supersede occurs)
        facts = [
            ("Mercury orbital characteristics", "Fact 1 notes that Mercury rotates slowly and lacks any moons."),
            ("Octopus cardiovascular biology", "Fact 2 explains that octopus blood is blue because of hemocyanin."),
            ("Honey preservation properties", "Fact 3 describes how low moisture and acidity protect honey from spoiling."),
        ]
        for p, r in facts:
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200

        # Query in cold start mode
        query_res = client.post(
            "/get_context",
            json={"query": "tell me facts", "max_tokens": 1500},
        )
        assert query_res.status_code == 200
        data = query_res.json()

        assert data["mode"] == "cold_start"
        assert data["debug_trace"] == []
        assert data["tokens_used"] <= 1500

        # Context must be ordered oldest -> newest
        context = data["context"]
        idx1 = context.find("Fact 1")
        idx2 = context.find("Fact 2")
        idx3 = context.find("Fact 3")
        assert idx1 != -1 and idx2 != -1 and idx3 != -1
        assert idx1 < idx2 < idx3

        # Test tiny token budget (word truncation)
        tiny_res = client.post(
            "/get_context",
            json={"query": "tell me facts", "max_tokens": 4},
        )
        assert tiny_res.status_code == 200
        tiny_data = tiny_res.json()
        assert tiny_data["mode"] == "cold_start"
        assert tiny_data["tokens_used"] <= 4


# 2. Transition to graph mode at >= 10 active bubbles and flip back on archive
def test_transition_to_graph_mode(temp_db):
    with TestClient(app) as client:
        # Ingest 9 distinct geology bubbles
        geology_notes = [
            ("basalt lava formation", "Basalt forms from rapid cooling of magnesium-rich lava at the earth surface."),
            ("granite magma intrusion", "Granite crystallizes slowly beneath the surface from silica-rich magma."),
            ("sandstone quartz grains", "Sandstone consists of mineral particles cemented together by silica or calcite."),
            ("limestone calcite fossils", "Limestone forms mainly from skeletal fragments of marine organisms."),
            ("shale clay mudstone", "Shale is a fine-grained sedimentary rock composed of mud and clay minerals."),
            ("marble metamorphic limestone", "Marble develops when limestone is subjected to intense heat and pressure."),
            ("slate foliated metamorphic", "Slate derives from shale through low-grade regional metamorphism."),
            ("quartzite sandstone recrystallization", "Quartzite is a hard metamorphic rock originally composed of pure quartz sandstone."),
            ("obsidian volcanic glass", "Obsidian forms when felsic lava cools rapidly with minimal crystal growth."),
            ("pumice vesicular texture", "Pumice is a light porous volcanic rock produced during explosive eruptions."),
        ]

        for i in range(9):
            p, r = geology_notes[i]
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200

        # 9 active bubbles -> cold start
        q9 = client.post("/get_context", json={"query": "geology minerals", "max_tokens": 1500})
        assert q9.json()["mode"] == "cold_start"

        # Ingest 10th bubble
        p10, r10 = geology_notes[9]
        res10 = client.post("/save_memory", json={"prompt": p10, "response": r10})
        assert res10.status_code == 200
        b10_id = res10.json()["bubble_id"]

        # 10 active bubbles -> transitions to graph mode
        q10 = client.post("/get_context", json={"query": "pumice volcanic rock eruptions", "max_tokens": 1500})
        assert q10.json()["mode"] == "graph"
        assert len(q10.json()["debug_trace"]) > 0

        # Archive last active bubble -> count drops to 9
        arch_res = client.post("/archive_last")
        assert arch_res.status_code == 200
        assert arch_res.json()["archived_bubble_id"] == b10_id

        # Back to cold start mode
        q_back = client.post("/get_context", json={"query": "geology minerals", "max_tokens": 1500})
        assert q_back.json()["mode"] == "cold_start"


# 3. Graph routing and debug_trace verification
def test_graph_routing_and_debug_trace(temp_db):
    with TestClient(app) as client:
        # Ingest 5 Quantum Physics bubbles
        physics_texts = [
            ("quantum physics entanglement", "Entanglement links two photons across distances instantaneously."),
            ("qubit superposition states", "Superposition allows qubits to exist in multiple states simultaneously."),
            ("quantum decoherence noise", "Decoherence happens when quantum states interact with surrounding noise."),
            ("quantum circuit algorithms", "Algorithms like Shor and Grover exploit quantum parallelism."),
            ("quantum teleportation protocol", "Teleportation transmits quantum information using shared entangled pairs."),
        ]
        # Ingest 5 Culinary Sourdough bubbles
        baking_texts = [
            ("sourdough fermentation starter", "Wild yeast and lactobacilli produce acid during dough fermentation."),
            ("flour hydration ratio", "High hydration dough yields an open crumb structure in artisanal bread."),
            ("gluten development folding", "Stretch and fold techniques develop gluten elasticity without kneading."),
            ("baking dutch oven steam", "Trapping steam inside a hot dutch oven produces a crispy blistered crust."),
            ("scoring bread loaf ear", "A razor blade scores the dough surface to direct oven spring expansion."),
        ]

        for p, r in physics_texts:
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200

        for p, r in baking_texts:
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200

        # Precondition check using the engine's vectorizer
        all_texts = [p + " " + r for p, r in physics_texts + baking_texts]
        vec = fit_shared_vectorizer(all_texts)
        assert vec is not None

        query_physics = "quantum entanglement qubit photon"
        phys_sim = compute_similarity(vec, query_physics, physics_texts[0][0] + " " + physics_texts[0][1])
        bake_sim = compute_similarity(vec, query_physics, baking_texts[0][0] + " " + baking_texts[0][1])
        # Assert fixture precondition: physics similarity is substantially higher than baking
        assert phys_sim > 0.30
        assert phys_sim > bake_sim

        # Run query targeting quantum physics
        res = client.post("/get_context", json={"query": query_physics, "max_tokens": 1500})
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "graph"

        # Check debug_trace contains required fields
        trace = data["debug_trace"]
        assert len(trace) > 0
        cluster_entries = [t for t in trace if t["type"] == "cluster"]
        bubble_entries = [t for t in trace if t["type"] == "bubble"]

        assert len(cluster_entries) >= 1
        assert len(bubble_entries) >= 1

        for t in trace:
            assert "type" in t
            assert "id" in t
            assert "cluster_id" in t
            assert "similarity" in t
            assert "node_type" in t
            assert "depth" in t
            assert "included" in t

        # Verify physics content was returned
        assert "entanglement" in data["context"].lower() or "qubit" in data["context"].lower()


# 4. Cross-cluster bridging
def test_cross_cluster_bridging(temp_db):
    with TestClient(app) as client:
        # Ingest 5 distinct Astronomy bubbles (Cluster A)
        astronomy_notes = [
            ("hubble telescope deep field observations", "Deep space telescopes photograph spiral galaxies and cosmic dust clouds."),
            ("supernova remnant nebula gas expansion", "Exploding massive stars leave behind expanding shells of ionized gas."),
            ("black hole gravitational event horizon", "Gravitational singularity prevents light from escaping beyond event horizon."),
            ("exoplanet atmospheric spectroscopy transit", "Starlight filtering through exoplanet atmospheres reveals chemical signatures."),
            ("pulsar rotating neutron star beams", "Rapidly rotating neutron stars emit periodic beams of electromagnetic radiation."),
        ]
        astronomy_ids = []
        for p, r in astronomy_notes:
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200
            astronomy_ids.append(res.json()["bubble_id"])

        # Ingest 5 distinct Botany bubbles (Cluster B)
        botany_notes = [
            ("chloroplast photosynthesis light reaction", "Chlorophyll pigments in plant leaves absorb photons to synthesize carbohydrates."),
            ("vascular xylem phloem fluid transport", "Vascular tissues conduct water from roots and distribute sugars to foliage."),
            ("angiosperm flower pollination fertilization", "Flowering plants rely on insect pollinators for successful fertilization and seeds."),
            ("root rhizosphere mycorrhizal fungi interaction", "Symbiotic fungal networks enhance nutrient absorption in plant root systems."),
            ("cellular stomata gas exchange transpiration", "Microscopic leaf pores open and close to balance gas exchange and water loss."),
        ]
        botany_ids = []
        for p, r in botany_notes:
            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200
            botany_ids.append(res.json()["bubble_id"])

        # Link astronomy_ids[0] and botany_ids[0] with a bridging edge in a committed transaction
        conn = get_connection()
        with conn:
            insert_bubble_link(conn, astronomy_ids[0], botany_ids[0], weight=0.50)
        conn.close()

        # Query Astronomy; the bridging edge should pull in botany_ids[0] during graph traversal
        res = client.post("/get_context", json={"query": "hubble telescope deep space galaxies", "max_tokens": 1500})
        assert res.status_code == 200
        data = res.json()
        assert data["mode"] == "graph"

        visited_bubble_ids = {t["id"] for t in data["debug_trace"] if t["type"] == "bubble"}
        assert astronomy_ids[0] in visited_bubble_ids
        assert botany_ids[0] in visited_bubble_ids, "Bridging edge should pull in botany node from different cluster"


# 5. Strict token cap enforcement
@pytest.mark.parametrize("max_tokens", [1, 3, 7, 25, 80, 500, 1500])
def test_strict_token_cap_enforcement(temp_db, max_tokens):
    with TestClient(app) as client:
        # Ingest 12 distinct memories so both modes can be evaluated
        for i in range(12):
            client.post(
                "/save_memory",
                json={
                    "prompt": f"Detail unique subject {i} software architecture",
                    "response": f"Architectural pattern {i} emphasizes decoupling and modularity across distinct system components {i}.",
                },
            )

        # Test Graph mode token cap
        res_graph = client.post("/get_context", json={"query": "architectural modularity", "max_tokens": max_tokens})
        assert res_graph.status_code == 200
        data_graph = res_graph.json()
        assert data_graph["tokens_used"] <= max_tokens
        if data_graph["context"]:
            assert estimate_tokens(data_graph["context"]) <= max_tokens

        # Archive down to cold start mode (< 10)
        for _ in range(4):
            client.post("/archive_last")

        res_cold = client.post("/get_context", json={"query": "architectural modularity", "max_tokens": max_tokens})
        assert res_cold.status_code == 200
        data_cold = res_cold.json()
        assert data_cold["mode"] == "cold_start"
        assert data_cold["tokens_used"] <= max_tokens
        if data_cold["context"]:
            assert estimate_tokens(data_cold["context"]) <= max_tokens


# 6. Auto-promotion and CORE protection
def test_auto_promotion_and_core_protection(temp_db):
    with TestClient(app) as client:
        # Ingest target STATE bubble S1
        s1_prompt = "robotics kinematic arm actuator servo control"
        s1_resp = "Robotics kinematic arm actuator servo control algorithms calculate joint angles."
        res1 = client.post("/save_memory", json={"prompt": s1_prompt, "response": s1_resp, "pin_as_core": False})
        assert res1.status_code == 200
        s1_id = res1.json()["bubble_id"]
        assert res1.json()["node_type"] == "STATE"

        # 5 subsequent bubbles designed to have 0.35 <= similarity <= 0.80 with S1
        related_bubbles = [
            ("robotics kinematic arm actuator servo control", "Robotics kinematic arm actuator servo control calibrates joint angles step alpha."),
            ("robotics kinematic arm actuator servo control", "Robotics kinematic arm actuator servo control monitors joint angles step beta."),
            ("robotics kinematic arm actuator servo control", "Robotics kinematic arm actuator servo control balances joint angles step gamma."),
            ("robotics kinematic arm actuator servo control", "Robotics kinematic arm actuator servo control streams joint angles step delta."),
            ("robotics kinematic arm actuator servo control", "Robotics kinematic arm actuator servo control models joint angles step epsilon."),
        ]

        conn = get_connection()
        for idx, (p, r) in enumerate(related_bubbles, 1):
            # Assert similarity precondition with S1 using the engine's vectorizer
            active_bubbles = get_all_active_bubbles(conn)
            active_corpus = [get_bubble_search_text(b) for b in active_bubbles]
            kw_rel = extract_keywords(p + " " + r, active_corpus)
            sum_rel = extract_extractive_summary(p, r)
            t_rel = sum_rel + " " + " ".join(kw_rel)

            s1_b = get_bubble_by_id(conn, s1_id)
            s1_text = get_bubble_search_text(s1_b)

            vec = fit_shared_vectorizer(active_corpus + [t_rel])
            sim = compute_similarity(vec, t_rel, s1_text)
            assert (
                config.LINK_SIMILARITY_MIN <= sim <= config.LINK_SIMILARITY_MAX
            ), f"Precondition failed on call {idx}: similarity {sim} not in [0.35, 0.80]"

            res = client.post("/save_memory", json={"prompt": p, "response": r})
            assert res.status_code == 200

            # On the 5th linking bubble, S1 incoming_links reaches 5 -> auto-promoted to CORE
            if idx == 5:
                assert s1_id in res.json()["promoted_node_ids"]

        # Verify S1 is now CORE in DB
        s1_record = get_bubble_by_id(conn, s1_id)
        assert s1_record is not None
        assert s1_record.node_type == "CORE"
        assert s1_record.incoming_links >= 5

        # Now test CORE protection: ingest identical duplicate with similarity > 0.80
        # For a CORE bubble, it must NOT be archived, and incoming_links must increment
        dup_prompt = s1_prompt
        dup_resp = s1_resp

        active_bubbles = get_all_active_bubbles(conn)
        active_corpus = [get_bubble_search_text(b) for b in active_bubbles]
        kw_dup = extract_keywords(dup_prompt + " " + dup_resp, active_corpus)
        sum_dup = extract_extractive_summary(dup_prompt, dup_resp)
        t_dup = sum_dup + " " + " ".join(kw_dup)
        s1_text = get_bubble_search_text(s1_record)

        vec_dup = fit_shared_vectorizer(active_corpus + [t_dup])
        sim_dup = compute_similarity(vec_dup, t_dup, s1_text)
        assert sim_dup > config.SUPERSEDE_THRESHOLD

        res_dup = client.post("/save_memory", json={"prompt": dup_prompt, "response": dup_resp})
        assert res_dup.status_code == 200
        # S1 must NOT be in archived_node_ids
        assert s1_id not in res_dup.json()["archived_node_ids"]

        # Verify S1 remains active CORE and incoming_links was incremented
        s1_after = get_bubble_by_id(conn, s1_id)
        assert s1_after.status == "active"
        assert s1_after.node_type == "CORE"
        assert s1_after.incoming_links == s1_record.incoming_links + 1
        conn.close()


# 7. Supersede and undo (/archive_last)
def test_supersede_and_archive_undo(temp_db):
    with TestClient(app) as client:
        # Ingest STATE bubble S1
        p1 = "kubernetes deployment replica set pod"
        r1 = "Controller manages replica pods across cluster nodes."
        res1 = client.post("/save_memory", json={"prompt": p1, "response": r1, "pin_as_core": False})
        s1_id = res1.json()["bubble_id"]

        # Ingest S2 with identical text (similarity > 0.80) to supersede S1
        p2 = p1
        r2 = r1

        kw1 = extract_keywords(p1 + " " + r1, [])
        sum1 = extract_extractive_summary(p1, r1)
        t1 = sum1 + " " + " ".join(kw1)

        vec = fit_shared_vectorizer([t1, t1])
        sim = compute_similarity(vec, t1, t1)
        assert sim > config.SUPERSEDE_THRESHOLD, f"Precondition failed: similarity {sim} not > 0.80"

        res2 = client.post("/save_memory", json={"prompt": p2, "response": r2})
        s2_id = res2.json()["bubble_id"]
        # S1 must be superseded (archived)
        assert s1_id in res2.json()["archived_node_ids"]

        conn = get_connection()
        assert get_bubble_by_id(conn, s1_id).status == "archived"
        assert get_bubble_by_id(conn, s2_id).status == "active"

        # Call /archive_last -> archives S2 (most recent active STATE bubble)
        arch1 = client.post("/archive_last")
        assert arch1.status_code == 200
        assert arch1.json()["archived_bubble_id"] == s2_id
        assert get_bubble_by_id(conn, s2_id).status == "archived"

        # Call /archive_last again when no active STATE bubbles remain
        arch2 = client.post("/archive_last")
        assert arch2.status_code == 200
        assert arch2.json()["archived_bubble_id"] is None
        assert "No active STATE bubble" in arch2.json()["message"]
        conn.close()


# 8. Edge cases and authentication
def test_edge_cases_and_auth(temp_db, monkeypatch):
    with TestClient(app) as client:
        # Edge case A: Empty database query
        res_empty = client.post("/get_context", json={"query": "anything", "max_tokens": 1500})
        assert res_empty.status_code == 200
        assert res_empty.json()["mode"] == "cold_start"
        assert res_empty.json()["context"] == ""
        assert res_empty.json()["tokens_used"] == 0

        # Edge case B: Query with empty vocabulary / only stop words
        res_stopwords = client.post("/get_context", json={"query": "the and of in on", "max_tokens": 1500})
        assert res_stopwords.status_code == 200

        # Edge case C: Input validation
        res_empty_q = client.post("/get_context", json={"query": "   ", "max_tokens": 1500})
        assert res_empty_q.status_code == 422

        res_zero_tok = client.post("/get_context", json={"query": "valid query", "max_tokens": 0})
        assert res_zero_tok.status_code == 422

        res_big_tok = client.post("/get_context", json={"query": "valid query", "max_tokens": 9000})
        assert res_big_tok.status_code == 422

        res_empty_p = client.post("/save_memory", json={"prompt": "  ", "response": "valid"})
        assert res_empty_p.status_code == 422

        # Edge case D: Authentication when MEMORY_API_KEY is configured
        test_key = "super-secret-key-12345"
        monkeypatch.setenv(config.API_KEY_ENV_VAR, test_key)

        # /health is always open
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json() == {"status": "ok"}

        # /get_context without auth key -> 401
        unauth_res = client.post("/get_context", json={"query": "test query", "max_tokens": 1500})
        assert unauth_res.status_code == 401

        # /get_context with wrong key -> 401
        wrong_res = client.post(
            "/get_context",
            headers={"X-API-Key": "wrong-key"},
            json={"query": "test query", "max_tokens": 1500},
        )
        assert wrong_res.status_code == 401

        # /get_context with correct key -> 200
        auth_res = client.post(
            "/get_context",
            headers={"X-API-Key": test_key},
            json={"query": "test query", "max_tokens": 1500},
        )
        assert auth_res.status_code == 200
