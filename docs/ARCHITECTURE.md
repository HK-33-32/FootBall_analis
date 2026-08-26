# Architecture

The system separates dense specialist perception from semantic adjudication. The supplied Football Core is accessed through its HTTP contract; its geometric event ledger is converted only into candidate windows. Qwen3-VL receives timestamped, deliberately sampled frames plus compact geometry and Player Memory context, and must return a schema-valid event or abstain.

Accepted semantic events, global players and source evidence are stored in SQLite. Statistics query those typed events rather than summaries or embeddings. The API and debug frontend expose model revision, confidence, second-pass reasons and a media link for every event.

The OpenAI-compatible VLM interface supports vLLM, SGLang or another explicit local server. There is no proprietary fallback. If perception or semantic inference is not configured, the requested production stage fails with an actionable error.

