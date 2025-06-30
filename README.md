# sample-finder

Problem Statement:
Producers and musicians face a growing challenge in navigating large, messy collections of royalty-free audio samples, where inconsistent or missing metadata (e.g., tempo, key, tags) makes it difficult to find the right sound for a creative project. The goal of this project is to build a semantic search engine that enables users to retrieve relevant audio samples using either natural language descriptions or reference audio clips. By leveraging audio feature extraction, machine learning, and embedding-based similarity search, the system aims to improve discoverability and creativity in music production workflows. This matters because better access to high-quality, usable samples accelerates creative output and democratizes music-making for producers of all skill levels.

Success Metric:
For text-based or audio-based similarity search, success will be measured using Mean Reciprocal Rank (MRR) and Precision@k on a human-labeled evaluation set of search queries and relevance judgments. If a classifier is used to infer tags or instrument types, macro-averaged F1-score will be used to evaluate multi-label classification performance.

## Milestones

- [ ] Scrape 1000+ samples from Freesound + Looperman
- [ ] Build audio feature extraction pipeline (librosa)
- [ ] Train and evaluate sample tag classifier (optional)
- [ ] Build semantic search (text & audio queries)
- [ ] Evaluate search quality (MRR, precision@k)
- [ ] Deploy search interface (CLI or Gradio)

⚠️ Note: Audio samples and metadata are downloaded locally via the Freesound API and are excluded from version control. Run `fetch_and_store()` in `freesound_api.py` to populate the data folder.
