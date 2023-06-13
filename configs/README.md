# Correspondence between model names and configurations

All experiments were run with T5. The default random seed is 42 and can be changed as required.

**Main Results & Abaltions** (./main_ablations/*)
- UniPoll: UniPoll.json
- w.o. A: main_Q.json
- w.o. Q: main_A.json
- w.o. Q, A: main.json
- Q: Q.json
- A: A.json

**Few Comments Experiments** (./few_comments/*)

- UniPoll-few_[comments length %]: e.g., UniPoll-few_20.json
- Q-few_[comments length %]: e.g., Q-few_20.json
- A-few_[comments length %]: e.g., A-few_20.json

**Low-Resource Experiments** (./low_resource/*)
- UniPoll-low_[training data %]: e.g., UniPoll-low_20.json
- Q-low_[training data %]: e.g., Q-low_20.json
- A-low_[training data %]: e.g., A-low_20.json