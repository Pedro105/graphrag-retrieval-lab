# 2wiki-dev

### Answer recall (bridge questions) — n=537, % with 95% CI

| Retriever | Units | 64 tok | 128 tok | 256 tok | 512 tok | 1024 tok |
|---|---|---|---|---|---|---|
| `dense_paragraph` | sentence | 0.7 <sub>0.2–1.5</sub> | 3.5 <sub>1.9–5.0</sub> | 14.2 <sub>11.4–17.3</sub> | 23.1 <sub>19.6–26.6</sub> | 35.9 <sub>32.0–40.2</sub> |
| `bm25_paragraph` | sentence | 1.1 <sub>0.4–2.0</sub> | 2.6 <sub>1.3–4.1</sub> | 11.0 <sub>8.4–13.8</sub> | 20.5 <sub>17.1–23.8</sub> | 30.5 <sub>26.6–34.5</sub> |
| `hybrid_paragraph` | sentence | 0.9 <sub>0.2–1.9</sub> | 3.7 <sub>2.0–5.6</sub> | 13.6 <sub>10.8–16.8</sub> | 25.0 <sub>21.6–28.9</sub> | 37.2 <sub>33.3–41.5</sub> |
| `dense_sentence` | sentence | 2.6 <sub>1.3–4.1</sub> | 9.5 <sub>7.1–11.9</sub> | 18.4 <sub>15.1–21.8</sub> | 26.3 <sub>22.5–30.0</sub> | 32.2 <sub>28.3–36.3</sub> |
| `dense_triple` | triple | 25.5 <sub>21.6–29.2</sub> | 32.6 <sub>28.5–36.9</sub> | 38.0 <sub>33.9–42.5</sub> | 45.6 <sub>41.3–49.7</sub> | 50.7 <sub>46.4–54.7</sub> |
| `graph_bfs_d1` | triple | 19.4 <sub>16.4–22.5</sub> | 24.6 <sub>20.9–28.3</sub> | 26.4 <sub>22.9–30.2</sub> | 26.4 <sub>22.9–30.2</sub> | 26.4 <sub>22.9–30.2</sub> |
| `graph_bfs_d2` | triple | 20.3 <sub>16.9–23.6</sub> | 61.5 <sub>57.5–65.7</sub> | 87.5 <sub>84.5–90.1</sub> | 92.2 <sub>89.8–94.2</sub> | 92.4 <sub>90.1–94.4</sub> |
| `graph_ppr` | triple | 38.2 <sub>34.1–42.5</sub> | 59.4 <sub>55.3–63.7</sub> | 76.7 <sub>73.0–80.3</sub> | 85.5 <sub>82.7–88.3</sub> | 86.8 <sub>84.0–89.6</sub> |
| `oracle_sentence` | sentence | 35.6 <sub>31.5–39.3</sub> | 92.7 <sub>90.5–94.8</sub> | 95.9 <sub>94.2–97.4</sub> | 95.9 <sub>94.2–97.4</sub> | 95.9 <sub>94.2–97.4</sub> |
| `oracle_triple` | triple | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> |

### Full reasoning chain retrieved (all questions) — n=1000, % with 95% CI

| Retriever | Units | 64 tok | 128 tok | 256 tok | 512 tok | 1024 tok |
|---|---|---|---|---|---|---|
| `dense_paragraph` | sentence | 3.4 <sub>2.3–4.6</sub> | 12.9 <sub>10.8–15.1</sub> | 25.4 <sub>22.8–28.1</sub> | 31.9 <sub>28.9–34.7</sub> | 38.0 <sub>34.8–41.1</sub> |
| `bm25_paragraph` | sentence | 3.2 <sub>2.2–4.3</sub> | 11.6 <sub>9.6–13.6</sub> | 21.1 <sub>18.6–23.7</sub> | 29.2 <sub>26.3–32.2</sub> | 35.8 <sub>33.0–39.0</sub> |
| `hybrid_paragraph` | sentence | 3.6 <sub>2.4–4.9</sub> | 13.3 <sub>11.3–15.5</sub> | 25.0 <sub>22.3–27.7</sub> | 34.2 <sub>31.4–37.2</sub> | 40.1 <sub>37.0–43.3</sub> |
| `dense_sentence` | sentence | 7.6 <sub>6.0–9.4</sub> | 19.6 <sub>17.2–22.1</sub> | 25.1 <sub>22.3–27.9</sub> | 28.2 <sub>25.5–31.2</sub> | 30.5 <sub>27.8–33.4</sub> |
| `dense_triple` | triple | 36.3 <sub>33.1–39.3</sub> | 39.8 <sub>36.5–43.0</sub> | 41.7 <sub>38.5–44.8</sub> | 44.9 <sub>41.9–48.3</sub> | 47.4 <sub>44.4–50.8</sub> |
| `graph_bfs_d1` | triple | 32.5 <sub>29.5–35.5</sub> | 36.0 <sub>32.8–39.1</sub> | 37.3 <sub>34.2–40.4</sub> | 37.3 <sub>34.2–40.4</sub> | 37.3 <sub>34.2–40.4</sub> |
| `graph_bfs_d2` | triple | 33.0 <sub>30.0–35.9</sub> | 57.7 <sub>54.7–60.8</sub> | 80.8 <sub>78.6–83.2</sub> | 87.0 <sub>85.0–89.0</sub> | 87.5 <sub>85.5–89.5</sub> |
| `graph_ppr` | triple | 32.5 <sub>29.6–35.4</sub> | 49.8 <sub>46.7–52.8</sub> | 67.4 <sub>64.7–70.5</sub> | 78.2 <sub>75.7–80.8</sub> | 79.7 <sub>77.4–82.2</sub> |
| `oracle_sentence` | sentence | 30.8 <sub>28.1–33.8</sub> | 84.8 <sub>82.4–87.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> |
| `oracle_triple` | triple | 99.9 <sub>99.7–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> |

### Gold units retrieved (all questions) — n=1000, % with 95% CI

| Retriever | Units | 64 tok | 128 tok | 256 tok | 512 tok | 1024 tok |
|---|---|---|---|---|---|---|
| `dense_paragraph` | sentence | 35.9 <sub>34.5–37.4</sub> | 51.2 <sub>49.8–52.7</sub> | 60.8 <sub>59.3–62.3</sub> | 65.2 <sub>63.7–66.8</sub> | 68.8 <sub>67.2–70.4</sub> |
| `bm25_paragraph` | sentence | 33.9 <sub>32.4–35.4</sub> | 48.2 <sub>46.8–49.8</sub> | 57.0 <sub>55.3–58.6</sub> | 63.2 <sub>61.7–64.8</sub> | 67.9 <sub>66.4–69.6</sub> |
| `hybrid_paragraph` | sentence | 35.9 <sub>34.4–37.4</sub> | 51.6 <sub>50.2–53.1</sub> | 61.1 <sub>59.7–62.6</sub> | 66.9 <sub>65.5–68.4</sub> | 70.4 <sub>68.9–72.1</sub> |
| `dense_sentence` | sentence | 33.3 <sub>31.6–35.2</sub> | 46.1 <sub>44.1–48.2</sub> | 51.7 <sub>49.7–53.8</sub> | 54.7 <sub>52.8–56.8</sub> | 56.7 <sub>54.7–58.8</sub> |
| `dense_triple` | triple | 67.7 <sub>66.0–69.3</sub> | 69.5 <sub>67.9–71.1</sub> | 70.5 <sub>68.9–72.1</sub> | 72.2 <sub>70.6–73.9</sub> | 73.5 <sub>71.9–75.2</sub> |
| `graph_bfs_d1` | triple | 64.6 <sub>63.0–66.2</sub> | 67.2 <sub>65.5–68.7</sub> | 67.9 <sub>66.2–69.5</sub> | 67.9 <sub>66.2–69.5</sub> | 67.9 <sub>66.2–69.5</sub> |
| `graph_bfs_d2` | triple | 64.9 <sub>63.2–66.5</sub> | 81.1 <sub>79.6–82.6</sub> | 92.1 <sub>91.0–93.2</sub> | 94.5 <sub>93.5–95.5</sub> | 94.7 <sub>93.7–95.7</sub> |
| `graph_ppr` | triple | 53.7 <sub>51.2–55.9</sub> | 70.3 <sub>68.2–72.3</sub> | 82.9 <sub>81.1–84.7</sub> | 88.7 <sub>87.2–90.3</sub> | 89.6 <sub>88.1–91.1</sub> |
| `oracle_sentence` | sentence | 60.7 <sub>59.0–62.6</sub> | 94.9 <sub>94.0–95.7</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> |
| `oracle_triple` | triple | 100.0 <sub>99.9–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> | 100.0 <sub>100.0–100.0</sub> |

### Entity linking: topic entity among the seeds

| Retriever | Seed hit |
|---|---|
| `graph_bfs_d1` | 97.8% |
| `graph_bfs_d2` | 97.8% |
| `graph_ppr` | 97.8% |

### Full reasoning chain retrieved by question type — 256 tokens, %

| Retriever | bridge_comparison (n=217) | comparison (n=246) | compositional (n=419) | inference (n=118) |
|---|---|---|---|---|
| `dense_paragraph` | 0.0 | 82.1 | 9.8 | 9.3 |
| `bm25_paragraph` | 0.0 | 72.8 | 5.0 | 9.3 |
| `hybrid_paragraph` | 0.0 | 82.1 | 8.1 | 11.9 |
| `dense_sentence` | 0.0 | 83.7 | 8.1 | 9.3 |
| `dense_triple` | 0.0 | 100.0 | 28.2 | 44.9 |
| `graph_bfs_d1` | 0.0 | 99.6 | 20.0 | 37.3 |
| `graph_bfs_d2` | 43.8 | 99.6 | 86.2 | 90.7 |
| `graph_ppr` | 25.8 | 86.2 | 72.3 | 87.3 |
| `oracle_sentence` | 100.0 | 100.0 | 100.0 | 100.0 |
| `oracle_triple` | 100.0 | 100.0 | 100.0 | 100.0 |
