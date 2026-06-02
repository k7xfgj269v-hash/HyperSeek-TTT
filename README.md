# HyperSeek-TTT

Toy-Scale Implementierung der DeepSeek V4-Pro Architektur + 2026 Test-Time Learning (In-Place TTT). Pipeline läuft end-to-end; eigentliche Wirksamkeitsprüfung bleibt dem Scale-up vorbehalten.

## Architekturübersicht

**Backbone**:
- **MLA** — Multi-head Latent Attention mit decoupled RoPE (DeepSeek-V2)
- **DeepSeekMoE** — routed + shared experts, sigmoid + bias router, ohne aux loss (V2+V3)
- **mHC-lite** — Mehrkopien-Residuum via Birkhoff-von Neumann Zerlegung, A-Matrix exakt doppelt-stochastisch (arXiv 2601.05732)
- RMSNorm / RoPE / SwiGLU

**Top-Ebene**:
- **MTP** — Multi-Token Prediction als Hilfsverlust, 1 head mit eingebettetem komplettem DeepSeekBlock (V3)
- **InPlaceTTT** — Einzelner Linear-Adapter + Gate; hidden-delta NTP inner loss; optional persistent `W_mem` über Chunks akkumuliert (arXiv 2604.06169)
- **Muon + AdamW** — ndim≥2 nimmt Muon, bias / RMSNorm weight / token_emb nehmen AdamW (arXiv 2502.16982)

**Konkrete Parameter siehe `model.py` → `Config`** — alle skalenabhängigen Felder (d_model / n_layers / kv_lora_rank / moe_inter_dim / Trainings-Hyperparameter …) sind dort zentral; beim Scale-up nur eine Stelle ändern.

Bewusst ausgelassen: FP4-Training, DSA / CSA / HCA / Hybrid Attention (V4-Pros Long-Context-Dreigespann, bei Toy-Skala bedeutungslos).

## Dateistruktur

```
HyperSeek-TTT/
├── model.py        Architektur (Config + alle nn.Module)
├── data.py         Toy CoT Arithmetik Tokenizer + Dataset
├── optim.py        Muon Optimizer
├── train.py        Trainings-Einstieg (auto MPS, grad clip, monitor)
├── eval.py         Inferenz + Experten-Spezialisierungstest
├── memory_train.py Hidden-Secret Recall Training + 3-Modi Fair Compare
├── memory_eval.py  Episode Memory + Chunked Recall Ablation
├── transformer.py  v0 Baseline (MiniGPT), als Referenz erhalten
└── bin/python      venv python 3.14
```

## Verwendung

```bash
cd HyperSeek-TTT

# Haupttraining (auto MPS; Schritte / Batches / lr siehe cfg.train_*)
./bin/python train.py

# Inferenz + Experten-Spezialisierungstest
./bin/python eval.py

# Long-Context / Episode-Memory Validierungsgerüst (3-Modi Fair Compare)
./bin/python memory_eval.py --task kv     --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task needle --episodes 100 --chunk-len 32
./bin/python memory_eval.py --task rule   --episodes 100 --chunk-len 32

# Hidden-Secret Recall Training (3-Modi Fair Compare)
./bin/python memory_train.py --steps 300 --batch-size 16 --secret-len 1
./bin/python memory_train.py --steps 100 --batch-size 8  --secret-len 4
```

## Trainings-Monitoring

`train.py` druckt alle 100 Schritte eine Zeile:

```
Step  XXXX loss X.XXXX route X.XXXX mhc_amp avg X.XXX max X.XXX
                                         ↑ nur Backbone-HC gezählt, MTP-intern ausgeschlossen
```

Von jedem Modul exponierte Monitoring-Felder (in Echtzeit beim Forward aktualisiert):

| Modul | Feld | Zweck |
|---|---|---|
| HyperConnect | `last_amp` | Signal-Verstärkung, V4-Ziel ≈ 1.6 |
| HyperConnect | `last_alpha`, `last_beta`, `last_lambda` | Normalisierte Gewichtsverteilung |
| HyperConnect | `last_replica_norms` | n_hc=4 Kopien-Auslastung |
| MoE | `last_load`, `last_router_logits` | Expertenlast |
| InPlaceTTT | `last_gate`, `last_L` | Gate-Mittelwert + Inner Loss |
| InPlaceTTT | `last_delta_norm`, `last_h_norm`, `last_update_norm`, `last_mem_norm` | Memory- / Update-Stärke |

## Experimentdesign: memory_train 3-Modi Fair Compare

`memory_train.py` trainiert die Aufgabe "Secret im Context versteckt, Query gibt Secret aus" mit drei Modi:

| Modus | W_mem Verhalten | atlas.W Lernpfad |
|---|---|---|
| `none` | Festes `W_mem = 0`, weder lesen noch schreiben | Lernt via Query-Forward; atlas degeneriert zu statischem Linear + Gate |
| `transient` | Pro Chunk vom Zero-Base aus Inner Update; keine Akkumulation über Chunks | Backpropagation über Functional-Pfad |
| `persistent` | W_mem über Chunks akkumuliert, Reset vor Episode-Start | Backpropagation über Functional-Pfad |

Alle drei Modi gehen einheitlich durch `build_functional_memory` (Functional-Pfad). So liegt der Unterschied nur in "ob W_mem über Chunks akkumuliert wird", und der **Grenznutzen persistenter Fast Weights** ist wirklich messbar.

`memory_eval.py` führt drei Long-Context-Aufgaben (kv_recall / needle / rule) × drei Modi Ablation durch.

## InPlaceTTT vs ATLAS

Aktuell ist InPlaceTTT Default; `AtlasMemory` ist nicht im Source-Tree. Bei Bedarf gemäß [arXiv 2505.23735](https://arxiv.org/abs/2505.23735) Spezifikation als neue Class hinzufügen.

| | InPlaceTTT (aktuell) | ATLAS (Alternative) |
|---|---|---|
| Inner Loss | hidden-delta MSE: `\|\|W·h_t - (h_{t+1}-h_t)\|\|²` | Assoziatives Gedächtnis: `\|\|M(k_t) - v_t\|\|²` |
| Interne Struktur | Einzelner Linear + Gate | M (2-Schicht MLP) + W_k / W_v + Gate |
| Parameter-Größenordnung | Einmal `d_model²` | Etwa 5× `d_model²` |
| Designphilosophie | Bestehende Struktur wiederverwenden + Fast Weight | Eigenständiges Memory-Modul hinzufügen |
| Paper | 2026-04 | 2025-05 |

**Wann ATLAS empfohlen ist**:
- Aufgabe hat klare Key-Value-Assoziationsstruktur (Query-style Recall, explizites Lookup)
- Key-Treffer sind diskret, Assoziationsmuster lassen sich leichter lernen als hidden delta
- Größere Speicherkapazität benötigt (M ist 2-Schicht MLP, ausdrucksstärker als einzelnes Linear)
- Direkter Vergleich gegen ATLAS Paper Baseline gewünscht

**Wann InPlaceTTT empfohlen ist**:
- Generelle NTP-style Aufgaben
- Parametereffizienz im Vordergrund (Memory-Branch ~80% weniger Parameter)
- Anschluss an 2026-Mainstream (Behrouz et al. Nachfolger)

## Hinweise zum Scale-up

**Aktuell nur Pipeline-Verifikation, keine Wirksamkeitsprüfung** — der reale Wert von InPlaceTTT Test-Time Learning, mHC Kopien-Auslastung, MTP-Gewinn, Muon vs AdamW Vorteil etc. ist auf Toy-Skala **alles nicht erkennbar**. Erst Scale-up validiert es.

### Parameter-Tuning-Reihenfolge

1. **Zuerst Tiefe `n_layers` bestätigen** — schwache Multitask-Erhaltung kommt fundamental von flachen Modellen. Erster Schritt beim Scale-up: Tiefe erhöhen. MTP-Anteil sinkt mit Tiefe automatisch.
2. **`n_mtp_heads = 1` beibehalten** — V3 / V4 sind beide depth=1, nicht anrühren.
3. **`seq_len` Änderung erfordert Model-Rekonstruktion** — `rope_cos / causal_mask` sind Buffer, nicht dynamisch erweiterbar.
4. **MoE-Block je nach Skala anpassen** — `n_routed_experts` / `moe_inter_dim` proportional zur Modellgröße skalieren.

### Mögliche Erweiterungs-Richtungen (Hooks vorhanden, neue Logik nötig)

- **TTT Doppel-Gate**: Aktuell InPlaceTTT einzelnes Sigmoid-Gate; Read-Gate + Write-Gate trennen, um W_mem Lesen / Schreiben getrennt zu steuern
- **Reasoning Token Format**: Toy CoT inline; beim Scale-up `<think>...</think>` als explizites Segment-Token hinzufügen

### Trainings-Monitoring (alle 100 Schritte diese Vier im Set)

| Indikator | Worauf achten |
|---|---|
| **`loss`** | Monoton fallend; Spike / langes Plateau alarmieren |
| **`gate`** (`InPlaceTTT.last_gate`) | **Schlüsselindikator, ob TTT wirklich lernt**. Während Training auf > 0.1 → TTT wirkt; bleibt ≈ 0.02 → Leerlauf |
| **`route`** (Router-überwachtes CE) | Auf Toy konvergiert 1.0–1.5; fällt nicht → Router nicht differenziert |
| **`mhc_amp avg / max`** | avg ≈ 1.6 (V4-Ziel), max < 5; max > 10 → mHC alpha / beta / lambda abgedriftet, Signal-Verstärkung außer Kontrolle |

Monitoring-Felder sind exponiert; beim Scale-up Kurven über mehrere Schritte plotten ist am intuitivsten.

## Designmängel

1. **Toy-Arithmetik kann Test-Time Learning nicht validieren** — InPlaceTTT bei geschlossenen deterministischen Aufgaben Gate im Wesentlichen Leerlauf (keine langreichweitigen Abhängigkeiten zu erinnern). Validierung erfordert Long-Context needle-in-haystack oder assoziative Recall-Aufgaben + BPE Tokenizer.

2. **Anpfropfung auf DeepSeek-V2-Lite extrem kostspielig** — alle Dimensionen passen nicht, Tokenizer komplett anders, Architekturen stark unterschiedlich (V2-Lite hat kein mHC / MTP / InPlaceTTT). "Anpfropfung" ≈ Projekt neu schreiben.

3. **n_layers zu flach** — aktuell Toy-Sicherheitsgrenze, aber Multitask-Erhaltung könnte trotzdem versagen; MTP head Anteil bleibt groß, Trainingsdynamik durch MTP gestört. Scale-up auf ≥ 12 Schichten mildert automatisch.

4. **Zeichenbasierter Tokenizer, nur Arithmetik** — natürliche Sprache erfordert BPE 32K–50K, gesamte `data.py` neu schreiben.

5. **mHC-lite empfindlich gegenüber `n_hc`** — `n_hc=4` → 24 perms sicher; `n_hc≥8` → Buffer-Explosion, K-Cap-Subsampling nötig (gemäß Paper Teilmenge < n_hc!).

6. **`make_long_effective_mask`** — `raise NotImplementedError` Platzhalter, für zukünftige Long-Memory-Designs reserviert. Bei tatsächlicher Verwendung muss neu geschrieben werden.

7. **InPlaceTTT Inner Step über Samples geteilt** — `reshape(-1, D)` fasst Batch und Window zusammen für `L_mem` Berechnung. `memory_train.py` umgeht via Batch=1 seriell; Haupt-`train.py` mit Batch > 1 leidet weiterhin unter Cross-Sample Fast-Weight Verschmutzung.

8. **MTP mit komplettem eingebettetem DeepSeekBlock** — konsistent mit V3 Paper, aber bei Toy-Skala MTP-Anteil groß und stört Backbone-Trainingsdiagnose. Scale-up mildert automatisch.

## Versteckte Constraints (Stolperfallen)

- `model(idx, return_mtp=True)` muss `mtp_tokens=next_tok_ids` mitgeben (Fallback ist raise, nicht mehr silent wrong)
- `seq_len` Änderung erfordert anschließend Model-Rekonstruktion (`rope_cos / causal_mask` sind Buffer, nicht dynamisch erweiterbar). `memory_train.py` macht via `make_cfg(seq_len, ...)` neues cfg → neues Model — korrekte Vorgehensweise
- InPlaceTTTs `persistent_memory` wird bei der Konstruktion festgelegt (`cfg.ttt_persistent_memory`), kann nicht mid-training gewechselt werden; `memory_train.py` erstellt pro Modus ein neues Model
- Haupt-`train.py` verwendet Attributnamen `model.atlas` für InPlaceTTT-Instanz (Benennung aus Historie, state_dict-Kompatibilität nicht gebrochen; tatsächlicher Typ ist InPlaceTTT)

## Referenzen

- DeepSeek-V2: arXiv 2405.04434
- DeepSeek-V3 Technical Report: arXiv 2412.19437
- DeepSeek-V4-Pro: 2026-04-24 Release ([HF Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro))
- mHC (manifold-constrained): arXiv 2512.24880
- **mHC-lite** (in diesem Projekt): arXiv 2601.05732
- Muon Optimizer: arXiv 2502.16982
- ATLAS (alternative Memory-Branch): arXiv 2505.23735
- Titans / MIRAS: arXiv 2501.00663
- TNT (ATLAS Trainingseffizienz, ICLR 2026): arXiv 2511.07343
- **In-Place TTT** (in diesem Projekt): arXiv 2604.06169
- LLMs-from-scratch MLA Referenz: github.com/rasbt/LLMs-from-scratch
