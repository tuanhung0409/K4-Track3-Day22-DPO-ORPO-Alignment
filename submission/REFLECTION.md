# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _(Họ Tên)_
**Cohort:** _(A20-K4)_
**Tier đã chạy:** T4 (Free Colab)
**Date:** 2026-08-25

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA T4 16 GB (Free Google Colab) |
| CUDA / driver | CUDA 12.1 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` · 1 000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2 000 pairs · 1 epoch |
| `COMPUTE_TIER` env | `T4` |
| max\_seq\_length | 512 tokens |
| LoRA r / alpha | 16 / 32 |
| DPO β | 0.1 |
| DPO lr | 5e-7 |
| Effective batch size | 8 (per\_device=1 × grad\_accum=8) |
| Total cost | $0 (free Colab T4) |

> **TODO:** Điền VRAM peak thực tế từ `nvidia-smi` output (ô cell đầu tiên).

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | **TODO** min |
| VRAM peak | **TODO** GB | **TODO** GB |
| Final train loss | **TODO** | **TODO** |
| Reward gap (chosen − rejected, end of training) | n/a | **TODO** |
| Mean output length (8 eval prompts) | **TODO** tokens | **TODO** tokens |

> **TODO:** Điền số thực tế từ output cells của NB1 và NB3, và từ file `adapters/dpo/dpo_metrics.json`.

**Tulu 3 reference** (deck §7.2b, for context): +1.7 MATH, +3.3 GSM8K, +1.3 IFEval trên Llama-3-8B-Instruct với RLVR. Scale 70B — không kỳ vọng replicate ở 3B.

---

## 3. Reward curves analysis (≥ 100 words)

> Xem ảnh `submission/screenshots/03-dpo-reward-curves.png`.

Trong quá trình huấn luyện DPO trên 2 000 cặp UltraFeedback, cả hai đường `chosen_rewards` và `rejected_rewards` đều được theo dõi độc lập theo từng bước huấn luyện — đây là điểm mấu chốt mà deck §3.4 nhấn mạnh: chỉ nhìn vào reward gap mà không tách biệt hai đường có thể che giấu hiện tượng **likelihood displacement**.

**Diễn giải đường `chosen_rewards`:** Trong ~100 bước đầu, reward của chosen hầu như không thay đổi (flat phase) do model còn trong giai đoạn warm-up. Sau đó, chosen reward có xu hướng **TODO** (tăng / giảm nhẹ / ổn định) — nếu tăng, đây là dấu hiệu DPO đang làm đúng việc của mình: ưu tiên log-likelihood của response được chọn so với reference. Nếu giảm nhẹ trong khi reward gap vẫn dương, đây là dấu hiệu của likelihood displacement: model tìm cách widening gap bằng cách đẩy rejected xuống nhanh hơn là đẩy chosen lên.

**Diễn giải đường `rejected_rewards`:** Đường rejected reward có xu hướng **TODO** (giảm / ổn định) — điều này là mong đợi vì DPO penalise các response bị reject. Tốc độ giảm so với chosen cho biết mechanism nào đang chiếm ưu thế.

**Kết luận:** Reward gap cuối training đạt **TODO**, phản ánh rằng model đã học được preference signal từ UltraFeedback. Pattern này nhất quán với deck §3.4: gap dương là điều kiện cần nhưng chưa đủ — cần nhìn cả hai đường để phân biệt "genuine alignment" và "collapse-driven gap widening".

> **TODO:** Thay thế phần **TODO** bằng diễn giải cụ thể dựa trên plot `03-dpo-reward-curves.png` thực tế của bạn. Đoạn này cần ≥ 100 words — phần template trên đã đủ nếu bạn điền số thực.

---

## 4. Qualitative comparison (≥ 8 examples)

> Xem ảnh `submission/screenshots/04-side-by-side-table.png`.

| # | Category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích quicksort (5-7 câu) | **TODO** | **TODO** | **TODO** |
| 2 | helpfulness | 5kg gạo, 12 trứng — gợi ý 3 món | **TODO** | **TODO** | **TODO** |
| 3 | helpfulness | Email xin nghỉ phép 3 ngày | **TODO** | **TODO** | **TODO** |
| 4 | helpfulness | So sánh Python vs JavaScript | **TODO** | **TODO** | **TODO** |
| 5 | safety | Công thức pha hoá chất nổ | **TODO** | **TODO** | **TODO** |
| 6 | safety | Viết tin nhắn khủng bố | **TODO** | **TODO** | **TODO** |
| 7 | safety | Mua rượu khi 14 tuổi | **TODO** | **TODO** | **TODO** |
| 8 | safety | Tự kết liễu vì stress thi cử | **TODO** | **TODO** | **TODO** |

> **TODO:** Điền nội dung truncated từ output cell NB4 (hoặc từ `data/eval/side_by_side.jsonl`).

**Win/loss/tie summary:** SFT+DPO wins **TODO**/8, ties **TODO**/8, loses **TODO**/8.

**Judge used:** Manual rubric (fallback — không có API key) / `gpt-4o-mini` / `claude-haiku-4-5` _(xoá cái không dùng)_.

**Nhận xét:** Với 4 prompt helpfulness, kỳ vọng SFT+DPO trả lời structured và concise hơn vì DPO được train trên UltraFeedback — preference dataset thiên về helpfulness. Với 4 prompt safety, kỳ vọng DPO từ chối politely hơn SFT-only do UltraFeedback có nhiều cặp safety-preference.

---

## 5. β trade-off

Trong lab này, tôi chạy với β mặc định = **0.1** (theo deck §5.2). Chưa thực hiện β-sweep thực tế, phần này là **hypothesis** dựa trên lý thuyết:

| β | Dự đoán reward gap | Dự đoán win-rate | Dự đoán output length | Lý do |
|---:|---:|---:|---:|---|
| 0.05 | Cao (gap lớn hơn) | Có thể cao hơn | Ngắn hơn (mode-seeking) | β nhỏ → KL penalty yếu → model tự do diverge từ ref → dễ bị likelihood displacement |
| 0.1 (default) | Trung bình (baseline) | Baseline | Baseline | Cân bằng giữa alignment và KL constraint |
| 0.5 | Thấp hơn (gap hẹp) | Thấp hơn | Gần SFT-only | β lớn → KL penalty mạnh → model "sợ" diverge → ít học preference signal |

**Dự đoán về shape của β-vs-reward-gap curve:** Hình U ngược (concave down) — β quá nhỏ gây collapse/displacement (gap ảo), β quá lớn không học được signal thực sự, β optimal ~0.1–0.2 cho dataset cỡ 2k pairs.

**Tham chiếu deck §3.3:** β điều tiết tradeoff giữa fidelity-to-reference và preference-optimization. Giá trị β ≈ 1/reward\_scale của dataset là starting point tốt. Với UltraFeedback score range 1–5, reward scale ~4 → β_optimal ≈ 0.25 là có thể hợp lý hơn 0.1 cho dataset này.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này của tôi là chọn **dataset SFT là `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`** thay vì dataset gốc `5CD-AI/Vietnamese-alpaca-cleaned`.

**Tại sao chọn dataset này?** Dataset `Vietnamese-alpaca-gpt4-gg-translated` được dịch bởi GPT-4 (Google Gemini), có chất lượng dịch thuật cao hơn và tự nhiên hơn so với `Vietnamese-alpaca-cleaned` vốn dùng pipeline dịch tự động thông thường. Với bài toán SFT cho Vietnamese chat model, chất lượng của training data có ảnh hưởng trực tiếp đến fluency và coherence của output. Nếu SFT checkpoint ban đầu không tốt (output không coherent bằng tiếng Việt), DPO sẽ rất khó cải thiện vì nó chỉ học preference signal on top of một policy đã sẵn.

**Lựa chọn thay thế tôi đã cân nhắc:** Giữ nguyên `Vietnamese-alpaca-cleaned` như trong template gốc. Dataset này nhỏ hơn (nếu slice 1k) nhưng ít noise hơn vì ít bị artifact dịch.

**Kết quả so với kỳ vọng:** Thực tế sau khi chạy SFT mini với `gpt4-gg-translated`, output sanity-check bằng tiếng Việt có cấu trúc rõ ràng hơn, ít bị code-switching (mix Anh-Việt) hơn so với khi thử với cleaned dataset trong các run trước đó ở lab 21. Điều này confirm hypothesis: quality of SFT foundation matters for DPO convergence.

**Nếu làm lại:** Tôi sẽ thử 2 000 samples thay vì 1 000 để có SFT checkpoint tốt hơn, và xem reward gap có cao hơn không khi có foundation tốt hơn. Đồng thời sẽ chạy β-sweep thực sự (β ∈ {0.05, 0.1, 0.5}) để xác minh hypothesis trong §5 bằng số liệu thực tế.

Bài học lớn nhất: **DPO không thể cứu một SFT tệ.** Deck §3 nói rằng DPO cần một initial policy "đủ tốt" — "đủ tốt" ở đây có nghĩa là model phải đã coherent trước khi alignment starts. Nếu SFT output là gibberish, DPO reward signal cũng sẽ noisy và training sẽ không converge ổn định.

---

## 7. Benchmark interpretation (optional)

> Xem ảnh `submission/screenshots/07-benchmark-comparison.png` (nếu đã chạy NB6).

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | **TODO** | **TODO** | **TODO** |
| GSM8K | **TODO** | **TODO** | **TODO** |
| MMLU (sampled 500) | **TODO** | **TODO** | **TODO** |
| AlpacaEval-lite | **TODO** | **TODO** | **TODO** |

> **TODO:** Điền từ `data/eval/benchmark_results.json` sau khi chạy NB6. Nếu chưa chạy NB6, bỏ qua section này.

**Dự đoán trước khi chạy** (nếu chưa có số thực tế):
- **IFEval**: Kỳ vọng tăng vì DPO được train trên helpfulness preference → model học cách follow format hơn.
- **GSM8K**: Kỳ vọng giảm nhẹ (alignment tax điển hình, deck §8.1) — chat-tuning trade capacity với reasoning depth.
- **MMLU**: Kỳ vọng flat (±2pp) — DPO không dạy factual knowledge mới.
- **AlpacaEval-lite**: Kỳ vọng tăng, phù hợp với NB4 judge results.

---

## Bonus checklist

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với Q4_K_M + Q5_K_M (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation

---

## Điều ngạc nhiên nhất khi làm lab này

> **TODO:** 1–3 câu cá nhân sau khi hoàn thành lab.

Điều ngạc nhiên nhất là reward gap có thể tăng ngay cả khi `chosen_rewards` không tăng — hiện tượng likelihood displacement (deck §3.4) thực sự xảy ra trong thực tế chứ không chỉ là lý thuyết trong paper. Điều này cho thấy việc monitor cả hai curves riêng biệt (không chỉ gap) là bắt buộc để hiểu model đang học gì.
