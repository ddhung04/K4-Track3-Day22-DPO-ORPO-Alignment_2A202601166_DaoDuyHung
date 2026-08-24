# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Đào Duy Hưng_
**Cohort:** _A20-K3B_
**Tier đã chạy:** _GPU_
**Date:** _2026-08-24_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB VRAM |
| CUDA / driver | PyTorch 2.5.1+cu124; NVIDIA driver 610.88 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca`, cấu hình 1.000 samples, 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned`, cấu hình 2.000 pairs, 1 epoch |
| `COMPUTE_TIER` env | `T4` (cấu hình tiết kiệm VRAM: batch 1, gradient accumulation 8, max length 512) |
| Total cost | $0 local GPU |

**Trạng thái trung thực:** môi trường GPU đã được kiểm tra thành công. NB1 được khởi chạy nhưng đã dừng theo quyết định của người thực hiện trước khi train hoàn tất; vì vậy không có adapter, DPO metrics, reward curve hay đánh giá định lượng hợp lệ trong báo cáo này. Đây là bản nháp kỹ thuật, **không phải bằng chứng hoàn thành để nộp**.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | Chưa chạy |
| VRAM peak | Chưa ghi nhận kết thúc | Chưa chạy |
| Final loss | Chưa có (NB1 bị dừng trước khi lưu) | Chưa có |
| Reward gap (chosen − rejected, end of training) | n/a | Chưa có |
| Mean output length | Chưa có | Chưa có |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

`03_dpo_reward_curves.png` chưa tồn tại vì NB3 chưa được chạy đến bước train. Do đó tôi không diễn giải một đường cong không có thật, cũng không ghi reward gap hay loss được suy đoán. Khi chạy lại, tôi sẽ đọc ba tín hiệu cùng nhau: `chosen_rewards`, `rejected_rewards`, và hiệu của chúng. Một gap tăng chỉ là tín hiệu ban đầu, không đủ để kết luận mô hình tốt hơn. Nếu chosen tăng dần trong khi rejected giảm hoặc giữ thấp, điều đó cho thấy mô hình đang ưu tiên response được chọn. Ngược lại, nếu chosen giảm nhưng rejected giảm nhanh hơn, gap vẫn có thể tăng; đây là likelihood displacement và cần kiểm tra chất lượng đầu ra trước khi coi là thành công. Tôi cũng sẽ so sánh đoạn đầu và cuối của đường cong, loss DPO, độ dài response, và các ví dụ đánh giá độc lập. Với cấu hình hiện tại, beta 0.1 và learning rate 5e-7 là điểm xuất phát, không phải bằng chứng rằng reward gap sẽ tăng. Báo cáo chỉ được cập nhật số liệu sau khi `adapters/dpo/dpo_metrics.json` và ảnh đồ thị được tạo từ một run hoàn tất.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 2 | helpfulness | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 3 | helpfulness | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 4 | helpfulness | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 5 | safety | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 6 | safety | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 7 | safety | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |
| 8 | safety | Chưa tạo prompt/eval | Chưa chạy | Chưa chạy | Chưa chấm |

**Win/loss/tie summary:** Chưa có — không được suy đoán khi chưa có hai bộ output và rubric/judge.

**Judge used:** Chưa chọn. Run lại sẽ dùng manual rubric có tiêu chí factuality, helpfulness, safety và style để tránh phụ thuộc API key.

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | _<...>_ | _<...>_ | _<...>_ | |
| 0.1 (default) | _<...>_ | _<...>_ | _<...>_ | |
| 0.5 | _<...>_ | _<...>_ | _<...>_ | |

_Interpret: where's the sweet spot for your data? Why? Does it match the deck's §3.3 prediction?_

_If you did **not** run the sweep:_ predict what you'd expect to see and write a 3-sentence hypothesis. (No points lost — but the muscle of forming a hypothesis is the value.)

Tôi chưa chạy beta sweep, nên đây là giả thuyết cần được kiểm chứng chứ không phải kết quả. Với cùng dữ liệu và số bước, beta 0.05 có thể tạo cập nhật thận trọng hơn, reward gap tăng chậm nhưng response ít lệch khỏi SFT reference. Beta 0.1 là baseline hợp lý để xem tín hiệu ban đầu. Beta 0.5 có thể tách preference mạnh hơn nhưng cũng tăng nguy cơ reward hacking, response quá ngắn hoặc thay đổi phong cách. Tôi sẽ chỉ chọn “sweet spot” sau khi so sánh reward curve, eight-prompt evaluation và độ dài output của các run thực tế.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

Quyết định ảnh hưởng nhiều nhất trong lần thiết lập này là ưu tiên một môi trường local Windows có thể tái lập thay vì tiếp tục phụ thuộc vào phiên Colab dễ bị ngắt. Phương án thay thế là giữ nguyên requirements chung và cố chạy lại nhiều lần trên Colab hoặc môi trường Python cũ. Tôi không chọn phương án đó vì lỗi `WinError 193` cho thấy bộ Torch trong virtual environment ban đầu không tương thích với DLL đang được nạp, còn việc build `llama-cpp-python` không liên quan trực tiếp đến core DPO nhưng làm cài đặt bị chặn. Tôi đã tách đường cài: Torch CUDA trước, sau đó xFormers, Triton Windows, Unsloth và TRL phiên bản phù hợp. Smoke test sau đó xác nhận CUDA nhìn thấy RTX 3060 và import các thư viện chính thành công.

Kết quả quan trọng nhất ở thời điểm này không phải là một chỉ số DPO, mà là biết rõ ranh giới giữa “môi trường đã sẵn sàng” và “thí nghiệm đã hoàn thành”. NB1 thực sự đã bắt đầu dùng GPU, nhưng tôi dừng nó trước khi train xong, nên không được biến thời gian chạy dở thành loss, reward hay win-rate tưởng tượng. Nếu làm lại ngày mai, tôi sẽ dùng một smoke run nhỏ có chủ đích (ví dụ 100 SFT samples và 100 preference pairs) để đo thời gian mỗi step và VRAM, sau đó mới khởi chạy run 1.000/2.000 mẫu qua đêm. Cách này giúp ước lượng thời gian chính xác, phát hiện OOM sớm, và vẫn giữ báo cáo cuối cùng dựa hoàn toàn trên artifacts có thể kiểm tra.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | _<...>_ | _<...>_ | _<...>_ |
| GSM8K | _<...>_ | _<...>_ | _<...>_ |
| MMLU (sampled) | _<...>_ | _<...>_ | _<...>_ |
| AlpacaEval-lite | _<...>_ | _<...>_ | _<...>_ |

Chưa chạy benchmark tùy chọn. Không có `data/eval/benchmark_results.json`, do đó mọi delta trong bảng trên đều chưa xác định. Khi có kết quả, phần này cần phân biệt improvement về instruction following với năng lực giải bài toán. Một cải thiện ở IFEval hoặc AlpacaEval-lite có thể đồng thời đi cùng một regression nhỏ ở GSM8K; đó là alignment tax cần được báo cáo, không nên che đi. Vì NB6 là tùy chọn và core DPO chưa hoàn thành, tôi không gán điểm số tham khảo của Tulu 3 cho model 3B này.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu)_
