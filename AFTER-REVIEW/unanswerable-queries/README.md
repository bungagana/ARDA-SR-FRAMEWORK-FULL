# Unanswerable Query Robustness Test

Ditambahkan pasca-review untuk menjawab kritik reviewer:

1. **Circularity** — 1.000 QA di benchmark utama digenerate dari corpus yang sama
   dan **100% answerable-by-construction** (`should_be_answerable=True` untuk
   semua 1.000 item — lihat `../../data/qa_dataset.json` dan `../../verify_answerability.py`).
   Tidak ada satupun contoh unanswerable (`r_i=1`) di test set utama, padahal
   definisi FRR (Eq. 24 di paper) mengasumsikan kedua kelas ada.
2. **Tidak ada real/out-of-corpus query** untuk menguji sisi lain dari false
   refusal — yaitu apakah sistem *benar* menolak ketika memang tidak ada bukti.

Folder ini berisi 60 pertanyaan **unanswerable** yang dibuat manual (bukan
LLM-generated dari corpus, jadi tidak sirkular), diverifikasi satu per satu
terhadap metadata KB asli (`kb/chunks.json`: daftar province/year/commodity
yang benar-benar tercakup), dibagi 4 subkategori @15 soal:

| unans_type | Deskripsi |
|---|---|
| `province_not_in_corpus` | Provinsi/wilayah yang tidak ada di 82 dokumen (mis. Bali, Jawa Barat, Papua Barat Daya) |
| `regulation_not_covered` | Nomor/tahun regulasi yang tidak ada di corpus (corpus hanya punya UU 29/2009, PP 19/2024, Permentrans 1/2024, 1/2025, 8/2025, Permenperin 10/2025, SNI 8926, SNI 6232:2025) |
| `commodity_not_covered` | Komoditas di luar 8 yang ada di corpus (padi, jagung, karet, udang, sapi, kopi, kakao, kelapa sawit) |
| `out_of_domain_mixing` | Topik di luar domain transmigrasi yang dibungkus terlihat relevan (pertambangan, KUR, BPJS, pajak, dst.) |

Setiap item punya field `reason` yang menjelaskan persis kenapa item itu
seharusnya tidak terjawabkan — berguna untuk verifikasi manual/annotator
sebelum dipakai di manuskrip.

## Cara menjalankan

Dari folder ini (bukan dari repo root):

```bash
# Sanity check dulu (6 query saja, ~1-2 menit)
python run_unanswerable_test.py --smoke

# Default: 3 metode representatif (standard_rag, selfrag, arda_sr)
python run_unanswerable_test.py

# Pilih metode sendiri
python run_unanswerable_test.py --methods llm_only,crag,react,arda_sr

# Semua 10 baseline + ARDA-SR (paling lengkap, tapi paling lama & mahal API-nya)
python run_unanswerable_test.py --all
```

Script ini **memakai ulang** `kb/`, `config.py`, `baselines/`, `arda_sr/` dari
repo root — tidak build ulang apa pun. Butuh environment yang sama seperti
`03_run_experiment.py` (`GEMINI_API_KEY` di `.env` repo root minimal; kalau
mau nambah LLM-judge sendiri di luar script ini, siapkan juga `OPENAI_API_KEY`).

Ada checkpoint/resume per metode (`results/{method}_unanswerable_ckpt.json`)
sama seperti `03_run_experiment.py` — aman kalau run keputus di tengah jalan,
tinggal jalankan command yang sama lagi.

## Output

- `results/{method}_unanswerable_results.json` — hasil mentah per query (jawaban, evidence, is_refusal, latency, dst — format sama seperti `results/{method}_results.json` di eksperimen utama)
- `results/summary.csv` / `summary.json` — **Appropriate Refusal Rate (ARR)** per metode, overall dan per `unans_type` (FAR = 1-ARR juga disertakan)
- `results/false_acceptances.csv` — daftar semua kasus sistem **tidak** menolak (justru menjawab) padahal seharusnya tidak ada bukti — ini kandidat contoh kualitatif buat manuskrip (mirip Fig. 4 di paper), termasuk jawabannya supaya bisa dicek apakah itu hallucination.

## Metrik: ARR (Appropriate Refusal Rate)

Pasangan dari FRR yang sudah ada di paper (Eq. 24), sengaja dibingkai "↑ higher is better" biar searah dengan cara FRR biasa dibaca di tabel (walau FRR sendiri secara notasi ↓ lower is better):

```
FRR = #{r̂=1 ∧ r=0} / #{r=0}     ← paper: fraksi query answerable yang DITOLAK (over-refusal)          [↓ lower better]
ARR = #{r̂=1 ∧ r=1} / #{r=1}     ← script ini: fraksi query unanswerable yang BENAR ditolak             [↑ higher better]
FAR = 1 - ARR                    ← fraksi query unanswerable yang malah DIJAWAB (hallucination risk)    [↓ lower better]
```

`is_refusal` dideteksi pakai heuristik yang **sama persis** dengan pipeline
utama (`baselines/base.py::_is_refusal`, `arda_sr/dda.py::_is_refusal`) —
supaya definisi refusal konsisten dengan yang dipakai untuk FRR di Table 5/6/7.

## Untuk manuskrip

Setelah run selesai, `results/summary.csv` bisa langsung jadi tabel baru
di Section 3 (mis. "Table X: FAR on out-of-corpus unanswerable queries"),
dan `results/false_acceptances.csv` bisa jadi sumber contoh kualitatif baru
di Fig. 4 (kolom "hallucination on out-of-scope query") — melengkapi bukti
bahwa DDA/ARDA-SR tidak hanya rendah FRR (jarang menolak yang seharusnya
dijawab) tapi **juga** rendah FAR (jarang menjawab yang seharusnya ditolak),
dua sisi trade-off yang sama sekali belum diuji di benchmark 1.000 QA utama.
