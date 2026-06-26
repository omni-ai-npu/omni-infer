// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{
    OffsetReferential, OffsetType, PreTokenizedString,
};
use tokenizers::{Model, PreTokenizer};

use rayon::prelude::*;

//Representation of each piece:
//- Range: A regular segment that needs to go through model.tokenize, where (s, e) represents its byte range in the normalized string after reconstruction.
//- PreIds: Preset tokens (e.g., AddedToken) have already been assigned an ID list during the pretok phase; (s, e) still records the byte range for byte budgeting.
#[derive(Clone)]
enum PieceEntry {
    Range { s: usize, e: usize },
    PreIds { s: usize, e: usize, ids: Vec<u32> },
}

struct TextPieces {
    norm: Arc<String>,       // Normalized string after reconstruction (concatenated in the order of pieces from splits)
    pieces: Vec<PieceEntry>, // Piece-by-piece description
}

#[pyclass]
pub struct InterleaveStats {
    #[pyo3(get)]
    pub texts: usize,
    #[pyo3(get)]
    pub total_pieces: usize,
    #[pyo3(get)]
    pub total_segments: usize,
    #[pyo3(get)]
    pub total_bytes: usize,
    #[pyo3(get)]
    pub us_norm_pretok: u64,
    #[pyo3(get)]
    pub us_build_segments: u64,
    #[pyo3(get)]
    pub us_tokenize_segments: u64,
    #[pyo3(get)]
    pub us_merge: u64,
    #[pyo3(get)]
    pub us_wall: u64,
    #[pyo3(get)]
    pub seg_us_p50: u64,
    #[pyo3(get)]
    pub seg_us_p95: u64,
    #[pyo3(get)]
    pub seg_us_mean: f64,
}

#[pyclass]
pub struct ChunkTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl ChunkTokenizer {
    #[new]
    pub fn new(tokenizer_json_path: &str) -> PyResult<Self> {
        let tok = Tokenizer::from_file(tokenizer_json_path)
            .map_err(|e| PyValueError::new_err(format!("load tokenizer failed: {e}")))?;
        Ok(Self { inner: tok })
    }

    /// Cross-text interleaved encoding (zero-copy + byte-budget segmentation; respecting the preset ids of AddedToken; no special tokens)
    #[pyo3(signature = (texts, bytes_per_segment = 32768, parallel = true))]
    pub fn encode_batch_interleaved_with_stats(
        &self,
        texts: Vec<String>,
        bytes_per_segment: usize,
        parallel: bool,
    ) -> PyResult<(Vec<Vec<u32>>, InterleaveStats)> {
        if bytes_per_segment == 0 {
            return Err(PyValueError::new_err("bytes_per_segment must be > 0"));
        }
        let wall_t0 = Instant::now();

        // 1) Cross-text preprocessing (normalize + pretok + reconstruct normalized strings + piece entries)
        let t0 = Instant::now();
        let per_text: Vec<TextPieces> = if parallel {
            {
                texts
                    .par_iter()
                    .map(|t| self.extract_piece_entries_zero_copy(t))
                    .collect::<Result<Vec<_>, _>>()?
            }
        } else {
            let mut v = Vec::with_capacity(texts.len());
            for t in texts.iter() {
                v.push(self.extract_piece_entries_zero_copy(t)?);
            }
            v
        };
        let mut total_pieces = 0usize;
        let mut total_bytes = 0usize;
        for tp in per_text.iter() {
            total_pieces += tp.pieces.len();
            total_bytes += tp.norm.len();
        }
        let us_norm_pretok = t0.elapsed().as_micros() as u64;
        let data = Arc::new(per_text);

        // 2) Segment by byte budget (using the byte length of each piece as the cost)
        let t1 = Instant::now();
        #[derive(Clone, Copy)]
        struct SegIdx {
            text_idx: usize,
            seg_idx: usize,
            start: usize,
            end: usize, // pieces index [start, end)
        }
        let mut tasks: Vec<SegIdx> = Vec::new();
        for (ti, tp) in data.iter().enumerate() {
            if tp.pieces.is_empty() {
                continue;
            }
            let mut start = 0usize;
            let mut seg = 0usize;
            while start < tp.pieces.len() {
                let mut end = start;
                let mut budget = 0usize;
                while end < tp.pieces.len() {
                    let piece_len = match &tp.pieces[end] {
                        PieceEntry::Range { s, e } | PieceEntry::PreIds { s, e, .. } => e.saturating_sub(*s),
                    };
                    if end > start && budget + piece_len > bytes_per_segment {
                        break;
                    }
                    budget += piece_len;
                    end += 1;
                }
                tasks.push(SegIdx {
                    text_idx: ti,
                    seg_idx: seg,
                    start,
                    end,
                });
                start = end;
                seg += 1;
            }
        }
        let us_build_segments = t1.elapsed().as_micros() as u64;
        let total_segments = tasks.len();

        // 3) Execute segment task: Append directly for PreIds; for Range, use model.tokenize(&norm[s..e])
        #[derive(Clone)]
        struct SegmentResult {
            text_idx: usize,
            seg_idx: usize,
            ids: Vec<u32>,
            seg_us: u64,
        }
        let model = self.inner.get_model();

        let run_task = |seg: SegIdx| -> SegmentResult {
            let tp = &data[seg.text_idx];
            let seg_t0 = Instant::now();
            let mut ids: Vec<u32> = Vec::new();
            for i in seg.start..seg.end {
                match &tp.pieces[i] {
                    PieceEntry::PreIds { ids: pre, .. } => {
                        ids.extend_from_slice(pre);
                    }
                    PieceEntry::Range { s, e } => {
                        if *s >= *e || *e > tp.norm.len() {
                            continue;
                        }
                        if let Some(piece) = tp.norm.get(*s..*e) {
                            if let Ok(toks) = model.tokenize(piece) {
                                ids.reserve(ids.len() + toks.len());
                                for tk in toks {
                                    ids.push(tk.id);
                                }
                            }
                        }
                    }
                }
            }
            let seg_us = seg_t0.elapsed().as_micros() as u64;
            SegmentResult {
                text_idx: seg.text_idx,
                seg_idx: seg.seg_idx,
                ids,
                seg_us,
            }
        };

        let t2 = Instant::now();
        let results: Vec<SegmentResult> = if parallel {
            {
                tasks.into_par_iter().map(run_task).collect()
            }
        } else {
            tasks.into_iter().map(run_task).collect()
        };
        let us_tokenize_segments = t2.elapsed().as_micros() as u64;

        // time disribution
        let mut seg_times: Vec<u64> = results.iter().map(|r| r.seg_us).collect();
        seg_times.sort_unstable();
        let seg_us_p50 = if seg_times.is_empty() { 0 } else { seg_times[seg_times.len() / 2] };
        let seg_us_p95 = if seg_times.is_empty() {
            0
        } else {
            seg_times[(seg_times.len() as f64 * 0.95) as usize].min(*seg_times.last().unwrap())
        };
        let seg_us_mean = if seg_times.is_empty() {
            0.0
        } else {
            (seg_times.iter().sum::<u64>() as f64) / (seg_times.len() as f64)
        };

        // 4) merge each text
        let t3 = Instant::now();
        let mut buckets: Vec<Vec<(usize, Vec<u32>)>> = vec![Vec::new(); texts.len()];
        for r in results.into_iter() {
            buckets[r.text_idx].push((r.seg_idx, r.ids));
        }
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(texts.len());
        for mut segs in buckets.into_iter() {
            if segs.is_empty() {
                out.push(Vec::new());
                continue;
            }
            segs.sort_by_key(|(k, _)| *k);
            let est: usize = segs.iter().map(|(_, v)| v.len()).sum();
            let mut ids: Vec<u32> = Vec::with_capacity(est);
            for (_, v) in segs.into_iter() {
                ids.extend_from_slice(&v);
            }
            out.push(ids);
        }
        let us_merge = t3.elapsed().as_micros() as u64;

        let stats = InterleaveStats {
            texts: texts.len(),
            total_pieces,
            total_segments,
            total_bytes,
            us_norm_pretok,
            us_build_segments,
            us_tokenize_segments,
            us_merge,
            us_wall: wall_t0.elapsed().as_micros() as u64,
            seg_us_p50,
            seg_us_p95,
            seg_us_mean,
        };
        Ok((out, stats))
    }
}

impl ChunkTokenizer {
    /// Extract "normalized string reconstructed by splits + per-piece entries (including preset IDs of AddedToken)"
    fn extract_piece_entries_zero_copy(&self, text: &str) -> PyResult<TextPieces> {
        // 1) Use AddedVocabulary to extract (it will place the identified Added/Special tokens into maybe_tokens)
        let added = self.inner.get_added_vocabulary();
        let normalizer = self.inner.get_normalizer();
        let mut pts: PreTokenizedString = added.extract_and_normalize(normalizer, text);

        // 2) Execute the pre_tokenizer again; splits that already have maybe_tokens will not be processed again.
        if let Some(pretok) = self.inner.get_pre_tokenizer() {
            pretok.pre_tokenize(&mut pts)
                .map_err(|e| PyValueError::new_err(format!("pre_tokenize failed: {e}")))?;
        }

        // 3)Reconstructing normalized strings and piece entries from splits (Normalized + Byte)
        let splits = pts.get_splits(OffsetReferential::Normalized, OffsetType::Byte);

        // estimate
        let mut total_len = 0usize;
        for (piece, _off, _maybe_tokens) in splits.iter() {
            total_len += piece.len();
        }
        let mut norm = String::with_capacity(total_len);
        let mut pieces: Vec<PieceEntry> = Vec::with_capacity(splits.len());
        let mut cursor = 0usize;

        for (piece, _off, maybe_tokens) in splits.into_iter() {
            if piece.is_empty() {
                continue;
            }
            let s = cursor;
            norm.push_str(piece);
            cursor += piece.len();
            let e = cursor;

            if let Some(tokens) = maybe_tokens {
                // preset ids:record all t.id
                let mut ids: Vec<u32> = Vec::with_capacity(tokens.len());
                for t in tokens {
                    ids.push(t.id);
                }
                pieces.push(PieceEntry::PreIds { s, e, ids });
            } else {
                pieces.push(PieceEntry::Range { s, e });
            }
        }

        Ok(TextPieces {
            norm: std::sync::Arc::new(norm),
            pieces,
        })
    }
}

#[pymodule]
fn tokenizers_chunk_ext<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ChunkTokenizer>()?;
    m.add_class::<InterleaveStats>()?;
    Ok(())
}