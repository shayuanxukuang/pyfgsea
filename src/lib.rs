use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use numpy::PyReadonlyArray1;
use std::collections::HashMap;

// -------------------------------------------------------------------------
// 1. Data Structures
// -------------------------------------------------------------------------

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultilevelDebugInfo {
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub accept_rates: Vec<f64>,
    #[pyo3(get)]
    pub current_level: usize,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GseaResult {
    #[pyo3(get)]
    pub es: f64,
    #[pyo3(get)]
    pub pval: f64,
    #[pyo3(get)]
    pub log2err: f64,
    #[pyo3(get)]
    pub debug_info: Option<MultilevelDebugInfo>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct TailCurve {
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub log_probs: Vec<f64>,
    #[pyo3(get)]
    pub populations: Vec<Vec<f64>>, 
    #[pyo3(get)]
    pub sample_size: usize,
}

// -------------------------------------------------------------------------
// 2. Core Math / ES Calculation
// -------------------------------------------------------------------------

#[inline]
fn calculate_es_inner(scores: &[f64], hits: &[usize], gsea_param: f64) -> f64 {
    let n_genes = scores.len();
    let n_hits = hits.len();
    if n_hits == 0 || n_hits == n_genes { return 0.0; }

    let nr: f64 = if (gsea_param - 1.0).abs() < 1e-6 {
        hits.iter().map(|&i| scores[i].abs()).sum()
    } else if gsea_param == 0.0 {
        n_hits as f64
    } else {
        hits.iter().map(|&i| scores[i].abs().powf(gsea_param)).sum()
    };

    if nr == 0.0 { return 0.0; }
    let inv_nr = 1.0 / nr;
    let inv_miss = 1.0 / ((n_genes - n_hits) as f64);
    let mut max_deviation: f64 = 0.0;
    let mut current_sum = 0.0;
    let mut last_hit_idx: isize = -1;

    for &hit_idx_usize in hits {
        let hit_idx = hit_idx_usize as isize;
        let num_miss = (hit_idx - last_hit_idx - 1) as f64;
        current_sum -= num_miss * inv_miss;
        if current_sum.abs() > max_deviation.abs() { max_deviation = current_sum; }

        let contribution = if (gsea_param - 1.0).abs() < 1e-6 {
            scores[hit_idx_usize].abs()
        } else if gsea_param == 0.0 { 1.0 } else {
            scores[hit_idx_usize].abs().powf(gsea_param)
        };
        current_sum += contribution * inv_nr;
        if current_sum.abs() > max_deviation.abs() { max_deviation = current_sum; }
        last_hit_idx = hit_idx;
    }
    max_deviation
}

#[pyfunction]
fn calculate_es(scores: PyReadonlyArray1<f64>, hits: Vec<usize>, gsea_param: f64) -> PyResult<f64> {
    let scores_slice = scores.as_slice()?;
    let es = calculate_es_inner(scores_slice, &hits, gsea_param);
    Ok(es)
}

// -------------------------------------------------------------------------
// 3. Batched / Curve Building Logic
// -------------------------------------------------------------------------

fn build_tail_curve_inner(
    scores_slice: &[f64],
    size: usize,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    sign: i32,
    max_metric: Option<f64>
) -> TailCurve {
    let n_genes = scores_slice.len();
    let use_abs = match score_type { Some("one_sided_signed") => false, _ => true };
    let get_metric = |es: f64| -> f64 {
        if use_abs { es.abs() } else if sign >= 0 { es } else { -es }
    };
    let target_metric = max_metric.unwrap_or(f64::INFINITY);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    
    let mut thresholds: Vec<f64> = Vec::new();
    let mut log_probs: Vec<f64> = Vec::new();
    let mut populations: Vec<Vec<f64>> = Vec::new();
    
    let mut current_hits_pop: Vec<Vec<usize>> = Vec::with_capacity(sample_size);
    let mut current_metrics: Vec<f64> = Vec::with_capacity(sample_size);
    
    for _ in 0..sample_size {
        let idx_vec = rand::seq::index::sample(&mut rng, n_genes, size);
        let mut rand_hits = idx_vec.into_vec();
        rand_hits.sort_unstable();
        let es = calculate_es_inner(scores_slice, &rand_hits, gsea_param);
        let metric = get_metric(es);
        current_hits_pop.push(rand_hits);
        current_metrics.push(metric);
    }
    populations.push(current_metrics.clone());
    log_probs.push(0.0);
    
    let mut current_level_count = 0;
    let mut current_log_prob: f64 = 0.0;
    let walk_steps = size.clamp(10, 100);
    
    loop {
        if current_log_prob.exp() < eps { break; }
        if current_level_count > 2000 { break; }
        
        let mut indices: Vec<usize> = (0..sample_size).collect();
        let cutoff_idx = sample_size / 2;
        indices.select_nth_unstable_by(cutoff_idx, |&i, &j| {
            current_metrics[j].total_cmp(&current_metrics[i])
        });
        
        let threshold_metric = current_metrics[indices[cutoff_idx]];
        if threshold_metric < 1e-9 { break; }
        if threshold_metric >= target_metric { break; } 
        
        let survivors: Vec<usize> = indices.iter().filter(|&&i| current_metrics[i] >= threshold_metric).cloned().collect();
        let count_pass = survivors.len();
        thresholds.push(threshold_metric);
        let p_i = count_pass as f64 / sample_size as f64;
        current_log_prob += p_i.ln();
        log_probs.push(current_log_prob);
        
        let elite_indices: Vec<usize>;
        if count_pass > cutoff_idx {
             let sample_indices = rand::seq::index::sample(&mut rng, count_pass, cutoff_idx);
             elite_indices = sample_indices.into_iter().map(|i| survivors[i]).collect();
        } else {
             elite_indices = survivors;
        }
        
        let mut next_hits_pop = Vec::with_capacity(sample_size);
        let mut next_metrics = Vec::with_capacity(sample_size);
        for &idx in &elite_indices {
            next_hits_pop.push(current_hits_pop[idx].clone());
            next_metrics.push(current_metrics[idx]);
        }
        for _ in 0..(sample_size - elite_indices.len()) {
            let parent_idx = elite_indices[rng.gen_range(0..elite_indices.len())];
            let mut current_hits = current_hits_pop[parent_idx].clone();
            let mut current_metric_val = current_metrics[parent_idx];
            for _ in 0..walk_steps {
                let rem_inner_idx = rng.gen_range(0..current_hits.len());
                let rem_gene = current_hits[rem_inner_idx];
                let mut gene_add = rng.gen_range(0..n_genes);
                let mut coll_attempts = 0;
                while current_hits.binary_search(&gene_add).is_ok() && coll_attempts < 10 {
                    gene_add = rng.gen_range(0..n_genes);
                    coll_attempts += 1;
                }
                if coll_attempts < 10 {
                    current_hits.remove(rem_inner_idx);
                    let insert_pos = current_hits.binary_search(&gene_add).unwrap_or_else(|e| e);
                    current_hits.insert(insert_pos, gene_add);
                    let cand_es = calculate_es_inner(scores_slice, &current_hits, gsea_param);
                    let cand_metric = get_metric(cand_es);
                    if cand_metric >= threshold_metric {
                        current_metric_val = cand_metric;
                    } else {
                        current_hits.remove(insert_pos);
                        let revert_pos = current_hits.binary_search(&rem_gene).unwrap_or_else(|e| e);
                        current_hits.insert(revert_pos, rem_gene);
                    }
                }
            }
            next_hits_pop.push(current_hits);
            next_metrics.push(current_metric_val);
        }
        current_hits_pop = next_hits_pop;
        current_metrics = next_metrics;
        populations.push(current_metrics.clone());
        current_level_count += 1;
    }
    TailCurve { thresholds, log_probs, populations, sample_size }
}

#[pyfunction]
fn query_tail_curve(curve: &TailCurve, obs_es: f64, score_type: Option<&str>, sign: Option<i32>) -> (f64, f64) {
    let use_abs = match score_type { Some("one_sided_signed") => false, _ => true };
    let metric = if use_abs { obs_es.abs() } else {
        let s = sign.unwrap_or(1);
        if s >= 0 { obs_es } else { -obs_es }
    };
    if metric < 1e-10 { return (1.0, 0.0); }
    
    let mut level_idx = 0;
    for (i, &t) in curve.thresholds.iter().enumerate() {
        if metric >= t { level_idx = i + 1; } else { break; }
    }
    if level_idx >= curve.populations.len() { level_idx = curve.populations.len() - 1; }
    
    let population = &curve.populations[level_idx];
    let log_prob_level = if level_idx < curve.log_probs.len() { curve.log_probs[level_idx] } else { curve.log_probs.last().copied().unwrap_or(0.0) };
    
    let count_better = population.iter().filter(|&&m| m >= metric).count();
    let tail_prob_raw = count_better as f64 / population.len() as f64;
    let pval = log_prob_level.exp() * tail_prob_raw;
    let log2err = 1.0 / (curve.sample_size as f64).sqrt();
    (pval, log2err)
}

#[pyfunction]
#[pyo3(signature = (scores, size, sample_size, seed, gsea_param, eps, score_type=None, sign=1))]
fn build_tail_curve(
    scores: PyReadonlyArray1<f64>,
    size: usize,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    sign: i32
) -> PyResult<TailCurve> {
    let scores_slice = scores.as_slice()?;
    Ok(build_tail_curve_inner(scores_slice, size, sample_size, seed, gsea_param, eps, score_type, sign, None))
}

// -------------------------------------------------------------------------
// 4. Optimized Runner (The "Trajectory" Engine)
// -------------------------------------------------------------------------

#[pyclass]
struct GseaPrerankedRunner {
    pathways: Vec<Vec<usize>>, // Raw indices (0..N_genes)
    min_size: usize,
    max_size: usize,
}

#[pymethods]
impl GseaPrerankedRunner {
    #[new]
    fn new(pathways: Vec<Vec<usize>>, min_size: usize, max_size: usize) -> Self {
        GseaPrerankedRunner { pathways, min_size, max_size }
    }

    #[pyo3(signature = (scores, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None, precheck_n=None, precheck_eps=None))]
    fn run(
        &self,
        scores: PyReadonlyArray1<f64>,
        sample_size: usize,
        seed: u64,
        gsea_param: f64,
        eps: f64,
        score_type: Option<&str>,
        bin_width: Option<usize>,
        precheck_n: Option<usize>,
        precheck_eps: Option<f64>
    ) -> PyResult<Vec<GseaResult>> {
        let scores_slice = scores.as_slice()?;
        if scores_slice.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("scores must be non-empty"));
        }
        let n_genes = scores_slice.len();

        // 1. Rank scores O(N log N)
        // We do this PER CALL because in rolling window, scores change, but gene list (pathways) is constant.
        let mut order: Vec<usize> = (0..n_genes).collect();
        order.sort_unstable_by(|&i, &j| scores_slice[j].total_cmp(&scores_slice[i]));

        let mut ranked_scores: Vec<f64> = Vec::with_capacity(n_genes);
        let mut rank_pos: Vec<usize> = vec![0; n_genes];
        for (rank, &idx) in order.iter().enumerate() {
            ranked_scores.push(scores_slice[idx]);
            rank_pos[idx] = rank;
        }

        // 2. Compute ES and Size for all pathways (No intermediate allocation!)
        // Zero-Allocation: We map raw indices to ranks ON THE FLY inside the parallel iterator.
        
        let path_stats: Vec<(usize, f64)> = self.pathways.par_iter().map(|p| {
            // Map raw indices to ranks
            let mut hits: Vec<usize> = p.iter()
                .map(|&g| if g < n_genes { rank_pos[g] } else { usize::MAX })
                .filter(|&r| r != usize::MAX)
                .collect();
            
            // Sort hits (required for ES calc)
            hits.sort_unstable();
            
            // Calc ES directly
            let es = calculate_es_inner(&ranked_scores, &hits, gsea_param);
            (hits.len(), es)
        }).collect();

        // 3. Group by Size (Binning)
        let mut size_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        let width = bin_width.unwrap_or(0);

        for (i, &(size, _es)) in path_stats.iter().enumerate() {
             if size == 0 { continue; } 
             // Binning: Round to nearest
             let key = if width > 0 {
                let k = ((size + width / 2) / width) * width;
                if k == 0 { width } else { k }
             } else {
                size
             };
             size_groups.entry(key).or_default().push(i);
        }
        let groups: Vec<(usize, Vec<usize>)> = size_groups.into_iter().collect();

        // 4. Build Curves (Batched)
        let use_abs = match score_type { Some("one_sided_signed") => false, _ => true };
        let pc_n = precheck_n.unwrap_or(64);
        let pc_eps = precheck_eps.unwrap_or(0.005);

        let curves_map: HashMap<usize, (Option<TailCurve>, Option<TailCurve>)> = groups.par_iter().map(|(key, indices)| {
            let batch_seed = seed + *key as u64;
            let sim_size = *key;

            let mut max_pos_metric = 0.0;
            let mut max_neg_metric = 0.0;
            let mut has_pos = false;
            let mut has_neg = false;

            for &idx in indices {
                let (_, es) = path_stats[idx];
                if es >= 0.0 {
                    has_pos = true;
                    if es > max_pos_metric { max_pos_metric = es; }
                } else {
                    has_neg = true;
                    let abs_es = es.abs();
                    if abs_es > max_neg_metric { max_neg_metric = abs_es; }
                }
            }

            let mut max_abs_metric = 0.0;
            if use_abs {
                max_abs_metric = max_pos_metric.max(max_neg_metric);
            }

            // Curve Builder Helper
            let build_checked = |sign: i32, target: f64| -> Option<TailCurve> {
                let pc_curve = build_tail_curve_inner(&ranked_scores, sim_size, pc_n, batch_seed + 99999, gsea_param, 1.0, score_type, sign, None);
                let (pc_pval, _) = query_tail_curve(&pc_curve, target, score_type, Some(sign));

                if pc_pval > pc_eps {
                    Some(pc_curve)
                } else {
                    Some(build_tail_curve_inner(&ranked_scores, sim_size, sample_size, batch_seed, gsea_param, eps, score_type, sign, Some(target)))
                }
            };

            let curve_pos = if use_abs {
                build_checked(1, max_abs_metric)
            } else if max_pos_metric > 0.0 || has_pos {
                build_checked(1, max_pos_metric)
            } else {
                None
            };

            let curve_neg = if !use_abs && (max_neg_metric > 0.0 || has_neg) {
                build_checked(-1, max_neg_metric)
            } else {
                None
            };

            (*key, (curve_pos, curve_neg))
        }).collect();

        // 5. Query Results
        let final_results: Vec<GseaResult> = (0..self.pathways.len()).into_par_iter().map(|i| {
            let (size, es) = path_stats[i];
            
            if size == 0 {
                return GseaResult { es: 0.0, pval: 1.0, log2err: 0.0, debug_info: None };
            }

            let key = if width > 0 {
                let k = ((size + width / 2) / width) * width;
                if k == 0 { width } else { k }
            } else {
                size
            };

            let (curve_pos, curve_neg) = match curves_map.get(&key) {
                Some(v) => v,
                None => return GseaResult { es, pval: 1.0, log2err: 0.0, debug_info: None },
            };

            let (curve, sign) = if use_abs {
                (curve_pos.as_ref(), 1)
            } else if es >= 0.0 {
                (curve_pos.as_ref(), 1)
            } else {
                (curve_neg.as_ref(), -1)
            };

            let (pval, log2err) = if let Some(c) = curve {
                query_tail_curve(c, es, score_type, Some(sign))
            } else {
                (1.0, 0.0)
            };

            GseaResult {
                es,
                pval,
                log2err,
                debug_info: None
            }
        }).collect();

        Ok(final_results)
    }
}

// -------------------------------------------------------------------------
// 5. Wrappers & Utilities
// -------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None))]
fn fgsea_multilevel_batched(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    bin_width: Option<usize>
) -> PyResult<Vec<GseaResult>> {
    // Just wrap in Runner
    let min_size = 0;
    let max_size = usize::MAX;
    let runner = GseaPrerankedRunner::new(pathways, min_size, max_size);
    runner.run(scores, sample_size, seed, gsea_param, eps, score_type, bin_width, None, None)
}

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None))]
fn fgsea_multilevel_batched_scores(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    bin_width: Option<usize>
) -> PyResult<Vec<GseaResult>> {
    // Alias for the same logic
    let min_size = 0;
    let max_size = usize::MAX;
    let runner = GseaPrerankedRunner::new(pathways, min_size, max_size);
    runner.run(scores, sample_size, seed, gsea_param, eps, score_type, bin_width, None, None)
}

// --- Standard Multilevel Logic (For Benchmark Baseline) ---

fn calculate_multilevel_pval(
    scores: &[f64],
    raw_hits: &[usize], 
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>
) -> GseaResult { 
    let n_genes = scores.len();
    let use_abs = match score_type { Some("one_sided_signed") => false, _ => true };

    let mut hits: Vec<usize> = raw_hits.iter().filter(|&&x| x < n_genes).cloned().collect();
    if hits.is_empty() {
        return GseaResult { es: 0.0, pval: 1.0, log2err: 0.0, debug_info: None };
    }
    hits.sort_unstable();
    hits.dedup();

    let obs_es = calculate_es_inner(scores, &hits, gsea_param);
    
    let get_metric = |es: f64| -> f64 {
        if use_abs { es.abs() } else if obs_es >= 0.0 { es } else { -es }
    };
    
    let obs_metric = get_metric(obs_es);
    if obs_metric < 1e-10 { return GseaResult { es: obs_es, pval: 1.0, log2err: 0.0, debug_info: None }; }
    
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut population: Vec<Vec<usize>> = Vec::with_capacity(sample_size);
    let mut pop_metric: Vec<f64> = Vec::with_capacity(sample_size);
    let mut better_count_l0 = 0;

    for _ in 0..sample_size {
        let idx_vec = rand::seq::index::sample(&mut rng, n_genes, hits.len());
        let mut rand_hits = idx_vec.into_vec();
        rand_hits.sort_unstable(); 
        let es = calculate_es_inner(scores, &rand_hits, gsea_param);
        let metric = get_metric(es);
        if metric >= obs_metric { better_count_l0 += 1; }
        population.push(rand_hits);
        pop_metric.push(metric);
    }
    
    let threshold_count = (sample_size as f64 * 0.05).ceil() as usize; 
    if better_count_l0 >= threshold_count {
        let pval = (better_count_l0 as f64 + 1.0) / (sample_size as f64 + 1.0);
        let safe_pval = pval.max(eps);
        let log2err = 1.0 / (sample_size as f64).sqrt(); 
        return GseaResult { es: obs_es, pval: safe_pval, log2err, debug_info: None };
    }

    let mut log_prob: f64 = 0.0;
    let mut current_level_count = 0;
    let walk_steps = hits.len().clamp(10, 100);
    let mut debug_thresholds: Vec<f64> = Vec::new();
    let mut debug_accept_rates: Vec<f64> = Vec::new();

    loop {
        let mut indices: Vec<usize> = (0..sample_size).collect();
        let cutoff_idx = sample_size / 2;
        indices.select_nth_unstable_by(cutoff_idx, |&i, &j| {
            pop_metric[j].total_cmp(&pop_metric[i])
        });
        
        let better_count = pop_metric.iter().filter(|&&m| m >= obs_metric).count();
        let threshold_metric = pop_metric[indices[cutoff_idx]];
        debug_thresholds.push(threshold_metric);
        
        if threshold_metric >= obs_metric || better_count == sample_size {
            let tail_prob = better_count as f64 / sample_size as f64;
            let pval = log_prob.exp() * tail_prob;
            let safe_pval = pval.max(f64::MIN_POSITIVE).max(eps);
            let log2err = 1.0 / ((sample_size / 2) as f64).sqrt(); 
            let debug_info = MultilevelDebugInfo {
                thresholds: debug_thresholds,
                accept_rates: debug_accept_rates,
                current_level: current_level_count,
            };
            return GseaResult { es: obs_es, pval: safe_pval, log2err, debug_info: Some(debug_info) };
        }
        
        log_prob += (cutoff_idx as f64 / sample_size as f64).ln();
        let elite_indices = &indices[0..cutoff_idx];
        let mut total_swaps = 0;
        let mut accepted_swaps = 0;

        for k in cutoff_idx..sample_size {
            let target_idx = indices[k];
            let parent_idx = elite_indices[rng.gen_range(0..elite_indices.len())];
            let mut current_hits = population[parent_idx].clone();
            let mut current_metric_val = pop_metric[parent_idx]; 

            for _ in 0..walk_steps {
                let rem_inner_idx = rng.gen_range(0..current_hits.len());
                let rem_gene = current_hits[rem_inner_idx];
                let mut gene_add = rng.gen_range(0..n_genes);
                let mut coll_attempts = 0;
                while current_hits.binary_search(&gene_add).is_ok() && coll_attempts < 10 { 
                    gene_add = rng.gen_range(0..n_genes); 
                    coll_attempts += 1;
                }
                
                if coll_attempts < 10 {
                    total_swaps += 1;
                    current_hits.remove(rem_inner_idx); 
                    let insert_pos = current_hits.binary_search(&gene_add).unwrap_or_else(|e| e);
                    current_hits.insert(insert_pos, gene_add);

                    let cand_es = calculate_es_inner(scores, &current_hits, gsea_param);
                    let cand_metric = get_metric(cand_es);
                    
                    if cand_metric >= threshold_metric {
                        accepted_swaps += 1;
                        current_metric_val = cand_metric;
                    } else {
                        current_hits.remove(insert_pos);
                        let revert_pos = current_hits.binary_search(&rem_gene).unwrap_or_else(|e| e);
                        current_hits.insert(revert_pos, rem_gene);
                    }
                }
            }
            population[target_idx] = current_hits;
            pop_metric[target_idx] = current_metric_val;
        }
        let rate = if total_swaps > 0 { accepted_swaps as f64 / total_swaps as f64 } else { 0.0 };
        debug_accept_rates.push(rate);
        current_level_count += 1;
        if current_level_count > 2000 { break; }
    }
    
    let safe_pval = eps.max(f64::MIN_POSITIVE);
    GseaResult { es: obs_es, pval: safe_pval, log2err: 0.0, debug_info: None }
}

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None))]
fn fgsea_multilevel(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>
) -> PyResult<Vec<GseaResult>> { 
    // FIXED: Now calls the actual logic instead of returning dummy results
    let scores_slice = scores.as_slice()?;
    if scores_slice.is_empty() { return Err(pyo3::exceptions::PyValueError::new_err("empty scores")); }

    let results: Vec<GseaResult> = pathways.par_iter().enumerate()
        .map(|(i, hits)| {
            calculate_multilevel_pval(scores_slice, hits, sample_size, seed + i as u64, gsea_param, eps, score_type)
        })
        .collect();
    Ok(results)
}

#[pyfunction]
fn get_random_es_means(
    scores: PyReadonlyArray1<f64>,
    sizes: Vec<usize>,
    nperm: usize,
    seed: u64,
    gsea_param: f64
) -> PyResult<Vec<(f64, f64)>> {
    let scores_slice = scores.as_slice()?;
    let n_genes = scores_slice.len();
    
    let results: Vec<(f64, f64)> = sizes.par_iter().enumerate().map(|(i, &size)| {
        let mut rng = ChaCha20Rng::seed_from_u64(seed + i as u64);
        let mut sum_pos = 0.0;
        let mut count_pos = 0.0;
        let mut sum_neg = 0.0;
        let mut count_neg = 0.0;

        for _ in 0..nperm {
            let idx_vec = rand::seq::index::sample(&mut rng, n_genes, size);
            let mut rand_hits = idx_vec.into_vec();
            rand_hits.sort_unstable(); 
            let es = calculate_es_inner(scores_slice, &rand_hits, gsea_param);
            if es >= 0.0 {
                sum_pos += es; count_pos += 1.0;
            } else {
                sum_neg += es; count_neg += 1.0;
            }
        }
        let pos_mean = if count_pos > 0.0 { sum_pos / count_pos } else { 1.0 }; 
        let neg_mean = if count_neg > 0.0 { sum_neg / count_neg } else { -1.0 };
        (pos_mean, neg_mean)
    }).collect();
    Ok(results)
}

#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fgsea_multilevel, m)?)?;
    m.add_function(wrap_pyfunction!(fgsea_multilevel_batched, m)?)?; // legacy name
    m.add_function(wrap_pyfunction!(fgsea_multilevel_batched_scores, m)?)?;
    m.add_function(wrap_pyfunction!(get_random_es_means, m)?)?;
    m.add_function(wrap_pyfunction!(build_tail_curve, m)?)?;
    m.add_function(wrap_pyfunction!(query_tail_curve, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_es, m)?)?;
    m.add_class::<TailCurve>()?;
    m.add_class::<GseaPrerankedRunner>()?;
    Ok(())
}