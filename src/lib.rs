use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::index::sample;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use statrs::distribution::{Beta, ContinuousCDF};
use std::cmp::Ordering;
use std::collections::BTreeMap;

const ALGORITHM_REVISION: &str = "fgsea-1.38-pr178-v1";
const MAX_WEIGHT_SUM: f64 = (1_u64 << 30) as f64;

#[pyclass]
#[derive(Clone, Debug)]
pub struct MultilevelDebugInfo {
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub threshold_hashes: Vec<u64>,
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
    pub log_pval: f64,
    #[pyo3(get)]
    pub log2err: f64,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub termination_reason: String,
    #[pyo3(get)]
    pub n_levels: usize,
    #[pyo3(get)]
    pub acceptance_rate_min: f64,
    #[pyo3(get)]
    pub acceptance_rate_mean: f64,
    #[pyo3(get)]
    pub ranking_hash: String,
    #[pyo3(get)]
    pub null_curve_size: usize,
    #[pyo3(get)]
    pub approximate: bool,
    #[pyo3(get)]
    pub algorithm_revision: String,
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
    #[pyo3(get)]
    pub eps: f64,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub termination_reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RunMode {
    Aligned,
    Fast,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScoreType {
    Std,
    Pos,
    Neg,
    LegacyAbs,
    LegacySigned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Direction {
    Pos,
    Neg,
    Abs,
}

fn parse_mode(value: &str) -> Result<RunMode, String> {
    match value {
        "aligned" => Ok(RunMode::Aligned),
        "fast" => Ok(RunMode::Fast),
        _ => Err("mode must be 'aligned' or 'fast'".to_string()),
    }
}

fn parse_score_type(value: &str) -> Result<ScoreType, String> {
    match value {
        "std" => Ok(ScoreType::Std),
        "pos" => Ok(ScoreType::Pos),
        "neg" => Ok(ScoreType::Neg),
        "two_sided_abs" => Ok(ScoreType::LegacyAbs),
        "one_sided_signed" => Ok(ScoreType::LegacySigned),
        _ => Err(
            "score_type must be 'std', 'pos', 'neg', 'two_sided_abs', or 'one_sided_signed'"
                .to_string(),
        ),
    }
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline]
fn hash_combine(state: u64, value: u64) -> u64 {
    mix64(state ^ mix64(value))
}

fn derive_seed(
    master: u64,
    ranking_hash: u64,
    size: usize,
    direction: Direction,
    stage: u64,
) -> u64 {
    let direction_tag = match direction {
        Direction::Pos => 0x504f_53,
        Direction::Neg => 0x4e45_47,
        Direction::Abs => 0x4142_53,
    };
    let mut state = hash_combine(master, ranking_hash);
    state = hash_combine(state, size as u64);
    state = hash_combine(state, direction_tag);
    hash_combine(state, stage)
}

#[derive(Clone, Debug)]
struct PreparedRanking {
    weights: Vec<i64>,
    rank_pos: Vec<usize>,
    ranking_hash: u64,
}

impl PreparedRanking {
    fn new(scores: &[f64], gsea_param: f64) -> Result<Self, String> {
        if scores.is_empty() {
            return Err("scores must be non-empty".to_string());
        }
        if !gsea_param.is_finite() || gsea_param < 0.0 {
            return Err("gsea_param must be finite and >= 0".to_string());
        }
        if scores.iter().any(|value| !value.is_finite()) {
            return Err("scores contain NaN or infinite values".to_string());
        }

        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                .then_with(|| left.cmp(right))
        });

        let ranked_scores: Vec<f64> = order.iter().map(|index| scores[*index]).collect();
        let raw_weights: Vec<f64> = ranked_scores
            .iter()
            .map(|value| {
                if gsea_param == 0.0 {
                    1.0
                } else {
                    value.abs().powf(gsea_param)
                }
            })
            .collect();
        if raw_weights.iter().any(|weight| !weight.is_finite()) {
            return Err("abs(score)^gsea_param produced a non-finite weight".to_string());
        }
        let total: f64 = raw_weights.iter().sum();
        if !total.is_finite() {
            return Err("the sum of ranked weights is not finite".to_string());
        }
        let weights = if total == 0.0 {
            vec![1_i64; raw_weights.len()]
        } else {
            let raw_scale = MAX_WEIGHT_SUM / total;
            let scale = if raw_scale >= 1.0 {
                raw_scale.floor()
            } else {
                raw_scale
            };
            if !scale.is_finite() {
                return Err("integer weight scaling produced a non-finite factor".to_string());
            }
            let mut weights = Vec::with_capacity(raw_weights.len());
            for weight in &raw_weights {
                let scaled = weight * scale;
                if !scaled.is_finite() {
                    return Err("integer weight scaling produced a non-finite value".to_string());
                }
                // R's round(..., digits=0) uses IEC 60559 ties-to-even.
                weights.push(scaled.round_ties_even() as i64);
            }
            weights
        };

        let mut rank_pos = vec![0_usize; scores.len()];
        for (rank, original_index) in order.iter().enumerate() {
            rank_pos[*original_index] = rank;
        }

        let mut ranking_hash = mix64(scores.len() as u64);
        for (score, weight) in ranked_scores.iter().zip(weights.iter()) {
            ranking_hash = hash_combine(ranking_hash, score.to_bits());
            ranking_hash = hash_combine(ranking_hash, *weight as u64);
        }

        Ok(Self {
            weights,
            rank_pos,
            ranking_hash,
        })
    }

    fn hash_string(&self) -> String {
        format!("{:016x}", self.ranking_hash)
    }
}

#[derive(Clone, Copy, Debug)]
struct ExactScore {
    hit_total: i64,
    hit_sum: i64,
    miss_total: i64,
    miss_count: i64,
}

impl ExactScore {
    fn zero(hit_total: i64, miss_total: i64) -> Self {
        Self {
            hit_total,
            hit_sum: 0,
            miss_total,
            miss_count: 0,
        }
    }

    fn numerator(self) -> i128 {
        self.hit_sum as i128 * self.miss_total as i128
            - self.miss_count as i128 * self.hit_total as i128
    }

    fn denominator(self) -> i128 {
        self.hit_total as i128 * self.miss_total as i128
    }

    fn negate(self) -> Self {
        Self {
            hit_total: self.hit_total,
            hit_sum: -self.hit_sum,
            miss_total: self.miss_total,
            miss_count: -self.miss_count,
        }
    }

    fn as_f64(self) -> f64 {
        self.hit_sum as f64 / self.hit_total as f64
            - self.miss_count as f64 / self.miss_total as f64
    }
}

impl PartialEq for ExactScore {
    fn eq(&self, other: &Self) -> bool {
        self.numerator() * other.denominator() == other.numerator() * self.denominator()
    }
}

impl Eq for ExactScore {}

impl PartialOrd for ExactScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExactScore {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.numerator() * other.denominator()).cmp(&(other.numerator() * self.denominator()))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompoundScore {
    score: ExactScore,
    gene_set_hash: u64,
}

impl PartialOrd for CompoundScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompoundScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| self.gene_set_hash.cmp(&other.gene_set_hash))
    }
}

#[derive(Clone, Copy, Debug)]
struct ScoreBundle {
    std: ExactScore,
    pos: ExactScore,
    neg: ExactScore,
}

fn calculate_exact_scores(weights: &[i64], hits: &[usize]) -> Option<ScoreBundle> {
    let n_genes = weights.len();
    if hits.is_empty() || hits.len() >= n_genes {
        return None;
    }
    let miss_total = (n_genes - hits.len()) as i64;
    let raw_hit_total: i64 = hits.iter().map(|index| weights[*index]).sum();
    let use_uniform = raw_hit_total == 0;
    let hit_total = if use_uniform {
        hits.len() as i64
    } else {
        raw_hit_total
    };
    let mut current = ExactScore::zero(hit_total, miss_total);
    let mut best_pos = current;
    let mut best_neg = current;
    let mut last: isize = -1;

    for position in hits {
        current.miss_count += *position as i64 - last as i64 - 1;
        if current > best_pos {
            best_pos = current;
        }
        if current < best_neg {
            best_neg = current;
        }

        current.hit_sum += if use_uniform { 1 } else { weights[*position] };
        if current > best_pos {
            best_pos = current;
        }
        if current < best_neg {
            best_neg = current;
        }
        last = *position as isize;
    }

    let negative_magnitude = best_neg.negate();
    let best_std = match best_pos.cmp(&negative_magnitude) {
        Ordering::Greater => best_pos,
        Ordering::Less => best_neg,
        Ordering::Equal => ExactScore::zero(hit_total, miss_total),
    };
    Some(ScoreBundle {
        std: best_std,
        pos: best_pos,
        neg: best_neg,
    })
}

fn direction_for(score_type: ScoreType, observed: &ScoreBundle) -> Direction {
    match score_type {
        ScoreType::Pos => Direction::Pos,
        ScoreType::Neg => Direction::Neg,
        ScoreType::LegacyAbs => Direction::Abs,
        ScoreType::Std | ScoreType::LegacySigned => {
            if observed.std >= observed.std.negate() {
                Direction::Pos
            } else {
                Direction::Neg
            }
        }
    }
}

fn metric_for(score_type: ScoreType, bundle: &ScoreBundle, direction: Direction) -> ExactScore {
    match score_type {
        ScoreType::Pos => bundle.pos,
        ScoreType::Neg => bundle.neg.negate(),
        ScoreType::LegacyAbs => bundle.pos.max(bundle.neg.negate()),
        ScoreType::Std | ScoreType::LegacySigned => match direction {
            Direction::Pos => bundle.std,
            Direction::Neg => bundle.std.negate(),
            Direction::Abs => bundle.std,
        },
    }
}

fn observed_es(score_type: ScoreType, bundle: &ScoreBundle) -> f64 {
    match score_type {
        ScoreType::Pos => bundle.pos.as_f64(),
        ScoreType::Neg => bundle.neg.as_f64(),
        ScoreType::LegacyAbs => {
            if bundle.pos >= bundle.neg.negate() {
                bundle.pos.as_f64()
            } else {
                bundle.neg.as_f64()
            }
        }
        ScoreType::Std | ScoreType::LegacySigned => bundle.std.as_f64(),
    }
}

fn mode_eligible(bundle: &ScoreBundle, direction: Direction) -> bool {
    match direction {
        Direction::Pos => bundle.std >= bundle.std.negate(),
        Direction::Neg => bundle.std <= bundle.std.negate(),
        Direction::Abs => true,
    }
}

fn gene_set_hash(hits: &[usize], hashes: &[u64]) -> u64 {
    hits.iter()
        .fold(0_u64, |value, index| value ^ hashes[*index])
}

fn sample_hits(rng: &mut ChaCha20Rng, n_genes: usize, size: usize) -> Vec<usize> {
    let mut hits = sample(rng, n_genes, size).into_vec();
    hits.sort_unstable();
    hits
}

fn trigamma(mut value: f64) -> f64 {
    let mut result = 0.0;
    while value < 8.0 {
        result += 1.0 / (value * value);
        value += 1.0;
    }
    let inverse = 1.0 / value;
    let inverse2 = inverse * inverse;
    result + inverse + inverse2 / 2.0 + inverse2 * inverse / 6.0
        - inverse2 * inverse2 * inverse / 30.0
        + inverse2 * inverse2 * inverse2 * inverse / 42.0
        - inverse2 * inverse2 * inverse2 * inverse2 * inverse / 30.0
}

fn multilevel_error_from_log(log_pval: f64, sample_size: usize) -> f64 {
    if !log_pval.is_finite() || log_pval > 0.0 || sample_size < 3 {
        return f64::NAN;
    }
    let depth = ((-log_pval / std::f64::consts::LN_2) + 1.0)
        .floor()
        .max(1.0);
    let per_level = trigamma((sample_size as f64 + 1.0) / 2.0) - trigamma(sample_size as f64 + 1.0);
    (depth * per_level).sqrt() / std::f64::consts::LN_2
}

#[pyfunction]
fn multilevel_error(pval: f64, sample_size: usize) -> PyResult<f64> {
    if !pval.is_finite() || pval <= 0.0 || pval > 1.0 {
        return Err(PyValueError::new_err("pval must be finite and in (0, 1]"));
    }
    if sample_size < 3 {
        return Err(PyValueError::new_err("sample_size must be >= 3"));
    }
    Ok(multilevel_error_from_log(pval.ln(), sample_size))
}

fn beta_mean_log(a: usize, b: usize) -> f64 {
    if a == 0 || b == 0 || a > b {
        return f64::NAN;
    }
    -(a..=b).map(|value| 1.0 / value as f64).sum::<f64>()
}

fn simple_error(n_more: usize, nperm: usize) -> f64 {
    if nperm == 0 || n_more == 0 || n_more >= nperm {
        return f64::INFINITY;
    }
    let crude = ((n_more + 1) as f64 / (nperm + 1) as f64).log2();
    let left = match Beta::new(n_more as f64, (nperm - n_more + 1) as f64) {
        Ok(beta) => beta.inverse_cdf(0.025).log2(),
        Err(_) => return f64::INFINITY,
    };
    let right = match Beta::new((n_more + 1) as f64, (nperm - n_more) as f64) {
        Ok(beta) => beta.inverse_cdf(0.975).log2(),
        Err(_) => return f64::INFINITY,
    };
    0.5 * (crude - left).max(right - crude)
}

fn simple_log2_error(n_more: usize, nperm: usize) -> f64 {
    let variance = trigamma((n_more + 1) as f64) - trigamma((nperm + 1) as f64);
    variance.max(0.0).sqrt() / std::f64::consts::LN_2
}

#[derive(Clone, Debug)]
struct LevelRecord {
    bound: CompoundScore,
    all_scores: Vec<(CompoundScore, bool)>,
    high_scores: Vec<(CompoundScore, bool)>,
}

#[derive(Clone, Debug)]
struct RulerSample {
    hits: Vec<usize>,
    compound: CompoundScore,
    eligible: bool,
}

#[derive(Clone, Debug)]
struct RulerOutcome {
    levels: Vec<LevelRecord>,
    accept_rates: Vec<f64>,
    log_mass: f64,
    status: &'static str,
    reason: String,
}

impl RulerOutcome {
    fn query(&self, target: CompoundScore, require_eligible: bool) -> Option<f64> {
        let mut log_probability = 0.0;
        for level in &self.levels {
            if target <= level.bound {
                let numerator = level
                    .all_scores
                    .iter()
                    .filter(|(score, eligible)| {
                        *score >= target && (!require_eligible || *eligible)
                    })
                    .count();
                if numerator == 0 {
                    log_probability += beta_mean_log(1, level.all_scores.len());
                    return Some(log_probability);
                }
                log_probability += beta_mean_log(numerator, level.all_scores.len());
                return Some(log_probability);
            }
            log_probability += beta_mean_log(level.high_scores.len() + 1, level.all_scores.len());
        }

        let last = self.levels.last()?;
        let numerator = last
            .high_scores
            .iter()
            .filter(|(score, eligible)| *score >= target && (!require_eligible || *eligible))
            .count();
        if numerator == 0 {
            log_probability += beta_mean_log(1, last.high_scores.len());
            return Some(log_probability);
        }
        log_probability += beta_mean_log(numerator, last.high_scores.len());
        Some(log_probability)
    }

    fn acceptance_summary(&self) -> (f64, f64) {
        if self.accept_rates.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let minimum = self
            .accept_rates
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let mean = self.accept_rates.iter().sum::<f64>() / self.accept_rates.len() as f64;
        (minimum, mean)
    }
}

fn score_sample(
    weights: &[i64],
    hits: Vec<usize>,
    score_type: ScoreType,
    direction: Direction,
    gene_hashes: &[u64],
) -> RulerSample {
    let bundle = calculate_exact_scores(weights, &hits).expect("valid sampled pathway size");
    RulerSample {
        compound: CompoundScore {
            score: metric_for(score_type, &bundle, direction),
            gene_set_hash: gene_set_hash(&hits, gene_hashes),
        },
        eligible: mode_eligible(&bundle, direction),
        hits,
    }
}

fn try_accepted_swap(
    sample: &mut RulerSample,
    rng: &mut ChaCha20Rng,
    weights: &[i64],
    score_type: ScoreType,
    direction: Direction,
    gene_hashes: &[u64],
    bound: CompoundScore,
) -> bool {
    let size = sample.hits.len();
    let old_position = rng.gen_range(0..size);
    let old_gene = sample.hits[old_position];
    let new_gene = rng.gen_range(0..weights.len());
    // Match fgsea's perturbation accounting: selecting the same member is an
    // accepted no-op move, while selecting another existing member is not.
    if new_gene == old_gene {
        return true;
    }
    if sample.hits.binary_search(&new_gene).is_ok() {
        return false;
    }

    let mut candidate_hits = sample.hits.clone();
    candidate_hits.remove(old_position);
    let insertion = candidate_hits
        .binary_search(&new_gene)
        .unwrap_or_else(|index| index);
    candidate_hits.insert(insertion, new_gene);
    let candidate = score_sample(weights, candidate_hits, score_type, direction, gene_hashes);
    if candidate.compound > bound {
        *sample = candidate;
        true
    } else {
        false
    }
}

fn build_ruler(
    weights: &[i64],
    size: usize,
    sample_size: usize,
    seed: u64,
    score_type: ScoreType,
    direction: Direction,
    target: CompoundScore,
    max_levels: usize,
    log_floor: Option<f64>,
) -> RulerOutcome {
    if max_levels == 0 {
        return RulerOutcome {
            levels: Vec::new(),
            accept_rates: Vec::new(),
            log_mass: 0.0,
            status: "max_level_exceeded",
            reason: "max_levels=0 prevented multilevel initialization".to_string(),
        };
    }

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let gene_hashes: Vec<u64> = (0..weights.len())
        .map(|index| mix64(seed ^ index as u64))
        .collect();
    let mut samples: Vec<RulerSample> = (0..sample_size)
        .map(|_| {
            score_sample(
                weights,
                sample_hits(&mut rng, weights.len(), size),
                score_type,
                direction,
                &gene_hashes,
            )
        })
        .collect();
    let mut levels = Vec::new();
    let mut accept_rates = Vec::new();
    let mut previous_bound: Option<CompoundScore> = None;
    let mut log_mass = 0.0_f64;

    loop {
        if levels.len() >= max_levels {
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "max_level_exceeded",
                reason: format!(
                    "multilevel boundary did not reach the target in {max_levels} levels"
                ),
            };
        }

        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by_key(|index| samples[*index].compound);
        let central = samples[order[sample_size / 2]].compound;
        let mut start = order
            .iter()
            .position(|index| samples[*index].compound >= central)
            .unwrap_or(sample_size);
        if start == 0 {
            let minimum = samples[order[0]].compound;
            start = order
                .iter()
                .position(|index| samples[*index].compound > minimum)
                .unwrap_or(sample_size);
        }
        if start == sample_size {
            let common = samples[order[0]].compound;
            if common >= target {
                let all_scores = order
                    .iter()
                    .map(|index| (samples[*index].compound, samples[*index].eligible))
                    .collect::<Vec<_>>();
                levels.push(LevelRecord {
                    bound: common,
                    high_scores: all_scores.clone(),
                    all_scores,
                });
                return RulerOutcome {
                    levels,
                    accept_rates,
                    log_mass,
                    status: "resolved",
                    reason: "all sampled compound scores were equal at or beyond the target"
                        .to_string(),
                };
            }
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "no_level_progress",
                reason: "all compound (exact ES, gene-set hash) scores were equal".to_string(),
            };
        }

        let bound = samples[order[start - 1]].compound;
        if previous_bound.is_some_and(|previous| bound <= previous) {
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "no_level_progress",
                reason: "compound multilevel boundary failed to increase strictly".to_string(),
            };
        }

        let all_scores = order
            .iter()
            .map(|index| (samples[*index].compound, samples[*index].eligible))
            .collect::<Vec<_>>();
        let high_scores = order[start..]
            .iter()
            .map(|index| (samples[*index].compound, samples[*index].eligible))
            .collect::<Vec<_>>();
        levels.push(LevelRecord {
            bound,
            all_scores,
            high_scores,
        });

        if bound >= target {
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "resolved",
                reason: "compound boundary reached the requested exact ES".to_string(),
            };
        }

        let retained = levels
            .last()
            .expect("a level was just appended")
            .high_scores
            .len();
        log_mass += beta_mean_log(retained + 1, sample_size);
        if log_floor.is_some_and(|floor| log_mass < floor) {
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "eps_floor",
                reason: "remaining unconditional tail mass is below the eps floor".to_string(),
            };
        }

        let high_indices = order[start..].to_vec();
        for low_index in &order[..start] {
            let parent = high_indices[rng.gen_range(0..high_indices.len())];
            samples[*low_index] = samples[parent].clone();
        }

        let accepted_target = sample_size.saturating_mul(size) / 2;
        let maximum_attempts = accepted_target.saturating_mul(10_000).max(10_000);
        let mut accepted = 0_usize;
        let mut attempts = 0_usize;
        while accepted < accepted_target && attempts < maximum_attempts {
            let sample_index = attempts % samples.len();
            attempts += 1;
            if try_accepted_swap(
                &mut samples[sample_index],
                &mut rng,
                weights,
                score_type,
                direction,
                &gene_hashes,
                bound,
            ) {
                accepted += 1;
            }
        }
        let acceptance_rate = if attempts == 0 {
            0.0
        } else {
            accepted as f64 / attempts as f64
        };
        accept_rates.push(acceptance_rate);
        if accepted < accepted_target {
            return RulerOutcome {
                levels,
                accept_rates,
                log_mass,
                status: "mixing_failure",
                reason: format!(
                    "accepted {accepted} of {accepted_target} required swaps in {attempts} attempts"
                ),
            };
        }

        // Match the upstream ruler's second decorrelation phase: after the
        // accepted-move budget is reached, perform the same number of
        // population-distributed proposal attempts once more.
        for decorrelation_attempt in 0..attempts {
            let sample_index = decorrelation_attempt % samples.len();
            let _ = try_accepted_swap(
                &mut samples[sample_index],
                &mut rng,
                weights,
                score_type,
                direction,
                &gene_hashes,
                bound,
            );
        }
        previous_bound = Some(bound);
    }
}

#[derive(Clone, Debug)]
struct PathObservation {
    null_size: usize,
    bundle: ScoreBundle,
    es: f64,
}

#[derive(Clone, Debug)]
struct PendingPath {
    index: usize,
    direction: Direction,
    target: ExactScore,
    mode_count: usize,
}

fn invalid_result(
    es: f64,
    reason: &str,
    ranking_hash: &str,
    null_size: usize,
    approximate: bool,
) -> GseaResult {
    GseaResult {
        es,
        pval: f64::NAN,
        log_pval: f64::NAN,
        log2err: f64::NAN,
        status: "invalid_input".to_string(),
        termination_reason: reason.to_string(),
        n_levels: 0,
        acceptance_rate_min: f64::NAN,
        acceptance_rate_mean: f64::NAN,
        ranking_hash: ranking_hash.to_string(),
        null_curve_size: null_size,
        approximate,
        algorithm_revision: ALGORITHM_REVISION.to_string(),
        debug_info: None,
    }
}

fn unresolved_result(
    observation: &PathObservation,
    status: &str,
    reason: &str,
    ranking_hash: &str,
    approximate: bool,
    outcome: Option<&RulerOutcome>,
) -> GseaResult {
    let (minimum, mean, levels, debug) = if let Some(value) = outcome {
        let (minimum, mean) = value.acceptance_summary();
        let thresholds = value
            .levels
            .iter()
            .map(|level| level.bound.score.as_f64())
            .collect();
        (
            minimum,
            mean,
            value.levels.len(),
            Some(MultilevelDebugInfo {
                thresholds,
                threshold_hashes: value
                    .levels
                    .iter()
                    .map(|level| level.bound.gene_set_hash)
                    .collect(),
                accept_rates: value.accept_rates.clone(),
                current_level: value.levels.len(),
            }),
        )
    } else {
        (f64::NAN, f64::NAN, 0, None)
    };
    GseaResult {
        es: observation.es,
        pval: f64::NAN,
        log_pval: f64::NAN,
        log2err: f64::NAN,
        status: status.to_string(),
        termination_reason: reason.to_string(),
        n_levels: levels,
        acceptance_rate_min: minimum,
        acceptance_rate_mean: mean,
        ranking_hash: ranking_hash.to_string(),
        null_curve_size: observation.null_size,
        approximate,
        algorithm_revision: ALGORITHM_REVISION.to_string(),
        debug_info: debug,
    }
}

#[allow(clippy::too_many_arguments)]
fn resolved_from_log(
    observation: &PathObservation,
    mut log_pval: f64,
    log2err: f64,
    eps: f64,
    ranking_hash: &str,
    approximate: bool,
    reason: &str,
    outcome: Option<&RulerOutcome>,
) -> GseaResult {
    log_pval = log_pval.min(0.0);
    let mut status = "resolved";
    let mut termination_reason = reason.to_string();
    let mut final_log2err = log2err;
    let pval = if eps > 0.0 && log_pval < eps.ln() {
        status = "eps_floor";
        termination_reason = format!("{reason}; estimated P-value is below eps={eps}");
        final_log2err = f64::NAN;
        eps
    } else if log_pval < f64::MIN_POSITIVE.ln() {
        status = "numerical_underflow";
        termination_reason = "P-value underflowed f64; consult log_pval".to_string();
        f64::MIN_POSITIVE
    } else {
        log_pval.exp().max(f64::MIN_POSITIVE)
    };

    let (minimum, mean, levels, debug) = if let Some(value) = outcome {
        let (minimum, mean) = value.acceptance_summary();
        let thresholds = value
            .levels
            .iter()
            .map(|level| level.bound.score.as_f64())
            .collect();
        (
            minimum,
            mean,
            value.levels.len(),
            Some(MultilevelDebugInfo {
                thresholds,
                threshold_hashes: value
                    .levels
                    .iter()
                    .map(|level| level.bound.gene_set_hash)
                    .collect(),
                accept_rates: value.accept_rates.clone(),
                current_level: value.levels.len(),
            }),
        )
    } else {
        (f64::NAN, f64::NAN, 0, None)
    };

    GseaResult {
        es: observation.es,
        pval,
        log_pval,
        log2err: final_log2err,
        status: status.to_string(),
        termination_reason,
        n_levels: levels,
        acceptance_rate_min: minimum,
        acceptance_rate_mean: mean,
        ranking_hash: ranking_hash.to_string(),
        null_curve_size: observation.null_size,
        approximate,
        algorithm_revision: ALGORITHM_REVISION.to_string(),
        debug_info: debug,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    scores: &[f64],
    pathways: &[Vec<usize>],
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type_value: &str,
    bin_width: Option<usize>,
    precheck_n: Option<usize>,
    precheck_eps: Option<f64>,
    mode_value: &str,
    nperm_simple: Option<usize>,
    max_levels: Option<usize>,
) -> Result<Vec<GseaResult>, String> {
    if sample_size < 3 {
        return Err("sample_size must be >= 3".to_string());
    }
    if !eps.is_finite() || !(0.0..=1.0).contains(&eps) {
        return Err("eps must be finite and between 0 and 1".to_string());
    }
    let mode = parse_mode(mode_value)?;
    let score_type = parse_score_type(score_type_value)?;
    let width = bin_width.unwrap_or(0);
    if mode == RunMode::Aligned && width != 0 {
        return Err("mode='aligned' requires bin_width=0 or None".to_string());
    }
    if mode == RunMode::Aligned && (precheck_n.is_some() || precheck_eps.is_some()) {
        return Err("precheck_n/precheck_eps are only available in mode='fast'".to_string());
    }
    if precheck_eps.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err("precheck_eps must be finite and between 0 and 1".to_string());
    }
    let sample_size = if sample_size % 2 == 0 {
        sample_size + 1
    } else {
        sample_size
    };
    let nperm_simple = if mode == RunMode::Fast {
        precheck_n.or(nperm_simple).unwrap_or(64)
    } else {
        nperm_simple.unwrap_or(1000)
    };
    if nperm_simple == 0 {
        return Err("nperm_simple must be > 0".to_string());
    }
    // No implicit statistical depth cap in aligned mode. Callers that need a
    // resource bound must opt into max_levels and receive an explicit failure
    // state if it is reached.
    let max_levels = max_levels.unwrap_or(usize::MAX);
    let ranking = PreparedRanking::new(scores, gsea_param)?;
    let n_genes = ranking.weights.len();
    let ranking_hash = ranking.hash_string();
    let approximate = mode == RunMode::Fast
        || width > 0
        || matches!(score_type, ScoreType::LegacyAbs | ScoreType::LegacySigned);

    let mut observations: Vec<Option<PathObservation>> = Vec::with_capacity(pathways.len());
    let mut initial_results: Vec<Option<GseaResult>> = (0..pathways.len()).map(|_| None).collect();
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for (index, raw_pathway) in pathways.iter().enumerate() {
        if raw_pathway.iter().any(|gene| *gene >= n_genes) {
            observations.push(None);
            initial_results[index] = Some(invalid_result(
                f64::NAN,
                "pathway contains an out-of-range gene index",
                &ranking_hash,
                0,
                approximate,
            ));
            continue;
        }
        let mut hits: Vec<usize> = raw_pathway
            .iter()
            .map(|gene| ranking.rank_pos[*gene])
            .collect();
        hits.sort_unstable();
        hits.dedup();
        if hits.is_empty() || hits.len() >= n_genes {
            observations.push(None);
            initial_results[index] = Some(invalid_result(
                0.0,
                "pathway size must be between 1 and N-1 after deduplication",
                &ranking_hash,
                hits.len(),
                approximate,
            ));
            continue;
        }
        let null_size = if width > 0 {
            let rounded = ((hits.len() + width / 2) / width) * width;
            if rounded == 0 {
                width
            } else {
                rounded
            }
        } else {
            hits.len()
        };
        if null_size == 0 || null_size >= n_genes {
            observations.push(None);
            initial_results[index] = Some(invalid_result(
                f64::NAN,
                "null curve size must be between 1 and N-1",
                &ranking_hash,
                null_size,
                approximate,
            ));
            continue;
        }
        let bundle = calculate_exact_scores(&ranking.weights, &hits)
            .expect("validated non-empty, non-universe pathway");
        let observation = PathObservation {
            null_size,
            bundle,
            es: observed_es(score_type, &bundle),
        };
        observations.push(Some(observation));
        groups.entry(null_size).or_default().push(index);
    }

    let group_results: Vec<Vec<(usize, GseaResult)>> = groups
        .par_iter()
        .map(|(null_size, path_indices)| {
            let simple_seed = derive_seed(
                seed,
                ranking.ranking_hash,
                *null_size,
                Direction::Abs,
                0x5349_4d50_4c45,
            );
            let mut simple_rng = ChaCha20Rng::seed_from_u64(simple_seed);
            let simple_null: Vec<ScoreBundle> = (0..nperm_simple)
                .map(|_| {
                    let hits = sample_hits(&mut simple_rng, n_genes, *null_size);
                    calculate_exact_scores(&ranking.weights, &hits)
                        .expect("validated simple null pathway size")
                })
                .collect();
            let mut resolved = Vec::new();
            let mut pending_by_direction: BTreeMap<Direction, Vec<PendingPath>> = BTreeMap::new();

            for path_index in path_indices {
                let observation = observations[*path_index]
                    .as_ref()
                    .expect("group contains only valid observations");
                if matches!(score_type, ScoreType::Std | ScoreType::LegacySigned)
                    && observation.bundle.std.numerator() == 0
                {
                    resolved.push((
                        *path_index,
                        resolved_from_log(
                            observation,
                            0.0,
                            0.0,
                            eps,
                            &ranking_hash,
                            approximate,
                            "zero enrichment score",
                            None,
                        ),
                    ));
                    continue;
                }
                let direction = direction_for(score_type, &observation.bundle);
                let target = metric_for(score_type, &observation.bundle, direction);
                if target <= target.negate() {
                    resolved.push((
                        *path_index,
                        resolved_from_log(
                            observation,
                            0.0,
                            0.0,
                            eps,
                            &ranking_hash,
                            approximate,
                            "zero enrichment score",
                            None,
                        ),
                    ));
                    continue;
                }

                let require_mode = matches!(score_type, ScoreType::Std | ScoreType::LegacySigned);
                let mode_count = if require_mode {
                    simple_null
                        .iter()
                        .filter(|bundle| mode_eligible(bundle, direction))
                        .count()
                } else {
                    nperm_simple
                };
                let n_more = simple_null
                    .iter()
                    .filter(|bundle| {
                        let tail = metric_for(score_type, bundle, direction) >= target;
                        tail && (!require_mode || mode_eligible(bundle, direction))
                    })
                    .count();
                if mode_count < 10 {
                    resolved.push((
                        *path_index,
                        unresolved_result(
                            observation,
                            "no_level_progress",
                            "fewer than 10 same-sign null samples; increase nperm_simple",
                            &ranking_hash,
                            approximate,
                            None,
                        ),
                    ));
                    continue;
                }

                let simple_probability = ((n_more + 1) as f64 / (mode_count + 1) as f64).min(1.0);
                let simple_log = simple_probability.ln();
                let simple_uncorrected = (n_more + 1) as f64 / (nperm_simple + 1) as f64;
                let mult_error = multilevel_error_from_log(simple_uncorrected.ln(), sample_size);
                let use_simple = mode == RunMode::Fast
                    || score_type == ScoreType::LegacyAbs
                    || mult_error >= simple_error(n_more, nperm_simple);
                if use_simple {
                    resolved.push((
                        *path_index,
                        resolved_from_log(
                            observation,
                            simple_log,
                            simple_log2_error(n_more, nperm_simple),
                            eps,
                            &ranking_hash,
                            approximate,
                            "simple estimator selected by expected log-error",
                            None,
                        ),
                    ));
                } else {
                    pending_by_direction
                        .entry(direction)
                        .or_default()
                        .push(PendingPath {
                            index: *path_index,
                            direction,
                            target,
                            mode_count,
                        });
                }
            }

            for (direction, pending) in pending_by_direction {
                let require_mode = matches!(score_type, ScoreType::Std | ScoreType::LegacySigned);
                let denominator = if require_mode {
                    (pending[0].mode_count + 1) as f64 / (nperm_simple + 1) as f64
                } else {
                    1.0
                };
                let log_floor = if eps > 0.0 {
                    Some(eps.ln() + denominator.ln())
                } else {
                    None
                };
                let maximum_target = pending
                    .iter()
                    .map(|path| CompoundScore {
                        score: path.target,
                        gene_set_hash: 0,
                    })
                    .max()
                    .expect("non-empty pending direction");
                let ruler_seed = derive_seed(
                    seed,
                    ranking.ranking_hash,
                    *null_size,
                    direction,
                    0x4d55_4c54_494c_564c,
                );
                let outcome = build_ruler(
                    &ranking.weights,
                    *null_size,
                    sample_size,
                    ruler_seed,
                    score_type,
                    direction,
                    maximum_target,
                    max_levels,
                    log_floor,
                );
                for path in pending {
                    debug_assert_eq!(path.direction, direction);
                    let observation = observations[path.index]
                        .as_ref()
                        .expect("pending path has a valid observation");
                    let target = CompoundScore {
                        score: path.target,
                        gene_set_hash: 0,
                    };
                    let target_boundary_reached =
                        outcome.levels.iter().any(|level| level.bound >= target);
                    let query = if outcome.status == "resolved" || target_boundary_reached {
                        outcome.query(target, require_mode)
                    } else {
                        None
                    };
                    if let Some(unconditional_log) = query {
                        let denominator = if require_mode {
                            (path.mode_count + 1) as f64 / (nperm_simple + 1) as f64
                        } else {
                            1.0
                        };
                        let conditional_log = (unconditional_log - denominator.ln()).min(0.0);
                        resolved.push((
                            path.index,
                            resolved_from_log(
                                observation,
                                conditional_log,
                                multilevel_error_from_log(conditional_log, sample_size),
                                eps,
                                &ranking_hash,
                                approximate,
                                "multilevel compound boundary resolved",
                                Some(&outcome),
                            ),
                        ));
                    } else if outcome.status == "eps_floor" {
                        let denominator = if require_mode {
                            (path.mode_count + 1) as f64 / (nperm_simple + 1) as f64
                        } else {
                            1.0
                        };
                        let conditional_log = (outcome.log_mass - denominator.ln()).min(0.0);
                        resolved.push((
                            path.index,
                            resolved_from_log(
                                observation,
                                conditional_log,
                                f64::NAN,
                                eps,
                                &ranking_hash,
                                approximate,
                                &outcome.reason,
                                Some(&outcome),
                            ),
                        ));
                    } else {
                        resolved.push((
                            path.index,
                            unresolved_result(
                                observation,
                                outcome.status,
                                &outcome.reason,
                                &ranking_hash,
                                approximate,
                                Some(&outcome),
                            ),
                        ));
                    }
                }
            }
            resolved
        })
        .collect();

    for group in group_results {
        for (index, result) in group {
            initial_results[index] = Some(result);
        }
    }
    Ok(initial_results
        .into_iter()
        .map(|result| result.expect("every pathway receives a result"))
        .collect())
}

#[inline]
fn calculate_es_components_inner(
    scores: &[f64],
    hits: &[usize],
    gsea_param: f64,
) -> (f64, f64, f64) {
    let n_genes = scores.len();
    let n_hits = hits.len();
    if n_hits == 0 || n_hits >= n_genes {
        return (0.0, 0.0, 0.0);
    }
    let mut weights: Vec<f64> = hits
        .iter()
        .map(|index| {
            if gsea_param == 0.0 {
                1.0
            } else {
                scores[*index].abs().powf(gsea_param)
            }
        })
        .collect();
    let mut total: f64 = weights.iter().sum();
    if total == 0.0 {
        weights.fill(1.0);
        total = n_hits as f64;
    }
    let miss_step = 1.0 / (n_genes - n_hits) as f64;
    let mut current = 0.0_f64;
    let mut best_pos = 0.0_f64;
    let mut best_neg = 0.0_f64;
    let mut last: isize = -1;
    for (hit_number, position) in hits.iter().enumerate() {
        current -= (*position as isize - last - 1) as f64 * miss_step;
        best_pos = best_pos.max(current);
        best_neg = best_neg.min(current);
        current += weights[hit_number] / total;
        best_pos = best_pos.max(current);
        best_neg = best_neg.min(current);
        last = *position as isize;
    }
    let standard = match best_pos.total_cmp(&(-best_neg)) {
        Ordering::Greater => best_pos,
        Ordering::Less => best_neg,
        Ordering::Equal => 0.0,
    };
    (standard, best_pos, best_neg)
}

#[inline]
fn calculate_es_inner(scores: &[f64], hits: &[usize], gsea_param: f64) -> f64 {
    calculate_es_components_inner(scores, hits, gsea_param).0
}

#[inline]
fn tail_metric(
    score_type: ScoreType,
    standard: f64,
    positive: f64,
    negative: f64,
    sign: i32,
) -> f64 {
    match score_type {
        ScoreType::Pos => positive,
        ScoreType::Neg => -negative,
        ScoreType::LegacyAbs => positive.max(-negative),
        ScoreType::Std | ScoreType::LegacySigned if sign < 0 => -standard,
        ScoreType::Std | ScoreType::LegacySigned => standard,
    }
}

#[inline]
fn observed_tail_metric(score_type: ScoreType, observed: f64, sign: i32) -> f64 {
    match score_type {
        ScoreType::Pos => observed.max(0.0),
        ScoreType::Neg => -observed.min(0.0),
        ScoreType::LegacyAbs => observed.abs(),
        ScoreType::Std | ScoreType::LegacySigned if sign < 0 => -observed,
        ScoreType::Std | ScoreType::LegacySigned => observed,
    }
}

#[pyfunction]
fn calculate_es(
    scores: PyReadonlyArray1<f64>,
    mut hits: Vec<usize>,
    gsea_param: f64,
) -> PyResult<f64> {
    let scores = scores.as_slice()?;
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("scores must be non-empty and finite"));
    }
    if !gsea_param.is_finite() || gsea_param < 0.0 {
        return Err(PyValueError::new_err("gsea_param must be finite and >= 0"));
    }
    if hits.iter().any(|index| *index >= scores.len()) {
        return Err(PyValueError::new_err("hits contain an out-of-range index"));
    }
    hits.sort_unstable();
    hits.dedup();
    Ok(calculate_es_inner(scores, &hits, gsea_param))
}

fn build_tail_curve_inner(
    scores: &[f64],
    size: usize,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    sign: i32,
) -> Result<TailCurve, String> {
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        return Err("scores must be non-empty and finite".to_string());
    }
    if size == 0 || size >= scores.len() || sample_size == 0 {
        return Err("size must be in [1, N-1] and sample_size must be > 0".to_string());
    }
    if !gsea_param.is_finite() || gsea_param < 0.0 {
        return Err("gsea_param must be finite and >= 0".to_string());
    }
    let parsed = parse_score_type(score_type.unwrap_or("std"))?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut population = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let hits = sample_hits(&mut rng, scores.len(), size);
        let (standard, positive, negative) =
            calculate_es_components_inner(scores, &hits, gsea_param);
        let metric = tail_metric(parsed, standard, positive, negative, sign);
        population.push(metric);
    }
    Ok(TailCurve {
        thresholds: Vec::new(),
        log_probs: vec![0.0],
        populations: vec![population],
        sample_size,
        eps,
        status: "resolved".to_string(),
        termination_reason: "legacy empirical tail curve; aligned mode uses the exact ruler"
            .to_string(),
    })
}

#[pyfunction]
fn query_tail_curve(
    curve: &TailCurve,
    obs_es: f64,
    score_type: Option<&str>,
    sign: Option<i32>,
) -> PyResult<(f64, f64)> {
    let parsed = parse_score_type(score_type.unwrap_or("std")).map_err(PyValueError::new_err)?;
    let metric = observed_tail_metric(parsed, obs_es, sign.unwrap_or(1));
    let population = curve.populations.last().cloned().unwrap_or_default();
    if population.is_empty() {
        return Ok((f64::NAN, f64::NAN));
    }
    let count = population.iter().filter(|value| **value >= metric).count();
    let pval = (count + 1) as f64 / (population.len() + 1) as f64;
    if curve.eps > 0.0 && pval < curve.eps {
        return Ok((curve.eps, f64::NAN));
    }
    Ok((
        pval.max(f64::MIN_POSITIVE),
        multilevel_error_from_log(pval.ln(), curve.sample_size.max(3)),
    ))
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
    sign: i32,
) -> PyResult<TailCurve> {
    if !eps.is_finite() || !(0.0..=1.0).contains(&eps) {
        return Err(PyValueError::new_err(
            "eps must be finite and between 0 and 1",
        ));
    }
    build_tail_curve_inner(
        scores.as_slice()?,
        size,
        sample_size,
        seed,
        gsea_param,
        eps,
        score_type,
        sign,
    )
    .map_err(PyValueError::new_err)
}

#[pyclass]
struct GseaPrerankedRunner {
    pathways: Vec<Vec<usize>>,
    _min_size: usize,
    _max_size: usize,
}

#[pymethods]
impl GseaPrerankedRunner {
    #[new]
    fn new(pathways: Vec<Vec<usize>>, min_size: usize, max_size: usize) -> PyResult<Self> {
        if min_size < 1 || max_size < min_size {
            return Err(PyValueError::new_err(
                "min_size must be >= 1 and max_size must be >= min_size",
            ));
        }
        Ok(Self {
            pathways,
            _min_size: min_size,
            _max_size: max_size,
        })
    }

    #[pyo3(signature = (scores, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None, precheck_n=None, precheck_eps=None, mode="aligned", nperm_simple=None, max_levels=None))]
    #[allow(clippy::too_many_arguments)]
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
        precheck_eps: Option<f64>,
        mode: &str,
        nperm_simple: Option<usize>,
        max_levels: Option<usize>,
    ) -> PyResult<Vec<GseaResult>> {
        run_engine(
            scores.as_slice()?,
            &self.pathways,
            sample_size,
            seed,
            gsea_param,
            eps,
            score_type.unwrap_or("std"),
            bin_width,
            precheck_n,
            precheck_eps,
            mode,
            nperm_simple,
            max_levels,
        )
        .map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None, mode="aligned", nperm_simple=None, max_levels=None))]
#[allow(clippy::too_many_arguments)]
fn fgsea_multilevel_batched(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    bin_width: Option<usize>,
    mode: &str,
    nperm_simple: Option<usize>,
    max_levels: Option<usize>,
) -> PyResult<Vec<GseaResult>> {
    run_engine(
        scores.as_slice()?,
        &pathways,
        sample_size,
        seed,
        gsea_param,
        eps,
        score_type.unwrap_or("std"),
        bin_width,
        None,
        None,
        mode,
        nperm_simple,
        max_levels,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None, bin_width=None, mode="aligned", nperm_simple=None, max_levels=None))]
#[allow(clippy::too_many_arguments)]
fn fgsea_multilevel_batched_scores(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    bin_width: Option<usize>,
    mode: &str,
    nperm_simple: Option<usize>,
    max_levels: Option<usize>,
) -> PyResult<Vec<GseaResult>> {
    fgsea_multilevel_batched(
        scores,
        pathways,
        sample_size,
        seed,
        gsea_param,
        eps,
        score_type,
        bin_width,
        mode,
        nperm_simple,
        max_levels,
    )
}

#[pyfunction]
#[pyo3(signature = (scores, pathways, sample_size, seed, gsea_param, eps, score_type=None, mode="aligned", nperm_simple=None, max_levels=None))]
#[allow(clippy::too_many_arguments)]
fn fgsea_multilevel(
    scores: PyReadonlyArray1<f64>,
    pathways: Vec<Vec<usize>>,
    sample_size: usize,
    seed: u64,
    gsea_param: f64,
    eps: f64,
    score_type: Option<&str>,
    mode: &str,
    nperm_simple: Option<usize>,
    max_levels: Option<usize>,
) -> PyResult<Vec<GseaResult>> {
    run_engine(
        scores.as_slice()?,
        &pathways,
        sample_size,
        seed,
        gsea_param,
        eps,
        score_type.unwrap_or("std"),
        None,
        None,
        None,
        mode,
        nperm_simple,
        max_levels,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (scores, sizes, nperm, seed, gsea_param, score_type=None))]
fn get_random_es_means(
    scores: PyReadonlyArray1<f64>,
    sizes: Vec<usize>,
    nperm: usize,
    seed: u64,
    gsea_param: f64,
    score_type: Option<&str>,
) -> PyResult<Vec<(f64, f64)>> {
    if nperm == 0 {
        return Err(PyValueError::new_err("nperm must be > 0"));
    }
    let ranking =
        PreparedRanking::new(scores.as_slice()?, gsea_param).map_err(PyValueError::new_err)?;
    let score_type =
        parse_score_type(score_type.unwrap_or("std")).map_err(PyValueError::new_err)?;
    let n_genes = ranking.weights.len();
    if sizes.iter().any(|size| *size == 0 || *size >= n_genes) {
        return Err(PyValueError::new_err(
            "every null pathway size must be between 1 and N-1",
        ));
    }
    let results: Vec<(f64, f64)> = sizes
        .par_iter()
        .map(|size| {
            let task_seed = derive_seed(
                seed,
                ranking.ranking_hash,
                *size,
                Direction::Abs,
                0x4e45_535f_4d45_414e,
            );
            let mut rng = ChaCha20Rng::seed_from_u64(task_seed);
            let mut positive_sum = 0.0;
            let mut positive_count = 0_usize;
            let mut negative_sum = 0.0;
            let mut negative_count = 0_usize;
            for _ in 0..nperm {
                let hits = sample_hits(&mut rng, n_genes, *size);
                let bundle = calculate_exact_scores(&ranking.weights, &hits)
                    .expect("validated NES null pathway size");
                let es = match score_type {
                    ScoreType::Pos => bundle.pos.as_f64(),
                    ScoreType::Neg => bundle.neg.as_f64(),
                    ScoreType::Std | ScoreType::LegacyAbs | ScoreType::LegacySigned => {
                        bundle.std.as_f64()
                    }
                };
                if es >= 0.0 {
                    positive_sum += es;
                    positive_count += 1;
                }
                if es <= 0.0 {
                    negative_sum += es;
                    negative_count += 1;
                }
            }
            let positive_mean = if positive_count == 0 {
                f64::NAN
            } else {
                positive_sum / positive_count as f64
            };
            let negative_mean = if negative_count == 0 {
                f64::NAN
            } else {
                negative_sum / negative_count as f64
            };
            (positive_mean, negative_mean)
        })
        .collect();
    Ok(results)
}

#[pyfunction]
fn algorithm_revision() -> &'static str {
    ALGORITHM_REVISION
}

#[pymodule]
fn _core(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(fgsea_multilevel, module)?)?;
    module.add_function(wrap_pyfunction!(fgsea_multilevel_batched, module)?)?;
    module.add_function(wrap_pyfunction!(fgsea_multilevel_batched_scores, module)?)?;
    module.add_function(wrap_pyfunction!(get_random_es_means, module)?)?;
    module.add_function(wrap_pyfunction!(build_tail_curve, module)?)?;
    module.add_function(wrap_pyfunction!(query_tail_curve, module)?)?;
    module.add_function(wrap_pyfunction!(calculate_es, module)?)?;
    module.add_function(wrap_pyfunction!(multilevel_error, module)?)?;
    module.add_function(wrap_pyfunction!(algorithm_revision, module)?)?;
    module.add_class::<MultilevelDebugInfo>()?;
    module.add_class::<GseaResult>()?;
    module.add_class::<TailCurve>()?;
    module.add_class::<GseaPrerankedRunner>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_scores_compare_equal_rationals_without_floats() {
        let half_a = ExactScore {
            hit_total: 2,
            hit_sum: 1,
            miss_total: 3,
            miss_count: 0,
        };
        let half_b = ExactScore {
            hit_total: 4,
            hit_sum: 2,
            miss_total: 7,
            miss_count: 0,
        };
        let smaller = ExactScore {
            hit_total: 3,
            hit_sum: 1,
            miss_total: 5,
            miss_count: 0,
        };
        assert_eq!(half_a, half_b);
        assert!(half_a > smaller);
    }

    #[test]
    fn standard_score_is_zero_for_equal_positive_and_negative_excursions() {
        let bundle = calculate_exact_scores(&[1, 1, 1, 1], &[1, 2]).unwrap();
        assert_eq!(bundle.pos.as_f64(), 0.5);
        assert_eq!(bundle.neg.as_f64(), -0.5);
        assert_eq!(bundle.std, ExactScore::zero(2, 2));
        assert_eq!(
            metric_for(ScoreType::Std, &bundle, Direction::Pos),
            ExactScore::zero(2, 2)
        );
        assert_eq!(
            metric_for(ScoreType::Std, &bundle, Direction::Neg),
            ExactScore::zero(2, 2)
        );
        assert_eq!(
            metric_for(ScoreType::Pos, &bundle, Direction::Pos).as_f64(),
            0.5
        );
    }

    #[test]
    fn float_score_helpers_match_std_pos_and_neg_tie_semantics() {
        let (standard, positive, negative) =
            calculate_es_components_inner(&[1.0, 1.0, 1.0, 1.0], &[1, 2], 0.0);
        assert_eq!(standard, 0.0);
        assert_eq!(positive, 0.5);
        assert_eq!(negative, -0.5);
        assert_eq!(
            tail_metric(ScoreType::Pos, standard, positive, negative, -1),
            0.5
        );
        assert_eq!(
            tail_metric(ScoreType::Neg, standard, positive, negative, 1),
            0.5
        );
        assert_eq!(
            tail_metric(ScoreType::LegacyAbs, standard, positive, negative, 1),
            0.5
        );
        assert_eq!(observed_tail_metric(ScoreType::Pos, 0.5, -1), 0.5);
        assert_eq!(observed_tail_metric(ScoreType::Neg, -0.5, 1), 0.5);
    }

    #[test]
    fn compound_score_breaks_exact_es_ties_with_hash() {
        let score = ExactScore {
            hit_total: 2,
            hit_sum: 1,
            miss_total: 3,
            miss_count: 0,
        };
        let low = CompoundScore {
            score,
            gene_set_hash: 7,
        };
        let high = CompoundScore {
            score,
            gene_set_hash: 8,
        };
        assert!(low < high);
    }

    #[test]
    fn ruler_query_uses_pseudocount_for_zero_eligible_hits() {
        let target = CompoundScore {
            score: ExactScore::zero(2, 2),
            gene_set_hash: 0,
        };
        let above = CompoundScore {
            score: ExactScore {
                hit_total: 2,
                hit_sum: 1,
                miss_total: 2,
                miss_count: 0,
            },
            gene_set_hash: 1,
        };
        let outcome = RulerOutcome {
            levels: vec![LevelRecord {
                bound: above,
                all_scores: vec![(above, false)],
                high_scores: vec![(above, false)],
            }],
            accept_rates: Vec::new(),
            log_mass: 0.0,
            status: "resolved",
            reason: String::new(),
        };
        let log_probability = outcome.query(target, true).unwrap();
        assert!(log_probability.is_finite());
        assert!(log_probability <= 0.0);
    }

    fn test_observation() -> PathObservation {
        let bundle = calculate_exact_scores(&[1, 1, 1, 1], &[0]).unwrap();
        PathObservation {
            null_size: 1,
            bundle,
            es: bundle.std.as_f64(),
        }
    }

    #[test]
    fn numerical_underflow_retains_log_probability_and_explicit_status() {
        let result = resolved_from_log(
            &test_observation(),
            -1000.0,
            2.0,
            0.0,
            "ranking",
            false,
            "test",
            None,
        );
        assert_eq!(result.status, "numerical_underflow");
        assert_eq!(result.pval, f64::MIN_POSITIVE);
        assert_eq!(result.log_pval, -1000.0);
        assert!(result.log2err.is_finite());
    }

    #[test]
    fn unresolved_failure_states_never_expose_a_numeric_pvalue() {
        for status in ["no_level_progress", "mixing_failure"] {
            let result = unresolved_result(
                &test_observation(),
                status,
                "injected unit-test failure",
                "ranking",
                false,
                None,
            );
            assert_eq!(result.status, status);
            assert!(result.pval.is_nan());
            assert!(result.log_pval.is_nan());
            assert!(result.log2err.is_nan());
        }
    }
}
