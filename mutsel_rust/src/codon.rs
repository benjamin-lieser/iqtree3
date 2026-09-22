//! Codon-level (61-state) version of the mutation-selection model.
//!
//! The mutation process is a single 4-state nucleotide rate matrix (see
//! [`Mu4`]), shared identically across all 3 codon positions, expanded into
//! a 61x61 codon mutation matrix (see [`expand_nt_to_codon_mu`]): the rate
//! between two codons that differ at exactly one nucleotide position is the
//! corresponding nucleotide rate at that position; codons differing at 0 or
//! >=2 positions get rate 0.
//!
//! `Mu4`/`expand_nt_to_codon_mu` build the *literal* mutation rate matrix --
//! there is no "equilibrium packed into the diagonal" convention here (the
//! diagonal is unused and just held at a fixed positive placeholder so
//! `.log()` stays finite). Instead, the Halpern & Bruno (1998) fixation
//! probability is computed directly from the mutation rate ratio, exactly
//! as in the original paper: the scaled selection coefficient between
//! states i and j is `S_ij = ln((pi_j * Mu_ji) / (pi_i * Mu_ij))` (see
//! [`calc_rate_matrix_codon`]) -- no separately-tracked neutral equilibrium
//! is needed.
//!
//! The fitness/selection landscape is *exactly* the amino-acid model's: the
//! same per-site 20-dimensional PCA-parametrized amino-acid log-frequency
//! vector is used, broadcast to the 61 codons through the genetic code (see
//! [`broadcast_aa_fitness_to_codon`]) so that synonymous codons of the same
//! amino acid always share the same target frequency `pi`.
//!
//! This module only supports the standard genetic code (61 sense codons) —
//! [`FelsensteinTree`](phylo_grad::FelsensteinTree) is generic over a
//! compile-time constant, so supporting other genetic codes (60-63 sense
//! codons) would need additional fixed-size instantiations.

use candle_core::Tensor;

/// Number of sense codons in the standard genetic code.
pub const N_CODON: usize = 61;

/// The standard genetic code (NCBI translation table 1), as a 64-entry
/// codon -> amino-acid (or '*' for stop) string. Codon index `i` corresponds
/// to nucleotides `(i/16, (i%16)/4, i%4)` with nucleotide coding
/// A=0,C=1,G=2,T=3 -- i.e. lexicographic order over "ACGT". This mirrors
/// IQ-TREE's `genetic_code1` table (`alignment/alignment.cpp`) byte for
/// byte, and the amino-acid letters use the same "ARNDCQEGHILKMFPSTWYV"
/// alphabet/order as IQ-TREE's `symbols_protein` (`alignment/alignment.cpp`),
/// so amino-acid state indices produced here (0-19) are directly compatible
/// with IQ-TREE's amino-acid state numbering.
const GENETIC_CODE_1: &str =
    "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF";

const AA_ALPHABET: &str = "ARNDCQEGHILKMFPSTWYV";

/// Builds the `codon_nt`/`codon_aa` lookup tables for the standard genetic
/// code. `codon_nt[state]` gives the nucleotide (0-3) at each of the 3
/// codon positions for the non-stop codon with dense state index `state`.
/// `codon_aa[state]` gives its encoded amino-acid state (0-19). States are
/// assigned by walking the 64 raw codons in order and skipping stop codons,
/// matching IQ-TREE's `Alignment::initCodon`.
pub fn standard_genetic_code_tables() -> ([[u8; 3]; N_CODON], [u8; N_CODON]) {
    let code: Vec<char> = GENETIC_CODE_1.chars().collect();
    assert_eq!(code.len(), 64, "genetic code table must have 64 entries");
    let aa_alphabet: Vec<char> = AA_ALPHABET.chars().collect();

    let mut codon_nt = [[0u8; 3]; N_CODON];
    let mut codon_aa = [0u8; N_CODON];
    let mut state = 0usize;
    for raw in 0..64usize {
        let aa_char = code[raw];
        if aa_char == '*' {
            continue;
        }
        assert!(
            state < N_CODON,
            "standard genetic code has more than {} sense codons",
            N_CODON
        );
        codon_nt[state] = [
            (raw / 16) as u8,
            ((raw % 16) / 4) as u8,
            (raw % 4) as u8,
        ];
        let aa_index = aa_alphabet
            .iter()
            .position(|&c| c == aa_char)
            .unwrap_or_else(|| panic!("Unknown amino acid letter '{}' in genetic code table", aa_char));
        codon_aa[state] = aa_index as u8;
        state += 1;
    }
    assert_eq!(
        state, N_CODON,
        "standard genetic code did not yield {} sense codons (got {})",
        N_CODON, state
    );
    (codon_nt, codon_aa)
}

/// Builds the literal 4-state nucleotide mutation rate matrix from 10
/// log-parameters given as a 1D tensor: the first 6 are symmetric
/// exchangeabilities (`r_ab == r_ba`, in the same pair order as
/// `data::load_lower_R_with_equi`'s lower triangle: (1,0),(2,0),(2,1),
/// (3,0),(3,1),(3,2)), the last 4 are per-state frequency-like scale
/// factors. `Mu[a,b] = r_ab * freq[b]` for `a != b` -- this *is* the actual
/// mutation rate from `a` to `b`, not a normalized/softmaxed frequency.
/// Because the exchangeabilities are symmetric, `Mu` satisfies detailed
/// balance for the equilibrium implied by `freq`:
/// `Mu[a,b] / Mu[b,a] == freq[b] / freq[a]` -- which is exactly what
/// [`calc_rate_matrix_codon`] uses instead of tracking a separate
/// equilibrium. The diagonal is unused; it's fixed at 1.0 so `.log()`
/// stays finite.
#[allow(non_snake_case)]
pub fn Mu4(log_params: &Tensor) -> Tensor {
    assert_eq!(log_params.elem_count(), 10, "Mu4 expects 10 log-parameters");
    let device = candle_core::Device::Cpu;
    let params = log_params.reshape(&[10]).unwrap().exp().unwrap();
    let exch = params.narrow(0, 0, 6).unwrap();
    let freq = params.narrow(0, 6, 4).unwrap();

    // Scatter the 6 symmetric exchangeabilities into the off-diagonal
    // entries of a 4x4 matrix (diagonal masked to 0), same pair order as
    // `data::load_lower_R_with_equi`: (1,0),(2,0),(2,1),(3,0),(3,1),(3,2).
    let idx: [u32; 16] = [
        0, 0, 1, 3, //
        0, 0, 2, 4, //
        1, 2, 0, 5, //
        3, 4, 5, 0, //
    ];
    let mask: [f64; 16] = [
        0.0, 1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, 1.0, //
        1.0, 1.0, 0.0, 1.0, //
        1.0, 1.0, 1.0, 0.0, //
    ];
    let idx_tensor = Tensor::from_vec(idx.to_vec(), &[16], &device).unwrap();
    let mask_tensor = Tensor::from_vec(mask.to_vec(), &[4, 4], &device).unwrap();

    let exch_matrix = exch
        .index_select(&idx_tensor, 0)
        .unwrap()
        .reshape(&[4, 4])
        .unwrap()
        .mul(&mask_tensor)
        .unwrap();

    let off_diagonal = exch_matrix.broadcast_mul(&freq.unsqueeze(0).unwrap()).unwrap();

    (off_diagonal + Tensor::eye(4, candle_core::DType::F64, &device).unwrap()).unwrap()
}

/// Expands the 4x4 nucleotide mutation rate matrix into the 61x61 codon
/// mutation matrix implied by applying it independently and identically at
/// each of the 3 codon positions. The rate between two codons differing at
/// exactly one nucleotide position is the corresponding nucleotide rate;
/// codons differing at 0 or >=2 positions get rate 0. The diagonal is
/// unused (like `Mu4`'s); it's fixed at 1.0 so `.log()` stays finite.
///
/// The construction is built entirely from `index_select`/elementwise ops
/// on `nt_mu`, so gradients flow back to the underlying rate-matrix
/// log-parameters.
pub fn expand_nt_to_codon_mu(nt_mu: &Tensor, codon_nt: &[[u8; 3]; N_CODON]) -> Tensor {
    let device = candle_core::Device::Cpu;
    let flat_nt_mu = nt_mu.reshape(&[16]).unwrap();

    let mut off_diag: Option<Tensor> = None;
    for pos in 0..3usize {
        let mut idx = vec![0u32; N_CODON * N_CODON];
        let mut mask = vec![0f64; N_CODON * N_CODON];
        for c1 in 0..N_CODON {
            for c2 in 0..N_CODON {
                if c1 == c2 {
                    continue;
                }
                let diff_positions: Vec<usize> =
                    (0..3).filter(|&p| codon_nt[c1][p] != codon_nt[c2][p]).collect();
                if diff_positions.len() == 1 && diff_positions[0] == pos {
                    let n1 = codon_nt[c1][pos] as u32;
                    let n2 = codon_nt[c2][pos] as u32;
                    idx[c1 * N_CODON + c2] = n1 * 4 + n2;
                    mask[c1 * N_CODON + c2] = 1.0;
                }
            }
        }
        let idx_tensor = Tensor::from_vec(idx, &[N_CODON * N_CODON], &device).unwrap();
        let mask_tensor = Tensor::from_vec(mask, &[N_CODON, N_CODON], &device).unwrap();

        let gathered = flat_nt_mu
            .index_select(&idx_tensor, 0)
            .unwrap()
            .reshape(&[N_CODON, N_CODON])
            .unwrap();
        let masked = gathered.mul(&mask_tensor).unwrap();

        off_diag = Some(match off_diag {
            None => masked,
            Some(acc) => (acc + masked).unwrap(),
        });
    }
    let off_diag = off_diag.unwrap();

    (off_diag + Tensor::eye(N_CODON, candle_core::DType::F64, &device).unwrap()).unwrap()
}

/// Broadcasts a per-site amino-acid log-frequency tensor `[L, 20]` to a
/// per-site codon log-frequency tensor `[L, 61]` through the genetic code:
/// every codon gets the log-frequency (fitness) of the amino acid it
/// encodes, so synonymous codons always share the same fitness.
pub fn broadcast_aa_fitness_to_codon(log_pi_aa: &Tensor, codon_aa: &[u8; N_CODON]) -> Tensor {
    let idx: Vec<u32> = codon_aa.iter().map(|&a| a as u32).collect();
    let idx_tensor = Tensor::from_vec(idx, &[N_CODON], &candle_core::Device::Cpu).unwrap();
    log_pi_aa.index_select(&idx_tensor, 1).unwrap()
}

/// Codon-dimension (61-state) counterpart of `model::calc_rate_matrix`,
/// using the Halpern & Bruno (1998) fixation-probability formula directly
/// in its original form -- the scaled selection coefficient between states
/// `i` and `j` is `S_ij = ln((pi_j * Mu_ji) / (pi_i * Mu_ij))`, computed
/// straight from the mutation rate ratio `Mu_ji / Mu_ij` rather than from a
/// separately-tracked neutral equilibrium. A separate, dimension-specialized
/// copy (not a generalization of `model::calc_rate_matrix`) so the
/// amino-acid path is left untouched.
pub fn calc_rate_matrix_codon(
    Mu: &Tensor,
    log_pi: &Tensor,
    global_scaling: &Tensor,
) -> (Tensor, Tensor) {
    let L = log_pi.dims()[0];

    // Codon pairs differing at more than one nucleotide position (and the
    // unused diagonal, at least for `expand_nt_to_codon_mu`'s output) are
    // structurally 0 in Mu -- a direct `.log()` would give -inf, and when
    // both Mu[i,j] and Mu[j,i] are 0 the ratio below becomes NaN. Adding a
    // tiny epsilon here (only for the log/ratio computation, not for `Mu`
    // itself below) keeps that finite; those entries still end up exactly
    // 0 in Q since they're multiplied by the *unperturbed* Mu, which is
    // genuinely 0 there.
    let log_mu = (Mu + 1e-30).unwrap().log().unwrap();
    // log_mu_ratio[i,j] = log(Mu[j,i]) - log(Mu[i,j])
    let log_mu_ratio = log_mu.t().unwrap().sub(&log_mu).unwrap();

    // log_pi_diff[l,i,j] = (log_pi[l,j] - log_pi[l,i]) + log_mu_ratio[i,j]
    //                    = ln((pi_j * Mu_ji) / (pi_i * Mu_ij))
    let log_pi_diff = log_pi
        .unsqueeze(1)
        .unwrap()
        .broadcast_sub(&log_pi.unsqueeze(2).unwrap())
        .unwrap()
        .broadcast_add(&log_mu_ratio.unsqueeze(0).unwrap())
        .unwrap();

    let fixation = log_pi_diff.apply_op1(crate::model::GOp {}).unwrap();

    let R = Mu.unsqueeze(0).unwrap().broadcast_as(&[L, N_CODON, N_CODON]).unwrap();

    let Q = R.mul(&fixation).unwrap();

    let pi = candle_nn::ops::softmax(&log_pi, 1).unwrap();
    let pi_sqrt = pi.sqrt().unwrap();

    let sqrt_pi_Q = pi_sqrt.unsqueeze(2).unwrap().broadcast_mul(&Q).unwrap();
    let S = sqrt_pi_Q.broadcast_div(&pi_sqrt.unsqueeze(1).unwrap()).unwrap();

    let S = S.broadcast_mul(&global_scaling).unwrap();
    (S, pi_sqrt)
}

/// Codon-dimension (61-state) counterpart of
/// `model::substitution_rates_tensor` (which hardcodes 20 states).
pub fn substitution_rates_tensor_codon(S: &Tensor, sqrt_pi: &Tensor) -> Tensor {
    let pi_outer = sqrt_pi.unsqueeze(2).unwrap().broadcast_mul(&sqrt_pi.unsqueeze(1).unwrap()).unwrap();
    let weighted = S.mul(&pi_outer).unwrap();

    let total = weighted.sum(2).unwrap().sum(1).unwrap();

    let eye = Tensor::eye(N_CODON, candle_core::DType::F64, S.device()).unwrap();
    let diag = weighted.broadcast_mul(&eye).unwrap().sum(2).unwrap().sum(1).unwrap();

    total.sub(&diag).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn standard_genetic_code_has_expected_size() {
        let (codon_nt, codon_aa) = standard_genetic_code_tables();
        // sanity: every nucleotide index is in range
        for triplet in codon_nt.iter() {
            for &n in triplet.iter() {
                assert!(n < 4);
            }
        }
        for &aa in codon_aa.iter() {
            assert!(aa < 20);
        }
    }

    #[test]
    fn codon_mu_is_reversible() {
        let log_r = Tensor::rand(-1.0, 1.0, &[10], &candle_core::Device::Cpu).unwrap();
        let nt_mu = Mu4(&log_r);
        let (codon_nt, _codon_aa) = standard_genetic_code_tables();
        let codon_mu = expand_nt_to_codon_mu(&nt_mu, &codon_nt);

        // log_pi doesn't matter for reversibility of S, use equal fitness everywhere.
        let log_pi = Tensor::zeros(&[1, N_CODON], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap();
        let (s, _sqrt_pi) = calc_rate_matrix_codon(
            &codon_mu,
            &log_pi,
            &Tensor::full(1.0, &[], &candle_core::Device::Cpu).unwrap(),
        );

        let diff = (&s - &s.transpose(1, 2).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        assert!(diff < 1e-10, "S - S^T should be ~0, got {}", diff);
    }

    #[test]
    fn single_nt_difference_rate_matches_nucleotide_gtr() {
        let log_r = Tensor::rand(-1.0, 1.0, &[10], &candle_core::Device::Cpu).unwrap();
        let nt_mu = Mu4(&log_r);
        let (codon_nt, codon_aa) = standard_genetic_code_tables();
        let codon_mu = expand_nt_to_codon_mu(&nt_mu, &codon_nt);

        let nt_mu_vec = nt_mu.to_vec2::<f64>().unwrap();
        let codon_mu_vec = codon_mu.to_vec2::<f64>().unwrap();

        let mut checked_synonymous = 0;
        let mut checked_multi_diff = 0;

        for c1 in 0..N_CODON {
            for c2 in 0..N_CODON {
                if c1 == c2 {
                    continue;
                }
                let diff_positions: Vec<usize> =
                    (0..3).filter(|&p| codon_nt[c1][p] != codon_nt[c2][p]).collect();

                if diff_positions.len() == 1 {
                    let p = diff_positions[0];
                    let expected = nt_mu_vec[codon_nt[c1][p] as usize][codon_nt[c2][p] as usize];
                    let got = codon_mu_vec[c1][c2];
                    assert!(
                        (expected - got).abs() < 1e-12,
                        "single-nt-diff codon rate mismatch: expected {}, got {}",
                        expected,
                        got
                    );
                    if codon_aa[c1] == codon_aa[c2] {
                        checked_synonymous += 1;
                    }
                } else {
                    let got = codon_mu_vec[c1][c2];
                    assert_eq!(got, 0.0, "codon pair differing at {} positions should have rate 0", diff_positions.len());
                    checked_multi_diff += 1;
                }
            }
        }

        assert!(checked_synonymous > 0, "expected at least one synonymous single-nt-diff pair");
        assert!(checked_multi_diff > 0, "expected at least one multi-nt-diff pair");
    }

    #[test]
    fn synonymous_rate_equals_pure_mutation_rate() {
        // `calc_rate_matrix_codon` derives the fixation probability from
        // the mutation rate ratio Mu_ji/Mu_ij, which for two states with
        // unequal nucleotide frequencies at the differing position is not
        // 1 in general, even when the two codons share the same target
        // (amino-acid) frequency: the Halpern & Bruno formula folds
        // mutation bias into the fixation probability by design.
        //
        // To isolate and directly test "fitness identical to the amino
        // acid model" (equal target frequency for synonymous codons implies
        // equal Halpern-Bruno selection *coefficient* only when the
        // mutation rate ratio is itself 1, i.e. g(0)=1 and rate = pure
        // mutation rate), use a nucleotide rate matrix with uniform
        // frequency factors (the last 4 of the 10 log-parameters all 0, so
        // freq = [1,1,1,1]) -- arbitrary (random) exchangeabilities are
        // fine, since with equal freq, Mu[a,b] == Mu[b,a] for every pair.
        let log_r = Tensor::cat(
            &[
                &Tensor::rand(-1.0, 1.0, &[6], &candle_core::Device::Cpu).unwrap(),
                &Tensor::zeros(&[4], candle_core::DType::F64, &candle_core::Device::Cpu).unwrap(),
            ],
            0,
        )
        .unwrap();
        let nt_mu = Mu4(&log_r);
        let (codon_nt, codon_aa) = standard_genetic_code_tables();
        let codon_mu = expand_nt_to_codon_mu(&nt_mu, &codon_nt);

        // random per-site amino-acid fitness, broadcast to codons
        let log_pi_aa = Tensor::rand(-1.0, 1.0, &[1, 20], &candle_core::Device::Cpu).unwrap();
        let log_pi_codon = broadcast_aa_fitness_to_codon(&log_pi_aa, &codon_aa);

        let (s, sqrt_pi) = calc_rate_matrix_codon(
            &codon_mu,
            &log_pi_codon,
            &Tensor::full(1.0, &[], &candle_core::Device::Cpu).unwrap(),
        );

        let s_vec = s.i(0).unwrap().to_vec2::<f64>().unwrap();
        let sqrt_pi_vec = sqrt_pi.i(0).unwrap().to_vec1::<f64>().unwrap();
        let codon_mu_vec = codon_mu.to_vec2::<f64>().unwrap();

        let mut checked = 0;
        for c1 in 0..N_CODON {
            for c2 in 0..N_CODON {
                if c1 == c2 || codon_aa[c1] != codon_aa[c2] {
                    continue;
                }
                let diff_positions: usize =
                    (0..3).filter(|&p| codon_nt[c1][p] != codon_nt[c2][p]).count();
                if diff_positions != 1 {
                    continue;
                }
                // Q_ij = S_ij * sqrt_pi_j / sqrt_pi_i
                let q = s_vec[c1][c2] * sqrt_pi_vec[c2] / sqrt_pi_vec[c1];
                let expected = codon_mu_vec[c1][c2];
                assert!(
                    (q - expected).abs() < 1e-8,
                    "synonymous rate should equal pure mutation rate: expected {}, got {}",
                    expected,
                    q
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "expected at least one synonymous single-nt-diff pair to check");
    }

    #[test]
    fn substitution_rates_tensor_codon_matches_manual_sum() {
        let log_r = Tensor::rand(-1.0, 1.0, &[10], &candle_core::Device::Cpu).unwrap();
        let nt_mu = Mu4(&log_r);
        let (codon_nt, _codon_aa) = standard_genetic_code_tables();
        let codon_mu = expand_nt_to_codon_mu(&nt_mu, &codon_nt);

        let log_pi = Tensor::rand(-1.0, 1.0, &[1, N_CODON], &candle_core::Device::Cpu).unwrap();
        let (s, sqrt_pi) = calc_rate_matrix_codon(
            &codon_mu,
            &log_pi,
            &Tensor::full(1.0, &[], &candle_core::Device::Cpu).unwrap(),
        );

        let rate_tensor = substitution_rates_tensor_codon(&s, &sqrt_pi)
            .to_vec1::<f64>()
            .unwrap();

        let s_vec = s.to_vec3::<f64>().unwrap();
        let sqrt_pi_vec = sqrt_pi.to_vec2::<f64>().unwrap();
        let mut manual = 0.0;
        for i in 0..N_CODON {
            for j in 0..N_CODON {
                if i != j {
                    manual += s_vec[0][i][j] * sqrt_pi_vec[0][j] * sqrt_pi_vec[0][i];
                }
            }
        }

        assert!((rate_tensor[0] - manual).abs() < 1e-8);
    }
}
