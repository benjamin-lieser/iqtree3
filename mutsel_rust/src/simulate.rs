//! Simulation of alignments under a fitted site-specific model (parametric bootstrap / posterior predictive).
//!
//! Two model sources:
//!
//!   --sitemodel <run>.sitemodel --tree <run>.treefile
//!       The binary site model file IQ-TREE writes for (and reads with --site-model-file) any site-specific model,
//!       independent of how it was fitted (MAP, generalized normal prior, HMC, ...): per site the frequencies pi and
//!       the 190 exchangeabilities R (upper triangle, row-major), Q_ab = R_ab pi_b. IQ-TREE does not normalize these
//!       per site (modelfactory.cpp), so the branch lengths of the tree the run inferred under them are used as
//!       they are. A file with the header "DRAWS k" holds k consecutive models per site, which IQ-TREE treats as an
//!       equally weighted mixture: every replicate draws one of them per site.
//!
//!   --npz <run>.hmc.npz
//!       The output of hmc::sample_internal: the tree the model was fitted on (`parents`, `branch_lengths`), the
//!       mutation process (`log_exchangeabilities`, `rate_scaling`) and the site frequencies and site rates.
//!       `--mode mean` (default) simulates under the posterior mean frequencies and site rates (the model of the
//!       .sitemodel with MUTSEL_TREE_DRAWS=1), `--mode posterior` draws every site of every replicate from the
//!       site's posterior samples (posterior predictive simulation; Mu, the rate scaling and the branch lengths stay
//!       at their EM estimates). Leaves are numbered by their index in the alignment the run was given (see
//!       prepare_mutsel_tree in utils/mutsel_wrapper.cpp), so --alignment is required and must be that file.
//!       `--tree <newick>` replaces the npz tree: the site models are simulated on that tree instead (e.g. another
//!       clade's tree, fitted under the same model -- the branch lengths are in the same units, since rate_scaling
//!       normalises every fit to a mean substitution rate of one), its leaves matched to --alignment by name; the
//!       alignment then only needs the model's site count, not the run's own sequences.
//!
//! `--rate-categories r1:w1,r2:w2,...` adds discrete across-site rate variation on top (e.g. the +G4 category table
//! of the run's .iqtree): every site of every replicate draws a category with probability w and its branch lengths are
//! multiplied by r.
//!
//! Every column is simulated independently down the tree: the root state is drawn from the site's stationary
//! distribution pi, every other node from the row of P(t) = exp(Q t) of its parent's state.
//!
//! With --alignment the sequences are written in its order and its gap pattern is copied: every cell that is not one
//! of the 20 amino acids in the original (gap, X, ?, B, Z, ...) is taken from it (unless --no-gaps). Without it the
//! sequences are written in tree order, gap-free.
//!
//! Usage:
//!   mutsel_simulate (--sitemodel <file> --tree <newick> | --npz <file>) [--alignment <file>] --out <prefix>
//!                   [--replicates 100] [--seed 1] [--mode mean|posterior] [--rate-categories r:w,...]
//!                   [--no-gaps] [--threads N]
//! writes <prefix>_1.fasta ... <prefix>_<replicates>.fasta

#![allow(non_snake_case)]

use std::{
    collections::HashMap,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use mutsel_rust::{
    Result, model,
    npz::{Array, Npz},
    optimization::Mu,
};
use nuts_rs::rand::{Rng, SeedableRng, rngs::ChaCha8Rng};
use phylo_grad::nalgebra::{SMatrix, SVector, SymmetricEigen};
use rayon::prelude::*;

/// Amino acid order of IQ-TREE, which the state indices of the model refer to
const AMINO_ACIDS: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";

fn is_amino_acid(residue: u8) -> bool {
    AMINO_ACIDS.contains(&residue.to_ascii_uppercase())
}

// ---------------------------------------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Mean,
    Posterior,
}

struct Args {
    sitemodel: Option<PathBuf>,
    tree: Option<PathBuf>,
    npz: Option<PathBuf>,
    alignment: Option<PathBuf>,
    out: String,
    replicates: usize,
    seed: u64,
    mode: Option<Mode>,
    rate_categories: Option<RateCategories>,
    keep_gaps: bool,
    threads: Option<usize>,
}

const USAGE: &str = "usage: mutsel_simulate (--sitemodel <file> --tree <newick> | --npz <run.hmc.npz> [--tree <newick>]) \
[--alignment <file>] --out <prefix> [--replicates 100] [--seed 1] [--mode mean|posterior] \
[--rate-categories r1:w1,r2:w2,...] [--no-gaps] [--threads N]";

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut parsed = Args {
        sitemodel: None,
        tree: None,
        npz: None,
        alignment: None,
        out: String::new(),
        replicates: 100,
        seed: 1,
        mode: None,
        rate_categories: None,
        keep_gaps: true,
        threads: None,
    };
    let mut out = None;
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{} needs a value\n{}", arg, USAGE));
        match arg.as_str() {
            "--sitemodel" => parsed.sitemodel = Some(PathBuf::from(value()?)),
            "--tree" => parsed.tree = Some(PathBuf::from(value()?)),
            "--npz" => parsed.npz = Some(PathBuf::from(value()?)),
            "--alignment" => parsed.alignment = Some(PathBuf::from(value()?)),
            "--out" => out = Some(value()?),
            "--replicates" => parsed.replicates = value()?.parse()?,
            "--seed" => parsed.seed = value()?.parse()?,
            "--mode" => {
                parsed.mode = Some(match value()?.as_str() {
                    "mean" => Mode::Mean,
                    "posterior" => Mode::Posterior,
                    other => return Err(format!("unknown mode {} (mean or posterior)", other).into()),
                })
            }
            "--rate-categories" => parsed.rate_categories = Some(RateCategories::parse(&value()?)?),
            "--no-gaps" => parsed.keep_gaps = false,
            "--threads" => parsed.threads = Some(value()?.parse()?),
            "-h" | "--help" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {}\n{}", other, USAGE).into()),
        }
    }
    parsed.out = out.ok_or(USAGE)?;
    match (&parsed.sitemodel, &parsed.tree, &parsed.npz) {
        (Some(_), Some(_), None) => {
            if parsed.mode.is_some() {
                return Err("--mode only applies to --npz".into());
            }
        }
        (None, tree, Some(_)) => {
            if parsed.alignment.is_none() {
                return Err(if tree.is_some() {
                    "--npz needs --alignment for the site count and the taxon order".into()
                } else {
                    "--npz needs --alignment: the npz tree has no taxon names".into()
                });
            }
        }
        _ => return Err(format!("give either --sitemodel and --tree, or --npz\n{}", USAGE).into()),
    }
    if parsed.replicates == 0 {
        return Err("--replicates must be at least 1".into());
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------------------------------------------------
// Input files
// ---------------------------------------------------------------------------------------------------------------------

/// Sequences of a FASTA or (relaxed, sequential) PHYLIP file, in file order
fn read_alignment(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Could not read {}: {}", path.display(), e))?;
    let mut sequences: Vec<(String, Vec<u8>)> = vec![];
    if text.trim_start().starts_with('>') {
        for line in text.lines() {
            let line = line.trim();
            if let Some(name) = line.strip_prefix('>') {
                sequences.push((name.split_whitespace().next().unwrap_or("").to_string(), vec![]));
            } else if let Some((_, sequence)) = sequences.last_mut() {
                sequence.extend(line.bytes().filter(|b| !b.is_ascii_whitespace()));
            }
        }
    } else {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or_else(|| format!("{} is empty", path.display()))?;
        let dims: Vec<usize> = header.split_whitespace().map(str::parse).collect::<std::result::Result<_, _>>()?;
        let [num_sequences, length] = dims[..] else {
            return Err(format!("{}: neither FASTA nor a PHYLIP header", path.display()).into());
        };
        for line in lines {
            let mut fields = line.split_whitespace();
            let name = fields.next().unwrap().to_string();
            sequences.push((name, fields.flat_map(str::bytes).collect()));
        }
        if sequences.len() != num_sequences || sequences.iter().any(|(_, s)| s.len() != length) {
            return Err(format!("{}: only sequential single-line PHYLIP is supported", path.display()).into());
        }
    }
    if sequences.is_empty() {
        return Err(format!("{} holds no sequences", path.display()).into());
    }
    if let Some((name, _)) = sequences.iter().find(|(_, s)| s.len() != sequences[0].1.len()) {
        return Err(format!("{}: sequence {} has a different length", path.display(), name).into());
    }
    Ok(sequences)
}

/// The binary site model file of IQ-TREE (write_binary_site_model_file in utils/mutsel_wrapper.cpp): the header
/// string ("UNUSED", or "DRAWS k" for k consecutive models per site), the number of models, then all frequencies
/// [models, 20] and all exchangeabilities [models, 190], little-endian f64
struct SiteModelFile {
    draws_per_site: usize,
    freqs: Vec<SVector<f64, 20>>,
    exchangeabilities: Vec<[f64; 190]>,
}

impl SiteModelFile {
    fn read(path: &Path) -> Result<SiteModelFile> {
        let mut bytes = vec![];
        std::fs::File::open(path)
            .map_err(|e| format!("Could not open {}: {}", path.display(), e))?
            .read_to_end(&mut bytes)?;
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let slice = bytes.get(pos..pos + n).ok_or_else(|| format!("{} is truncated", path.display()))?;
            pos += n;
            Ok(slice)
        };
        let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap()) as usize;
        let f64s = |b: &[u8]| b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();

        let header_len = u64_at(take(8)?);
        let header = String::from_utf8_lossy(take(header_len)?).to_string();
        let num_models = u64_at(take(8)?);
        let freqs = f64s(take(num_models * 20 * 8)?);
        let rates = f64s(take(num_models * 190 * 8)?);
        if pos != bytes.len() {
            return Err(format!("{}: {} trailing bytes after the site models", path.display(), bytes.len() - pos).into());
        }
        let draws_per_site = match header.strip_prefix("DRAWS ") {
            Some(k) => k.trim().parse()?,
            None => 1,
        };
        if draws_per_site == 0 || num_models % draws_per_site != 0 {
            return Err(format!("{}: {} models with header {:?}", path.display(), num_models, header).into());
        }
        let freqs = freqs
            .chunks_exact(20)
            .map(|pi| {
                // As IQ-TREE's reader: frequencies are renormalized
                let pi = SVector::<f64, 20>::from_column_slice(pi);
                pi / pi.sum()
            })
            .collect::<Vec<_>>();
        if let Some(i) = freqs.iter().position(|pi| pi.iter().any(|&p| !(p > 0.0))) {
            return Err(format!("{}: model {} has a non-positive frequency", path.display(), i).into());
        }
        let exchangeabilities: Vec<[f64; 190]> = rates.chunks_exact(190).map(|r| r.try_into().unwrap()).collect();
        if let Some(i) = exchangeabilities.iter().position(|r| r.iter().any(|&x| !(x > 0.0))) {
            return Err(format!("{}: model {} has a non-positive exchangeability", path.display(), i).into());
        }
        Ok(SiteModelFile { draws_per_site, freqs, exchangeabilities })
    }

    fn num_sites(&self) -> usize {
        self.freqs.len() / self.draws_per_site
    }

    /// Process of model `draw` of a site
    fn process(&self, site: usize, draw: usize) -> SiteProcess {
        let index = site * self.draws_per_site + draw;
        SiteProcess::from_exchangeabilities(&self.freqs[index], &self.exchangeabilities[index])
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Substitution process of a site
// ---------------------------------------------------------------------------------------------------------------------

/// The substitution process of one site, P(t)_ab = sqrt_pi_b / sqrt_pi_a sum_k V_ak V_bk exp(lambda_k t),
/// from the eigendecomposition of the symmetric A = D^1/2 Q D^-1/2 = S + diag(Q_aa), D = diag(pi)
struct SiteProcess {
    sqrt_pi: SVector<f64, 20>,
    eigenvectors: SMatrix<f64, 20, 20>,
    eigenvalues: SVector<f64, 20>,
}

impl SiteProcess {
    /// From the symmetric S (zero diagonal) and sqrt_pi of phylo_grad's parametrization, Q_ab = S_ab sqrt_pi_b / sqrt_pi_a
    fn new(S: &SMatrix<f64, 20, 20>, sqrt_pi: &SVector<f64, 20>) -> SiteProcess {
        let mut A = *S;
        for a in 0..20 {
            A[(a, a)] = -(0..20).filter(|&b| b != a).map(|b| S[(a, b)] * sqrt_pi[b]).sum::<f64>() / sqrt_pi[a];
        }
        let eigen = SymmetricEigen::new(A);
        SiteProcess { sqrt_pi: *sqrt_pi, eigenvectors: eigen.eigenvectors, eigenvalues: eigen.eigenvalues }
    }

    /// From IQ-TREE's parametrization, Q_ab = R_ab pi_b, with R given as its upper triangle in row-major order
    fn from_exchangeabilities(pi: &SVector<f64, 20>, R: &[f64; 190]) -> SiteProcess {
        let sqrt_pi = pi.map(f64::sqrt);
        let mut S = SMatrix::<f64, 20, 20>::zeros();
        let mut index = 0;
        for a in 0..20 {
            for b in (a + 1)..20 {
                // The inverse of model::phylograd2iqtree_parametrization
                S[(a, b)] = R[index] * sqrt_pi[a] * sqrt_pi[b];
                S[(b, a)] = S[(a, b)];
                index += 1;
            }
        }
        SiteProcess::new(&S, &sqrt_pi)
    }

    fn stationary(&self) -> SVector<f64, 20> {
        self.sqrt_pi.component_mul(&self.sqrt_pi)
    }

    /// Row a of P(t), the distribution of the state after time t starting in a
    fn transition_row(&self, a: usize, t: f64) -> SVector<f64, 20> {
        let weights = SVector::<f64, 20>::from_fn(|k, _| self.eigenvectors[(a, k)] * (self.eigenvalues[k] * t).exp());
        let row = self.eigenvectors * weights;
        SVector::from_fn(|b, _| (row[b] * self.sqrt_pi[b] / self.sqrt_pi[a]).max(0.0))
    }
}

/// Discrete across-site rate variation: relative rates and their probabilities
#[derive(Debug, Clone)]
struct RateCategories {
    rates: Vec<f64>,
    weights: Vec<f64>,
}

impl RateCategories {
    /// "r1:w1,r2:w2,..."
    fn parse(text: &str) -> Result<RateCategories> {
        let mut categories = RateCategories { rates: vec![], weights: vec![] };
        for entry in text.split(',').filter(|e| !e.trim().is_empty()) {
            let (rate, weight) = entry.split_once(':').ok_or_else(|| format!("rate category {:?} is not rate:weight", entry))?;
            let (rate, weight): (f64, f64) = (rate.trim().parse()?, weight.trim().parse()?);
            if !(rate >= 0.0) || !(weight >= 0.0) {
                return Err(format!("rate category {:?} has a negative entry", entry).into());
            }
            categories.rates.push(rate);
            categories.weights.push(weight);
        }
        if categories.weights.iter().sum::<f64>() <= 0.0 {
            return Err(format!("--rate-categories {:?} has no positive weight", text).into());
        }
        Ok(categories)
    }

    fn sample(&self, rng: &mut ChaCha8Rng) -> f64 {
        self.rates[sample(&self.weights, rng)]
    }
}

/// Draws an index from unnormalized non-negative weights
fn sample(weights: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let u = uniform(rng) * weights.iter().sum::<f64>();
    let mut cumulative = 0.0;
    for (index, &w) in weights.iter().enumerate() {
        cumulative += w;
        if u < cumulative {
            return index;
        }
    }
    // Rounding: the last index with positive weight
    weights.iter().rposition(|&w| w > 0.0).unwrap()
}

fn uniform(rng: &mut ChaCha8Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Independent stream per (seed, replicate, site), so the result does not depend on the thread schedule
fn site_rng(seed: u64, replicate: usize, site: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(((replicate as u64) << 32) | site as u64);
    rng
}

// ---------------------------------------------------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------------------------------------------------

/// Topologically sorted tree as phylo_grad uses it: leaves first, every parent after its children, the root last
struct Tree {
    parents: Vec<usize>,
    branch_lengths: Vec<f64>,
    num_leaves: usize,
    /// Leaf names, if the tree has them (Newick); the npz tree is named by the alignment
    leaf_names: Option<Vec<String>>,
}

impl Tree {
    fn from_npz(npz: &Npz) -> Result<Tree> {
        let parents = &npz.get("parents")?.data;
        let branch_lengths = npz.get("branch_lengths")?.data.clone();
        let num_nodes = parents.len();
        if branch_lengths.len() != num_nodes || parents.last() != Some(&-1.0) {
            return Err("parents and branch_lengths of the npz do not describe a rooted tree".into());
        }
        let mut is_parent = vec![false; num_nodes];
        for &p in &parents[..num_nodes - 1] {
            is_parent[p as usize] = true;
        }
        let num_leaves = is_parent.iter().take_while(|&&p| !p).count();
        let sorted = parents[..num_nodes - 1].iter().enumerate().all(|(child, &p)| p as usize > child)
            && is_parent[num_leaves..].iter().all(|&p| p);
        if !sorted {
            return Err("the tree of the npz is not topologically sorted with the leaves first".into());
        }
        let parents = parents.iter().map(|&p| if p < 0.0 { usize::MAX } else { p as usize }).collect();
        Ok(Tree { parents, branch_lengths, num_leaves, leaf_names: None })
    }

    /// Reads a Newick tree with branch lengths. Internal node labels (support values) are ignored, the root is the
    /// outermost node (any rooting gives the same distribution, the processes are reversible).
    fn from_newick(path: &Path) -> Result<Tree> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("Could not read {}: {}", path.display(), e))?;
        let s = text.trim().as_bytes();
        // name, branch length, children
        let mut nodes: Vec<(String, Option<f64>, Vec<usize>)> = vec![];
        let mut stack: Vec<usize> = vec![];
        let mut root = None;
        let mut i = 0;
        while i < s.len() && root.is_none() {
            match s[i] {
                b'(' => {
                    let id = nodes.len();
                    nodes.push((String::new(), None, vec![]));
                    if let Some(&parent) = stack.last() {
                        nodes[parent].2.push(id);
                    }
                    stack.push(id);
                    i += 1;
                }
                b',' => i += 1,
                c if c.is_ascii_whitespace() => i += 1,
                b')' => {
                    i += 1;
                    let id = stack.pop().ok_or_else(|| format!("{}: unbalanced parentheses", path.display()))?;
                    let (label, length) = newick_label(s, &mut i)?;
                    nodes[id].0 = label;
                    nodes[id].1 = length;
                    if stack.is_empty() {
                        root = Some(id);
                    }
                }
                b';' => return Err(format!("{}: ';' before the tree is closed", path.display()).into()),
                _ => {
                    let (name, length) = newick_label(s, &mut i)?;
                    let &parent = stack.last().ok_or_else(|| format!("{}: leaf outside parentheses", path.display()))?;
                    if name.is_empty() {
                        return Err(format!("{}: unnamed leaf", path.display()).into());
                    }
                    let id = nodes.len();
                    nodes.push((name, length, vec![]));
                    nodes[parent].2.push(id);
                }
            }
        }
        let root = root.ok_or_else(|| format!("{}: no complete tree", path.display()))?;

        // Leaves in order of appearance, internal nodes in post-order after them
        let mut leaves = vec![];
        let mut internal = vec![];
        let mut visit = vec![(root, false)];
        while let Some((node, expanded)) = visit.pop() {
            if nodes[node].2.is_empty() {
                leaves.push(node);
            } else if expanded {
                internal.push(node);
            } else {
                visit.push((node, true));
                visit.extend(nodes[node].2.iter().rev().map(|&child| (child, false)));
            }
        }
        let num_leaves = leaves.len();
        let order: Vec<usize> = leaves.iter().chain(&internal).copied().collect();
        let mut new_id = vec![0; nodes.len()];
        for (id, &node) in order.iter().enumerate() {
            new_id[node] = id;
        }
        let mut parents = vec![usize::MAX; order.len()];
        let mut branch_lengths = vec![0.0; order.len()];
        for &node in &order {
            for &child in &nodes[node].2 {
                parents[new_id[child]] = new_id[node];
                branch_lengths[new_id[child]] = nodes[child]
                    .1
                    .ok_or_else(|| format!("{}: the tree needs branch lengths ({})", path.display(), nodes[child].0))?;
            }
        }
        let leaf_names: Vec<String> = leaves.iter().map(|&leaf| nodes[leaf].0.clone()).collect();
        let mut seen = std::collections::HashSet::new();
        if let Some(name) = leaf_names.iter().find(|name| !seen.insert(*name)) {
            return Err(format!("{}: leaf {} appears twice", path.display(), name).into());
        }
        Ok(Tree { parents, branch_lengths, num_leaves, leaf_names: Some(leaf_names) })
    }

    fn root(&self) -> usize {
        self.parents.len() - 1
    }

    fn length(&self) -> f64 {
        self.branch_lengths.iter().sum()
    }

    /// States of all nodes of one column, the root drawn from the stationary distribution and every branch length
    /// multiplied by `rate`
    fn simulate(&self, process: &SiteProcess, rate: f64, rng: &mut ChaCha8Rng, states: &mut [usize]) {
        let root = self.root();
        states[root] = sample(process.stationary().as_slice(), rng);
        // Parents have higher indices than their children, so the reverse order is a pre-order
        for node in (0..root).rev() {
            let row = process.transition_row(states[self.parents[node]], rate * self.branch_lengths[node]);
            states[node] = sample(row.as_slice(), rng);
        }
    }
}

/// A Newick label (optionally single-quoted) and its optional ":length"
fn newick_label(s: &[u8], i: &mut usize) -> Result<(String, Option<f64>)> {
    let is_end = |c: u8| matches!(c, b':' | b',' | b')' | b'(' | b';') || c.is_ascii_whitespace();
    let label = if s.get(*i) == Some(&b'\'') {
        let start = *i + 1;
        let end = start + s[start..].iter().position(|&c| c == b'\'').ok_or("unterminated quote in the Newick tree")?;
        *i = end + 1;
        String::from_utf8_lossy(&s[start..end]).to_string()
    } else {
        let start = *i;
        while *i < s.len() && !is_end(s[*i]) {
            *i += 1;
        }
        String::from_utf8_lossy(&s[start..*i]).to_string()
    };
    while *i < s.len() && s[*i].is_ascii_whitespace() {
        *i += 1;
    }
    let mut length = None;
    if s.get(*i) == Some(&b':') {
        *i += 1;
        let start = *i;
        while *i < s.len() && !is_end(s[*i]) {
            *i += 1;
        }
        let text = std::str::from_utf8(&s[start..*i])?;
        length = Some(text.parse::<f64>().map_err(|_| format!("branch length {:?} in the Newick tree", text))?);
    }
    Ok((label, length))
}

// ---------------------------------------------------------------------------------------------------------------------
// Site models
// ---------------------------------------------------------------------------------------------------------------------

/// Site frequencies and site rates of an hmc npz: the posterior means, and the posterior samples for Mode::Posterior
struct NpzSiteModels {
    scaled_Mu: SMatrix<f64, 20, 20>,
    mean_pi: Vec<SVector<f64, 20>>,
    mean_site_rate: Vec<f64>,
    /// [num_sites][num_samples] posterior samples
    pi_samples: Vec<Vec<SVector<f64, 20>>>,
    log_site_rates: Option<Vec<Vec<f64>>>,
}

impl NpzSiteModels {
    fn from_npz(npz: &Npz, mode: Mode) -> Result<NpzSiteModels> {
        let log_S = &npz.get("log_exchangeabilities")?.data;
        let rate_scaling = npz.get("rate_scaling")?.to_scalar()?;
        let mean_pi = npz.get("mean_pi")?.rows::<20>()?;
        let num_sites = mean_pi.len();
        let mean_site_rate = match npz.get_opt("mean_site_rate") {
            Some(rates) => rates.data.clone(),
            None => vec![1.0; num_sites],
        };
        let (mut pi_samples, mut log_site_rates) = (vec![], None);
        if mode == Mode::Posterior {
            let samples = npz.get("pi_samples")?;
            let per_site = samples.shape[1];
            pi_samples = samples.rows::<20>()?.chunks_exact(per_site).map(<[_]>::to_vec).collect();
            log_site_rates = npz
                .get_opt("log_site_rates")
                .map(|rates: &Array| rates.data.chunks_exact(rates.shape[1]).map(<[_]>::to_vec).collect());
        }
        Ok(NpzSiteModels { scaled_Mu: Mu(log_S) * rate_scaling, mean_pi, mean_site_rate, pi_samples, log_site_rates })
    }

    fn process(&self, pi: &SVector<f64, 20>, site_rate: f64) -> SiteProcess {
        // rate_matrix only needs log_pi up to a constant, as in hmc::rate_matrices
        let (S, sqrt_pi) = model::rate_matrix(&self.scaled_Mu, &pi.map(f64::ln));
        SiteProcess::new(&(S * site_rate), &sqrt_pi)
    }

    /// The site model of a random posterior sample of the site
    fn posterior_process(&self, site: usize, rng: &mut ChaCha8Rng) -> SiteProcess {
        let samples = &self.pi_samples[site];
        let k = (rng.next_u64() % samples.len() as u64) as usize;
        let rate = self.log_site_rates.as_ref().map_or(1.0, |rates| rates[site][k].exp());
        self.process(&samples[k], rate)
    }
}

/// Where the process of a site in a replicate comes from
enum SiteModels {
    /// The same process in every replicate (precomputed)
    Fixed(Vec<SiteProcess>),
    /// A .sitemodel with several draws per site: one drawn per site and replicate
    Draws(SiteModelFile),
    /// Posterior samples of an npz: one drawn per site and replicate
    Posterior(NpzSiteModels),
}

impl SiteModels {
    fn num_sites(&self) -> usize {
        match self {
            SiteModels::Fixed(processes) => processes.len(),
            SiteModels::Draws(file) => file.num_sites(),
            SiteModels::Posterior(models) => models.mean_pi.len(),
        }
    }

    /// Calls f with the process of `site` (drawing it first if it is random)
    fn with_process<T>(&self, site: usize, rng: &mut ChaCha8Rng, f: impl FnOnce(&SiteProcess, &mut ChaCha8Rng) -> T) -> T {
        match self {
            SiteModels::Fixed(processes) => f(&processes[site], rng),
            SiteModels::Draws(file) => {
                let draw = (rng.next_u64() % file.draws_per_site as u64) as usize;
                f(&file.process(site, draw), rng)
            }
            SiteModels::Posterior(models) => {
                let process = models.posterior_process(site, rng);
                f(&process, rng)
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------

fn write_fasta(path: &str, names: &[String], sequences: &[Vec<u8>]) -> Result<()> {
    let file = std::fs::File::create(path).map_err(|e| format!("Could not create {}: {}", path, e))?;
    let mut writer = BufWriter::new(file);
    for (name, sequence) in names.iter().zip(sequences) {
        writeln!(writer, ">{}", name)?;
        writer.write_all(sequence)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }

    let (tree, models, source) = if let Some(npz_path) = &args.npz {
        let npz = Npz::read(npz_path)?;
        let tree = match &args.tree {
            Some(path) => Tree::from_newick(path)?,
            None => Tree::from_npz(&npz)?,
        };
        let mode = args.mode.unwrap_or(Mode::Mean);
        let npz_models = NpzSiteModels::from_npz(&npz, mode)?;
        let models = match mode {
            Mode::Mean => SiteModels::Fixed(
                (0..npz_models.mean_pi.len())
                    .into_par_iter()
                    .map(|site| npz_models.process(&npz_models.mean_pi[site], npz_models.mean_site_rate[site]))
                    .collect(),
            ),
            Mode::Posterior => SiteModels::Posterior(npz_models),
        };
        let source = match mode {
            Mode::Mean => "npz, posterior mean site models",
            Mode::Posterior => "npz, posterior draw per site",
        };
        (tree, models, source.to_string())
    } else {
        let tree = Tree::from_newick(args.tree.as_ref().unwrap())?;
        let file = SiteModelFile::read(args.sitemodel.as_ref().unwrap())?;
        let source = format!("sitemodel, {} model(s) per site", file.draws_per_site);
        let models = if file.draws_per_site == 1 {
            SiteModels::Fixed((0..file.num_sites()).into_par_iter().map(|site| file.process(site, 0)).collect())
        } else {
            SiteModels::Draws(file)
        };
        (tree, models, source)
    };
    let num_sites = models.num_sites();

    // Output rows: the leaf each one is simulated at, its name and, for the gap pattern, its original sequence
    let alignment = args.alignment.as_deref().map(read_alignment).transpose()?;
    let (row_leaf, names, original): (Vec<usize>, Vec<String>, Option<Vec<Vec<u8>>>) = match (&alignment, &tree.leaf_names) {
        (Some(alignment), leaf_names) => {
            let aln = args.alignment.as_ref().unwrap().display();
            if alignment[0].1.len() != num_sites {
                return Err(format!("{} has {} columns, the model {} sites", aln, alignment[0].1.len(), num_sites).into());
            }
            if alignment.len() != tree.num_leaves {
                return Err(format!("{} has {} sequences, the tree {} leaves", aln, alignment.len(), tree.num_leaves).into());
            }
            let row_leaf = match leaf_names {
                // npz: leaf i is sequence i of the alignment the run was given
                None => (0..tree.num_leaves).collect(),
                Some(leaf_names) => {
                    let leaf_of: HashMap<&str, usize> = leaf_names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
                    alignment
                        .iter()
                        .map(|(name, _)| {
                            leaf_of.get(name.as_str()).copied().ok_or_else(|| format!("{} of {} is not in the tree", name, aln))
                        })
                        .collect::<std::result::Result<_, _>>()?
                }
            };
            let names = alignment.iter().map(|(name, _)| name.clone()).collect();
            let original = alignment.iter().map(|(_, s)| s.to_ascii_uppercase()).collect();
            (row_leaf, names, args.keep_gaps.then_some(original))
        }
        (None, Some(leaf_names)) => ((0..tree.num_leaves).collect(), leaf_names.clone(), None),
        (None, None) => unreachable!("--npz requires --alignment"),
    };

    println!(
        "mutsel_simulate: {} taxa, {} sites, tree length {:.4}, {}{}, {} replicates, seed {}, gaps {}",
        tree.num_leaves,
        num_sites,
        tree.length(),
        source,
        match &args.rate_categories {
            Some(c) => format!(", {} rate categories", c.rates.len()),
            None => String::new(),
        },
        args.replicates,
        args.seed,
        if original.is_some() { "copied from the alignment" } else { "none" }
    );

    if let Some(dir) = Path::new(&args.out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    for replicate in 1..=args.replicates {
        // columns[site][row]
        let columns: Vec<Vec<u8>> = (0..num_sites)
            .into_par_iter()
            .map(|site| {
                let mut rng = site_rng(args.seed, replicate, site);
                let mut states = vec![0usize; tree.parents.len()];
                models.with_process(site, &mut rng, |process, rng| {
                    let rate = args.rate_categories.as_ref().map_or(1.0, |c| c.sample(rng));
                    tree.simulate(process, rate, rng, &mut states);
                });
                row_leaf
                    .iter()
                    .enumerate()
                    .map(|(row, &leaf)| match &original {
                        Some(original) if !is_amino_acid(original[row][site]) => original[row][site],
                        _ => AMINO_ACIDS[states[leaf]],
                    })
                    .collect()
            })
            .collect();
        let sequences: Vec<Vec<u8>> =
            (0..row_leaf.len()).map(|row| columns.iter().map(|column| column[row]).collect()).collect();
        write_fasta(&format!("{}_{}.fasta", args.out, replicate), &names, &sequences)?;
    }
    println!("mutsel_simulate: wrote {}_1.fasta ... {}_{}.fasta", args.out, args.out, args.replicates);
    Ok(())
}
