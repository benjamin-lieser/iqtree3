#![allow(non_snake_case)]

pub mod codon;
pub mod data;
pub mod felsenstein;
pub mod felsenstein_codon;
mod io;
pub mod model;
mod pca;
pub mod optimization;
pub mod optimization_codon;
mod utils;

use std::{
    io::Write,
    mem::MaybeUninit,
    path::Path,
};

use candle_core::IndexOp;

// These two functions are exported, nothing else.

#[unsafe(no_mangle)]
pub extern "C" fn rust_set_rayon_threads(num_threads: u32) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build_global()
        .unwrap_or_else(|_| println!("Rust Threadnumber has already been set. Ignoring."));
}

/// parents: [num_nodes]
/// branch_lengths: [num_nodes]
/// alignment: [num_sites * num_leaves] (row-major)
/// out_site_freq: [num_sites * 20] (row-major)
/// out_rate_matrix: [num_sites * 190] (row-major)
/// out_rate_para: [num_para] from the rate model 1 for gamma, 2 * num_cat for free rate
/// out variables do not need to be initialized
/// prior_R_file and prior_pi_file can be null pointers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_mutsel(
    parents: *const i32,
    branch_lengths: *const f64,
    alignment: *const u8,
    num_sites: u32,
    num_leaves: u32,
    num_nodes: u32,
    model_str: *const std::os::raw::c_char,
    prior_R_file: *const std::os::raw::c_char,
    _prior_pi_file: *const std::os::raw::c_char,
    verbose: u8,
    out_site_freq: *mut f64,
    out_rate_matrix: *mut f64,
    output_prefix: *const std::os::raw::c_char,
) {
    let out_prefix = unsafe { std::ffi::CStr::from_ptr(output_prefix) }
        .to_str()
        .unwrap();
    let (saved_stdout, saved_stderr, tee_handle) = io::tee_to_file(&format!("{}.log", out_prefix));

    let parents = unsafe { std::slice::from_raw_parts(parents, num_nodes as usize) };
    let branch_lengths = unsafe { std::slice::from_raw_parts(branch_lengths, num_nodes as usize) };
    let alignment =
        unsafe { std::slice::from_raw_parts(alignment, (num_sites * num_leaves) as usize) };

    let out_site_freq = unsafe {
        std::slice::from_raw_parts_mut(
            out_site_freq as *mut MaybeUninit<f64>,
            (num_sites * 20) as usize,
        )
    };

    let out_rate_matrix = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_matrix as *mut MaybeUninit<f64>,
            (num_sites * 190) as usize,
        )
    };

    let model_cstr = unsafe { std::ffi::CStr::from_ptr(model_str) };
    let model_str = model_cstr.to_str().unwrap();

    println!("Starting mutsel optimization with {}", model_str);

    let mutsel_params = parse_mutsel_str(model_str);

    let felsenstein = create_felsenstein_tree(
        parents,
        branch_lengths,
        alignment,
        num_sites as usize,
        num_leaves as usize,
    );

    let prior_R_file = if prior_R_file.is_null() {
        None
    } else {
        let cstr = unsafe { std::ffi::CStr::from_ptr(prior_R_file) };
        Some(Path::new(cstr.to_str().unwrap()))
    };

    let (S, sqrt_pi) = optimization::optimize_internal(
        felsenstein,
        branch_lengths,
        mutsel_params,
        prior_R_file,
        crate::Verbosity::from_u8(verbose),
        out_prefix,
    )
    .unwrap();

    let (R, pi) = model::phylograd2iqtree_parametrization(&S, &sqrt_pi).unwrap();
    let site_freq = pi.to_vec2().unwrap();

    for site_index in 0..num_sites as usize {
        for aa_index in 0..20 {
            let value = site_freq[site_index][aa_index];
            out_site_freq[site_index * 20 + aa_index].write(value);
        }
    }

    for site_index in 0..num_sites as usize {
        let mut idx = 0;
        // Upper diagonal
        for i in 0..20 {
            for j in (i + 1)..20 {
                out_rate_matrix[site_index * 190 + idx]
                    .write(R.i((site_index, i, j)).unwrap().to_scalar().unwrap());
                idx += 1;
            }
        }
    }
    std::io::stdout().flush().unwrap();
    io::restore_stdout_stderr(saved_stdout, saved_stderr, tee_handle);
}

#[derive(Debug, Clone, Copy)]
pub struct MutselParams {
    pi_reg: f64,
    Mu_reg: f64,
    site_rate_reg: f64,
    branch_length_reg: f64,
}

fn parse_mutsel_str(model_str: &str) -> MutselParams {
    let model_str = model_str.trim();
    let model_upper = model_str.to_ascii_uppercase();

    let (pi_reg, Mu_reg, site_rate_reg, branch_length_reg) = if model_upper == "MUTSEL" {
        (0.32, 9.67, 2.0, 10.0)
    } else if model_upper.starts_with("MUTSEL{") && model_str.ends_with('}') {
        let params_str = &model_str[7..model_str.len() - 1];
        let values = params_str
            .split('/')
            .map(|value| value.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert!(
            values.len() == 4,
            "Invalid MUTSEL format: expected MUTSEL{{pi_reg/Mu_reg/site_rate_reg/branch_length_reg}}, got {}",
            model_str
        );
        (values[0], values[1], values[2], values[3])
    } else {
        panic!(
            "Invalid MUTSEL format: expected MUTSEL or MUTSEL{{pi_reg/Mu_reg/site_rate_reg/branch_length_reg}}, got {}",
            model_str
        );
    };

    MutselParams {
        pi_reg,
        Mu_reg,
        site_rate_reg,
        branch_length_reg,
    }
}

fn create_felsenstein_tree(
    parents: &[i32],
    distances: &[f64],
    alignment: &[u8],
    L : usize,
    N : usize,
) -> phylo_grad::FelsensteinTree<20> {
    let mut felsenstein = phylo_grad::FelsensteinTree::<20>::new(parents, distances);

    let mut sites = vec![];
    sites.resize(L, vec![]);
    for site_idx in 0..L {
        sites[site_idx].resize(N, phylo_grad::nalgebra::SVector::<f64, 20>::zeros());
        for seq_idx in 0..N {
            let residue = alignment[site_idx * N + seq_idx];
            if residue < 20 {
                sites[site_idx][seq_idx][residue as usize] = 1.0;
            } else {
                for i in 0..20 {
                    sites[site_idx][seq_idx][i] = 1.0;
                }
            }
        }
    }
    felsenstein.bind_leaf_pl(sites);
    felsenstein
}

/// Builds the real 61-state codon Felsenstein tree from dense codon-state
/// leaf data (states 0..60, or >=61 for gap/ambiguous -> uniform).
fn create_felsenstein_tree_codon(
    parents: &[i32],
    distances: &[f64],
    alignment: &[u8],
    L: usize,
    N: usize,
) -> phylo_grad::FelsensteinTree<{ codon::N_CODON }> {
    let mut felsenstein = phylo_grad::FelsensteinTree::<{ codon::N_CODON }>::new(parents, distances);

    let mut sites = vec![];
    sites.resize(L, vec![]);
    for site_idx in 0..L {
        sites[site_idx].resize(N, phylo_grad::nalgebra::SVector::<f64, { codon::N_CODON }>::zeros());
        for seq_idx in 0..N {
            let residue = alignment[site_idx * N + seq_idx] as usize;
            if residue < codon::N_CODON {
                sites[site_idx][seq_idx][residue] = 1.0;
            } else {
                for i in 0..codon::N_CODON {
                    sites[site_idx][seq_idx][i] = 1.0;
                }
            }
        }
    }
    felsenstein.bind_leaf_pl(sites);
    felsenstein
}

/// Builds an amino-acid (20-state) view of a codon alignment: each leaf
/// codon state is translated to its encoded amino acid through
/// `codon_aa` (gap/ambiguous codon states, or an ambiguous amino acid,
/// -> uniform over the 20 amino acids). Used only to run the amino-acid
/// model's light-PMSF initialization phase unchanged on codon data.
fn create_aa_view_of_codon_alignment(
    parents: &[i32],
    distances: &[f64],
    alignment: &[u8],
    L: usize,
    N: usize,
    codon_aa: &[u8; codon::N_CODON],
) -> phylo_grad::FelsensteinTree<20> {
    let mut felsenstein = phylo_grad::FelsensteinTree::<20>::new(parents, distances);

    let mut sites = vec![];
    sites.resize(L, vec![]);
    for site_idx in 0..L {
        sites[site_idx].resize(N, phylo_grad::nalgebra::SVector::<f64, 20>::zeros());
        for seq_idx in 0..N {
            let residue = alignment[site_idx * N + seq_idx] as usize;
            let aa = if residue < codon::N_CODON {
                Some(codon_aa[residue] as usize)
            } else {
                None
            };
            match aa {
                Some(aa) if aa < 20 => sites[site_idx][seq_idx][aa] = 1.0,
                _ => {
                    for i in 0..20 {
                        sites[site_idx][seq_idx][i] = 1.0;
                    }
                }
            }
        }
    }
    felsenstein.bind_leaf_pl(sites);
    felsenstein
}

/// Number of upper-triangle rate entries for the 61-state codon model.
const N_CODON_RATE_ENTRIES: usize = codon::N_CODON * (codon::N_CODON - 1) / 2;

/// Codon-level (61-state) counterpart of [`rust_mutsel`]. The mutation
/// process is a single 4-state nucleotide GTR shared across all 3 codon
/// positions (see `codon::Mu4`/`codon::expand_nt_to_codon_mu`); the fitness
/// landscape is exactly the amino-acid model's, broadcast to codons through
/// the genetic code. Only the standard genetic code (61 sense codons) is
/// supported.
///
/// alignment: [num_sites * num_leaves] dense codon states (0..60, or >=61
///   for gap/ambiguous), in the same dense non-stop-codon numbering as
///   `codon_nt`/`codon_aa`.
/// codon_nt: [61*3] nucleotide (0-3) at each of the 3 codon positions, for
///   each of the 61 non-stop codons (dense state order).
/// codon_aa: [61] amino-acid state (0-19) encoded by each of the 61
///   non-stop codons (dense state order).
/// out_site_freq: [num_sites * 61]
/// out_rate_matrix: [num_sites * 1830] (1830 = 61*60/2)
/// out variables do not need to be initialized
/// prior_R_file and prior_pi_file can be null pointers (not yet used for
/// the codon model: v1 always starts from a uniform 4-state GTR)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_mutsel_codon(
    parents: *const i32,
    branch_lengths: *const f64,
    alignment: *const u8,
    num_sites: u32,
    num_leaves: u32,
    num_nodes: u32,
    codon_nt: *const u8,
    codon_aa: *const u8,
    model_str: *const std::os::raw::c_char,
    _prior_R_file: *const std::os::raw::c_char,
    _prior_pi_file: *const std::os::raw::c_char,
    verbose: u8,
    out_site_freq: *mut f64,
    out_rate_matrix: *mut f64,
    output_prefix: *const std::os::raw::c_char,
) {
    let out_prefix = unsafe { std::ffi::CStr::from_ptr(output_prefix) }
        .to_str()
        .unwrap();
    let (saved_stdout, saved_stderr, tee_handle) = io::tee_to_file(&format!("{}.log", out_prefix));

    let parents = unsafe { std::slice::from_raw_parts(parents, num_nodes as usize) };
    let branch_lengths = unsafe { std::slice::from_raw_parts(branch_lengths, num_nodes as usize) };
    let alignment =
        unsafe { std::slice::from_raw_parts(alignment, (num_sites * num_leaves) as usize) };

    let codon_nt_slice = unsafe { std::slice::from_raw_parts(codon_nt, codon::N_CODON * 3) };
    let codon_aa_slice = unsafe { std::slice::from_raw_parts(codon_aa, codon::N_CODON) };
    let mut codon_nt_table = [[0u8; 3]; codon::N_CODON];
    for i in 0..codon::N_CODON {
        codon_nt_table[i] = [
            codon_nt_slice[i * 3],
            codon_nt_slice[i * 3 + 1],
            codon_nt_slice[i * 3 + 2],
        ];
    }
    let mut codon_aa_table = [0u8; codon::N_CODON];
    codon_aa_table.copy_from_slice(codon_aa_slice);

    let out_site_freq = unsafe {
        std::slice::from_raw_parts_mut(
            out_site_freq as *mut MaybeUninit<f64>,
            (num_sites as usize) * codon::N_CODON,
        )
    };

    let out_rate_matrix = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_matrix as *mut MaybeUninit<f64>,
            (num_sites as usize) * N_CODON_RATE_ENTRIES,
        )
    };

    let model_cstr = unsafe { std::ffi::CStr::from_ptr(model_str) };
    let model_str = model_cstr.to_str().unwrap();

    println!("Starting mutsel codon optimization with {}", model_str);

    let mutsel_params = parse_mutsel_str(model_str);

    let felsenstein_codon = create_felsenstein_tree_codon(
        parents,
        branch_lengths,
        alignment,
        num_sites as usize,
        num_leaves as usize,
    );
    let felsenstein_aa = create_aa_view_of_codon_alignment(
        parents,
        branch_lengths,
        alignment,
        num_sites as usize,
        num_leaves as usize,
        &codon_aa_table,
    );

    let (S, sqrt_pi) = optimization_codon::optimize_internal_codon(
        felsenstein_aa,
        felsenstein_codon,
        branch_lengths,
        mutsel_params,
        codon_nt_table,
        codon_aa_table,
        crate::Verbosity::from_u8(verbose),
        out_prefix,
    )
    .unwrap();

    let (R, pi) = model::phylograd2iqtree_parametrization(&S, &sqrt_pi).unwrap();
    let site_freq = pi.to_vec2().unwrap();

    for site_index in 0..num_sites as usize {
        for state in 0..codon::N_CODON {
            let value = site_freq[site_index][state];
            out_site_freq[site_index * codon::N_CODON + state].write(value);
        }
    }

    for site_index in 0..num_sites as usize {
        let mut idx = 0;
        // Upper diagonal
        for i in 0..codon::N_CODON {
            for j in (i + 1)..codon::N_CODON {
                out_rate_matrix[site_index * N_CODON_RATE_ENTRIES + idx]
                    .write(R.i((site_index, i, j)).unwrap().to_scalar().unwrap());
                idx += 1;
            }
        }
    }
    std::io::stdout().flush().unwrap();
    io::restore_stdout_stderr(saved_stdout, saved_stderr, tee_handle);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verbosity {
    Quiet,
    Min,
    Med,
    Max,
    Debug,
}

impl Verbosity {
    fn from_u8(value: u8) -> Verbosity {
        match value {
            0 => Verbosity::Quiet,
            1 => Verbosity::Min,
            2 => Verbosity::Med,
            3 => Verbosity::Max,
            4 => Verbosity::Debug,
            _ => panic!("Invalid verbosity level"),
        }
    }

    /// True if the current verbosity is at least `level`, i.e. -v (Med) also
    /// enables everything gated at Min. Quiet never prints.
    fn should_print(&self, level: Verbosity) -> bool {
        *self != Verbosity::Quiet && *self >= level
    }
}