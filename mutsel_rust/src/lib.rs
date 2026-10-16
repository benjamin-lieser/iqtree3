#![allow(non_snake_case)]

pub mod data;
pub mod felsenstein;
mod io;
pub mod model;
pub mod optimization;
pub mod svi;
mod utils;

use std::{
    io::Write,
    mem::MaybeUninit,
    path::Path,
};


// These two functions are exported, nothing else.

/// Stack size of the rayon workers and the thread running the inference. Called from C++, the Rust
/// runtime does not install its stack overflow handler, so an overflow is a plain SIGSEGV.
/// Stacks are only reserved address space. RUST_MIN_STACK can raise it.
fn thread_stack_size() -> usize {
    const DEFAULT: usize = 256 << 20;
    std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map_or(DEFAULT, |s| s.max(DEFAULT))
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_set_rayon_threads(num_threads: u32) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .stack_size(thread_stack_size())
        .build_global()
        .unwrap_or_else(|_| println!("Rust Threadnumber has already been set. Ignoring."));
}

/// parents: [num_nodes]
/// branch_lengths: [num_nodes]
/// alignment: [num_sites * num_leaves] (row-major)
/// num_samples: site models per site (posterior samples, whose likelihoods IQ-TREE averages), 1 for a single model
/// out_site_freq: [num_sites * num_samples * 20] (row-major), the samples of a site consecutive
/// out_rate_matrix: [num_sites * num_samples * 190] (row-major)
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
    num_samples: u32,
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
        unsafe { std::slice::from_raw_parts(alignment, num_sites as usize * num_leaves as usize) };

    let out_site_freq = unsafe {
        std::slice::from_raw_parts_mut(
            out_site_freq as *mut MaybeUninit<f64>,
            num_sites as usize * num_samples as usize * 20,
        )
    };

    let out_rate_matrix = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_matrix as *mut MaybeUninit<f64>,
            num_sites as usize * num_samples as usize * 190,
        )
    };

    let model_cstr = unsafe { std::ffi::CStr::from_ptr(model_str) };
    let model_str = model_cstr.to_str().unwrap();

    println!("Starting mutsel optimization with {}", model_str);

    let mutsel_params = parse_mutsel_str(model_str);

    let prior_R_file = if prior_R_file.is_null() {
        None
    } else {
        let cstr = unsafe { std::ffi::CStr::from_ptr(prior_R_file) };
        Some(Path::new(cstr.to_str().unwrap()))
    };

    // Variational inference of the site frequencies under the Gaussian prior, jointly with Mu and the
    // branch lengths. On a thread of our own, the stack of the C++ main thread may be small.
    let (S, sqrt_pi) = std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("mutsel".to_string())
            .stack_size(thread_stack_size())
            .spawn_scoped(scope, || {
                svi::infer_internal(
                    parents,
                    num_samples as usize,
                    branch_lengths,
                    alignment,
                    num_sites as usize,
                    num_leaves as usize,
                    mutsel_params,
                    prior_R_file,
                    crate::Verbosity::from_u8(verbose),
                    out_prefix,
                )
            })
            .expect("failed to spawn the mutsel inference thread")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
    .unwrap();

    let (R, pi) = model::phylograd2iqtree_parametrization(&S, &sqrt_pi).unwrap();
    let site_freq = pi.to_vec2().unwrap();
    let R = R.to_vec3::<f64>().unwrap();

    // Site models [num_sites * num_samples], the samples of a site consecutive
    let num_models = num_sites as usize * num_samples as usize;
    assert_eq!(site_freq.len(), num_models);
    for model_index in 0..num_models {
        for aa_index in 0..20 {
            out_site_freq[model_index * 20 + aa_index].write(site_freq[model_index][aa_index]);
        }
        let mut idx = 0;
        // Upper diagonal
        for i in 0..20 {
            for j in (i + 1)..20 {
                out_rate_matrix[model_index * 190 + idx].write(R[model_index][i][j]);
                idx += 1;
            }
        }
    }
    std::io::stdout().flush().unwrap();
    io::restore_stdout_stderr(saved_stdout, saved_stderr, tee_handle);
}

#[derive(Debug, Clone, Copy)]
pub struct MutselParams {
    pub Mu_reg: f64,
}

/// MUTSEL, MUTSEL{Mu_reg}, or the old MUTSEL{pi_reg/Mu_reg} whose pi_reg is ignored now that the
/// prior on the site frequencies is learned
fn parse_mutsel_str(model_str: &str) -> MutselParams {
    let model_str = model_str.trim();
    let model_upper = model_str.to_ascii_uppercase();

    let Mu_reg = if model_upper == "MUTSEL" {
        9.67
    } else if model_upper.starts_with("MUTSEL{") && model_str.ends_with('}') {
        let params_str = &model_str[7..model_str.len() - 1];
        let values = params_str
            .split('/')
            .map(|value| value.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        match values.len() {
            1 => values[0],
            2 => {
                println!("MUTSEL: pi_reg {} is ignored, the prior on the site frequencies is a trained Gaussian", values[0]);
                values[1]
            }
            _ => panic!("Invalid MUTSEL format: expected MUTSEL{{Mu_reg}}, got {}", model_str),
        }
    } else {
        panic!(
            "Invalid MUTSEL format: expected MUTSEL or MUTSEL{{Mu_reg}}, got {}",
            model_str
        );
    };

    MutselParams { Mu_reg }
}

pub fn create_felsenstein_tree(
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