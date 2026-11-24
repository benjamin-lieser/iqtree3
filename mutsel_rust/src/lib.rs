#![allow(non_snake_case)]

mod batch;
mod data;
mod gen_normal;
mod io;
mod lbfgs;
mod model;
mod npz;
mod optimization;
mod pca;
mod prior;

use std::{io::Write, mem::MaybeUninit};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

// These two functions are exported, nothing else.

#[unsafe(no_mangle)]
pub extern "C" fn rust_set_rayon_threads(num_threads: u32) {
    // Workers run stolen jobs on top of their own stack while waiting in a join, so phylo_grad's per-site
    // frames can nest deeply over all sites; the default 2 MB overflowed on 70k sites with 64 threads.
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .stack_size(256 << 20)
        .build_global()
        .unwrap_or_else(|_| println!("Rust Threadnumber has already been set. Ignoring."));
}

/// parents: [num_nodes]
/// branch_lengths: [num_nodes]
/// alignment: [num_sites * num_leaves] (row-major)
/// model_str: MUTSEL{pi_reg/mu_reg/rate_reg}, the generalized-normal model (see gen_normal.rs)
/// num_tree_draws: the number of models per site, all the MAP site model with weight 1 / num_tree_draws
/// out_site_freq: [num_sites * num_tree_draws * 20] (row-major), the site frequencies of the models
/// out_rate_matrix: [num_sites * num_tree_draws * 190] (row-major), the upper triangle of their rate matrices
/// seed: unused (the MAP estimate is deterministic)
/// out_draw_weights: [num_sites * num_tree_draws], the weights of the draws of every site (summing to 1)
/// out variables do not need to be initialized
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_mutsel(
    parents: *const i32,
    branch_lengths: *const f64,
    alignment: *const u8,
    num_sites: u32,
    num_leaves: u32,
    num_nodes: u32,
    model_str: *const std::os::raw::c_char,
    verbose: u8,
    num_tree_draws: u32,
    seed: u64,
    out_site_freq: *mut f64,
    out_rate_matrix: *mut f64,
    out_draw_weights: *mut f64,
    output_prefix: *const std::os::raw::c_char,
) {
    let out_prefix = unsafe { std::ffi::CStr::from_ptr(output_prefix) }
        .to_str()
        .unwrap();
    let (saved_stdout, saved_stderr, tee_handle) = io::tee_to_file(&format!("{}.log", out_prefix));

    let parents = unsafe { std::slice::from_raw_parts(parents, num_nodes as usize) };
    // phylo_grad only checks this in debug builds
    assert!(
        parents.last() == Some(&-1)
            && parents[..parents.len() - 1]
                .iter()
                .enumerate()
                .all(|(child, &parent)| parent as usize > child && parent as u32 >= num_leaves && (parent as u32) < num_nodes),
        "parents are not topologically sorted with the leaves first"
    );
    let branch_lengths = unsafe { std::slice::from_raw_parts(branch_lengths, num_nodes as usize) };
    let alignment =
        unsafe { std::slice::from_raw_parts(alignment, (num_sites * num_leaves) as usize) };
    assert!(num_tree_draws >= 1);
    let num_models = (num_sites * num_tree_draws) as usize;

    let out_site_freq = unsafe {
        std::slice::from_raw_parts_mut(
            out_site_freq as *mut MaybeUninit<f64>,
            num_models * 20,
        )
    };

    let out_rate_matrix = unsafe {
        std::slice::from_raw_parts_mut(
            out_rate_matrix as *mut MaybeUninit<f64>,
            num_models * 190,
        )
    };

    let out_draw_weights =
        unsafe { std::slice::from_raw_parts_mut(out_draw_weights as *mut MaybeUninit<f64>, num_models) };

    let model_cstr = unsafe { std::ffi::CStr::from_ptr(model_str) };
    let model_str = model_cstr.to_str().unwrap();

    // The generalized-normal model, the strengths of the priors are set in the model string
    let parameters = gen_normal::Parameters::parse(model_str).unwrap_or_else(|e| panic!("{}", e));
    let _ = seed;
    println!("Starting mutsel optimization");

    let (S, sqrt_pi, weights) = gen_normal::run(
        parents,
        branch_lengths,
        alignment,
        num_sites as usize,
        num_leaves as usize,
        crate::Verbosity::from_u8(verbose),
        out_prefix,
        parameters,
        num_tree_draws as usize,
    )
    .unwrap();

    assert_eq!(S.len(), num_models);
    assert_eq!(weights.len(), num_models);
    for (out, weight) in out_draw_weights.iter_mut().zip(&weights) {
        out.write(*weight);
    }
    for (site_index, (S, sqrt_pi)) in S.iter().zip(&sqrt_pi).enumerate() {
        let (R, pi) = model::phylograd2iqtree_parametrization(S, sqrt_pi);
        for aa_index in 0..20 {
            out_site_freq[site_index * 20 + aa_index].write(pi[aa_index]);
        }
        let mut idx = 0;
        // Upper diagonal
        for i in 0..20 {
            for j in (i + 1)..20 {
                out_rate_matrix[site_index * 190 + idx].write(R[(i, j)]);
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
