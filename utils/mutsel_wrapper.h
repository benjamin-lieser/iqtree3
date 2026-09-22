// C++ wrapper for the mutsel rust library

#ifndef IQTREE_UTILS_MUTSEL_WRAPPER_H
#define IQTREE_UTILS_MUTSEL_WRAPPER_H

#include <vector>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include "tree/mtree.h"
#include "alignment/alignment.h"
#include "tree/phylotree.h"
#include "main/outstreambuf.h"

/**
 * Convert a IQ-TREE tree into the format required by the mutsel library.
 */
std::pair<std::vector<double>, std::vector<int32_t>> prepare_mutsel_tree(MTree *tree);

/**
 * Returns the alignment in the format required by mutsel library as dense
 * [L, N] matrix of dense states (0..alignment->num_states-1, or
 * alignment->num_states for gap/ambiguous) and returns L the sequence
 * length and N the number of sequences. Works for any alignment state
 * count (amino acid or codon).
 */
std::tuple<std::vector<uint8_t>, int32_t, int32_t> prepare_mutsel_alignment(Alignment *alignment);

/**
 * Builds the codon<->nucleotide and codon->amino-acid lookup tables the
 * codon mutsel model needs, from the alignment's already-loaded genetic
 * code (alignment->codon_table/genetic_code). codon_nt is
 * alignment->num_states*3 bytes (nucleotide 0-3 at each of the 3 codon
 * positions, A=0,C=1,G=2,T=3); codon_aa is alignment->num_states bytes
 * (amino-acid state 0-19 encoded by each codon, using the same
 * "ARNDCQEGHILKMFPSTWYV" alphabet/order as Alignment's protein states).
 * Only the standard genetic code (61 sense codons) is supported.
 */
void buildCodonMutationTables(Alignment *alignment, std::vector<uint8_t> &codon_nt, std::vector<uint8_t> &codon_aa);

// Mutsel internals, if not compiled in we provide dummy implementations here, otherwise these functions need to be linked.
#if defined(USE_MUTSEL)
extern "C" void rust_mutsel(int32_t *parents,
                            double *branch_lengths,
                            uint8_t *alignment,
                            uint32_t site_num,
                            uint32_t leave_num,
                            uint32_t node_num,
                            const char *model_string,
                            const char *priorMuFile,
                            const char *priorPiFile,
                            uint8_t verbose,
                            double *out_site_freq,
                            double *out_R_matrices,
                            const char *output_prefix);

extern "C" void rust_mutsel_codon(int32_t *parents,
                                  double *branch_lengths,
                                  uint8_t *alignment,
                                  uint32_t site_num,
                                  uint32_t leave_num,
                                  uint32_t node_num,
                                  const uint8_t *codon_nt,
                                  const uint8_t *codon_aa,
                                  const char *model_string,
                                  const char *priorMuFile,
                                  const char *priorPiFile,
                                  uint8_t verbose,
                                  double *out_site_freq,
                                  double *out_R_matrices,
                                  const char *output_prefix);

extern "C" void rust_set_rayon_threads(int32_t num_threads);
#else
void rust_mutsel(int32_t *parents,
                 double *branch_lengths,
                 uint8_t *alignment,
                 uint32_t site_num,
                 uint32_t leave_num,
                 uint32_t node_num,
                 const char *model_string,
                 const char *priorMuFile,
                 const char *priorPiFile,
                 uint8_t verbose,
                 double *out_site_freq,
                 double *out_R_matrices,
                 const char *output_prefix);

void rust_mutsel_codon(int32_t *parents,
                       double *branch_lengths,
                       uint8_t *alignment,
                       uint32_t site_num,
                       uint32_t leave_num,
                       uint32_t node_num,
                       const uint8_t *codon_nt,
                       const uint8_t *codon_aa,
                       const char *model_string,
                       const char *priorMuFile,
                       const char *priorPiFile,
                       uint8_t verbose,
                       double *out_site_freq,
                       double *out_R_matrices,
                       const char *output_prefix);

void rust_set_rayon_threads(int32_t num_threads);
#endif

/**
 * Simple read function which just puts the binary file into vectors. frequences get normalized, rates are checked to be positive. It returns the rate model string of the file.
 * Format is documented in the body of the function.
 * @param nstates the number of states the site model file is expected to hold (20 for amino acid, 61 for codon)
 */
std::string read_binary_site_model_file_internal(std::string &filename, std::vector<double> &site_freq, std::vector<double> &rate_matrices, std::vector<int> &site_model, int nstates);

/**
 * Takes the data from the file (or the rust_mutsel function) and updates the alignment object.
 * This deals with duplicated pattern and stuff.
 */
void write_site_models_to_alignment(Alignment &alignment, const double *site_freq, const double *rate_matrices, int len);

/**
 * Read a site model file and update the alignment object.
 * This is a simple wrapper function around read_binary_site_model_file_internal and write_site_models_to_alignment.
 */
void read_site_model_file(const std::string &filename, Alignment &alignment);

/**
 * Print a site model from the alignment to binary file.
 */
void write_binary_site_model_file(const std::string &filename, Alignment &alignment);

/**
 * Compute the site frequency model using the mutsel library. It writes a site model file with the inferred parameters and also updates the alignment object with the inferred site frequency and rate parameters. It returns a string representing the rate model, e.g., "G4{0.5}" or "R5{0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9/1.0}".
 *
 * Works for both amino-acid (MUTSEL, model name unchanged) and codon
 * alignments: whether alignment->seq_type == SEQ_CODON decides which Rust
 * entry point is called (rust_mutsel vs rust_mutsel_codon) -- the model
 * name and the rest of the orchestration (guide tree loading, site model
 * file writing) is shared. For codon data, mutation is a single 4-state
 * nucleotide GTR shared across all 3 codon positions and fitness/selection
 * is the amino-acid model's, broadcast to codons through the genetic code;
 * only the standard genetic code (61 sense codons) is currently supported.
 */
void computeMutselSiteFrequencyModel(Params &params, Alignment *alignment);

/**
 * Per-site substitution rate implied by the fitted MUTSEL model
 */
DoubleVector computeMutselSiteRates(Alignment &alignment);

#endif // IQTREE_UTILS_MUTSEL_WRAPPER_H