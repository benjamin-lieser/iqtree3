// C++ wrapper for the mutsel rust library

#include "mutsel_wrapper.h"

#if !defined(USE_MUTSEL)
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
                 const char *output_prefix)
{
    std::cout << "Mutsel support not compiled in!" << std::endl;
    exit(1);
}

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
                       const char *output_prefix)
{
    std::cout << "Mutsel support not compiled in!" << std::endl;
    exit(1);
}

void rust_set_rayon_threads(int32_t num_threads)
{
    return;
}
#endif

std::pair<std::vector<double>, std::vector<int32_t>> prepare_mutsel_tree(MTree *orig_tree)
{
    // Copy tree

    MTree *tree = new MTree(*orig_tree);

    if (tree->rooted)
    {
        tree->convertToUnrooted();
    }

    std::vector<double> branch_lengths;
    std::vector<int32_t> parent_indices;

    auto num_nodes = tree->nodeNum;

    branch_lengths.resize(num_nodes);
    parent_indices.resize(num_nodes);

    auto orig_root = tree->root;
    ASSERT(orig_root->isLeaf());

    auto root = orig_root->neighbors[0]->node;

    std::vector<std::pair<Node *, Node *>> stack;
    stack.push_back({root, nullptr});

    branch_lengths.at(root->id) = 0.0; // root branch length
    parent_indices.at(root->id) = -1;  // root has no parent

    while (!stack.empty())
    {
        Node *node = stack.back().first;
        Node *dad = stack.back().second;
        stack.pop_back();

        for (auto nei : node->neighbors)
        {
            if (dad != nullptr && nei->node->id == dad->id)
            {
                continue;
            }
            Node *child = nei->node;

            if (nei->length < 0.0)
            {
                throw std::runtime_error("The guide Tree for MUTSEL needs to have branchlengths. Please provide a tree with branch lengths.");
            }

            branch_lengths.at(child->id) = nei->length;
            parent_indices.at(child->id) = node->id;

            stack.push_back({child, node});
        }
    }

    delete tree;

    return {std::move(branch_lengths), std::move(parent_indices)};
}

// Returns the alignment in the format required by mutsel library as dense [L, N] matrix and returns L the sequence length.
// Works for any alignment state count (amino acid or codon): unknown/ambiguous states map to the gap sentinel alignment->num_states.
std::tuple<std::vector<uint8_t>, int32_t, int32_t> prepare_mutsel_alignment(Alignment *alignment)
{
    // [L, N] matrix where L is sequence length and N is number of sequences
    std::vector<uint8_t> sequences;
    size_t L = alignment->getNSite();
    size_t N = alignment->getNSeq();
    size_t nstates = alignment->num_states;

    std::cout << "Preparing alignment for mutsel inference: " << N << " sequences, " << L << " sites" << std::endl;

    sequences.resize(L * N);

    for (size_t site = 0; site < L; ++site)
    {
        Pattern &pat = alignment->at(alignment->getPatternID(site));
        for (size_t seq_idx = 0; seq_idx < N; ++seq_idx)
        {
            auto state = static_cast<uint8_t>(pat.at(seq_idx));
            if (!(state < nstates))
            {
                // map unknown/ambiguous states to gaps (state nstates)
                state = static_cast<uint8_t>(nstates);
            }
            sequences[site * N + seq_idx] = state;
        }
    }

    return std::make_tuple(std::move(sequences), static_cast<int32_t>(L), static_cast<int32_t>(N));
}

void buildCodonMutationTables(Alignment *alignment, std::vector<uint8_t> &codon_nt, std::vector<uint8_t> &codon_aa)
{
    ASSERT(alignment->seq_type == SEQ_CODON);
    // Same alphabet/order as Alignment's protein states (alignment.cpp: symbols_protein).
    static const char aa_alphabet[] = "ARNDCQEGHILKMFPSTWYV";

    size_t nstates = alignment->num_states;
    codon_nt.assign(nstates * 3, 0);
    codon_aa.assign(nstates, 0);

    for (size_t state = 0; state < nstates; ++state)
    {
        int raw = (int)(unsigned char)alignment->codon_table[state]; // 0..63
        codon_nt[state * 3 + 0] = (uint8_t)(raw / 16);
        codon_nt[state * 3 + 1] = (uint8_t)((raw % 16) / 4);
        codon_nt[state * 3 + 2] = (uint8_t)(raw % 4);

        char aa_char = alignment->genetic_code[raw];
        const char *pos = strchr(aa_alphabet, aa_char);
        if (pos == nullptr)
        {
            outError(std::string("Unknown amino acid letter '") + aa_char + "' in genetic code table");
        }
        codon_aa[state] = (uint8_t)(pos - aa_alphabet);
    }
}

std::string read_binary_site_model_file_internal(std::string &filename, std::vector<double> &site_freq, std::vector<double> &rate_matrices, std::vector<int> &site_model, int nstates)
{
    cout << endl
         << "Reading site-specific model file " << filename << " ..." << endl;

    site_freq.clear();
    site_model.clear();
    rate_matrices.clear();

    std::ifstream in;
    std::string rate_model_string;

    // assert little-endian
    uint16_t num = 1;
    if (*((uint8_t *)&num) != 1)
    {
        throw std::runtime_error("Only little-endian machine is supported for reading binary site model file");
    }

    try
    {
        in.open(filename, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Failed to open binary site model file");
        }

        auto read_exact = [&](char *buffer, std::streamsize size, const char *field_name)
        {
            in.read(buffer, size);
            if (!in || in.gcount() != size)
            {
                throw std::runtime_error(std::string("Failed to read ") + field_name + " from binary site model file");
            }
        };

        // Read in the model string. First 8 bytes little endian length of the string, followed by the string itself.
        uint64_t rate_model_string_length;
        read_exact(reinterpret_cast<char *>(&rate_model_string_length), sizeof(uint64_t), "rate model string length");

        std::vector<char> rate_model_string_buf(rate_model_string_length);
        if (rate_model_string_length > 0)
        {
            read_exact(rate_model_string_buf.data(), static_cast<std::streamsize>(rate_model_string_length), "rate model string");
        }
        rate_model_string = std::string(rate_model_string_buf.data(), rate_model_string_length);

        // Read number of sites. 8 bytes

        uint64_t num_sites;
        read_exact(reinterpret_cast<char *>(&num_sites), sizeof(uint64_t), "number of sites");

        int nrates = nstates * (nstates - 1) / 2;

        // num_sites * nstates doubles for site frequencies
        site_freq.resize(num_sites * nstates);
        read_exact(reinterpret_cast<char *>(site_freq.data()),
                   static_cast<std::streamsize>(num_sites * nstates * sizeof(double)),
                   "site frequencies");

        // num_sites * nrates doubles for rate matrices
        rate_matrices.resize(num_sites * nrates);
        read_exact(reinterpret_cast<char *>(rate_matrices.data()),
                   static_cast<std::streamsize>(num_sites * nrates * sizeof(double)),
                   "rate matrices");

        for (size_t i = 0; i < num_sites; ++i)
        {
            site_model.push_back(i);
            for (int j = 0; j < nstates; ++j)
            {
                if (site_freq[i * nstates + j] <= 1e-10)
                    throw std::runtime_error("Frequencies must be strictly bigger than 1e-10");
            }
            double sum = 0;
            for (int j = 0; j < nstates; ++j)
            {
                sum += site_freq[i * nstates + j];
            }
            if (std::abs(sum - 1.0) > 1e-4)
            {
                std::cout << "Warning: frequencies for site " << i + 1 << " do not sum to 1, normalizing..." << std::endl;
                for (int j = 0; j < nstates; ++j)
                {
                    site_freq[i * nstates + j] /= sum;
                }
            }
            for (int j = 0; j < nrates; ++j)
            {
                if (rate_matrices[i * nrates + j] <= 0.0)
                    throw "Rate parameters must be positive";
            }
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Error reading site model file: " + std::string(e.what()));
    }

    return rate_model_string;
}

void write_site_models_to_alignment(Alignment &alignment, const double *site_freq, const double *rate_matrices, int len)
{
    ASSERT(alignment.ptn_rate_mat.empty() &&
           alignment.ptn_state_freq.empty());

    int nstates = alignment.num_states;
    int nrates = alignment.getNumRates();

    size_t nsite = alignment.getNSite();
    if (len != static_cast<int>(nsite))
    {
        throw std::runtime_error("Site model file site count does not match alignment length");
    }

    IntVector site_model(nsite, -1); // map each site to a model
    IntVector pattern_first_site(alignment.getNPattern(), -1);
    for (size_t site = 0; site < nsite; ++site)
    {
        if (pattern_first_site[alignment.getPatternID(site)] == -1)
        {
            pattern_first_site[alignment.getPatternID(site)] = static_cast<int>(site);
        }
    }

    bool aln_changed = false;

    vector<double *> models_freq;
    vector<double *> models_rate;
    for (size_t site = 0; site < nsite; ++site)
    {
        site_model[site] = models_freq.size();

        const double *state_freq_ptr = site_freq + site * nstates;
        const double *rate_mat_ptr = rate_matrices + site * nrates;

        bool add = true;
        int first_site = pattern_first_site[alignment.getPatternID(site)];
        if (first_site < static_cast<int>(site) && site_model[first_site] != -1)
        {
            int first_model = site_model[first_site];
            bool matched_freq_and_rate = true;
            for (int i = 0; i < nstates; ++i)
            {
                if (state_freq_ptr[i] != models_freq[first_model][i])
                {
                    matched_freq_and_rate = false;
                    break;
                }
            }
            if (matched_freq_and_rate)
            {
                for (int i = 0; i < nrates; ++i)
                {
                    if (rate_mat_ptr[i] != models_rate[first_model][i])
                    {
                        matched_freq_and_rate = false;
                        break;
                    }
                }
            }

            if (matched_freq_and_rate)
            {
                site_model[site] = first_model;
                add = false;
            }
            else
            {
                aln_changed = true;
            }
        }

        if (add)
        {
            double *site_freq_entry = new double[nstates];
            memcpy(site_freq_entry, state_freq_ptr, sizeof(double) * nstates);
            models_freq.push_back(site_freq_entry);
            double *site_rate_entry = new double[nrates];
            memcpy(site_rate_entry, rate_mat_ptr, sizeof(double) * nrates);
            models_rate.push_back(site_rate_entry);
        }
    }

    if (aln_changed)
    {
        cout << "Regrouping alignment sites..." << endl;
        alignment.regroupSitePattern(site_model);
        pattern_first_site = IntVector(alignment.getNPattern(), -1);
        for (size_t site = 0; site < nsite; ++site)
        {
            if (pattern_first_site[alignment.getPatternID(site)] == -1)
            {
                pattern_first_site[alignment.getPatternID(site)] = static_cast<int>(site);
            }
        }
    }

    size_t used_models = 0;
    vector<bool> model_used(models_freq.size(), false);
    for (size_t ptn = 0; ptn < alignment.getNPattern(); ++ptn)
    {
        int first_site = pattern_first_site[ptn];
        int model_id = site_model[first_site];
        used_models++;
        model_used[model_id] = true;
        alignment.ptn_rate_mat.push_back(models_rate[model_id]);
        alignment.ptn_state_freq.push_back(models_freq[model_id]);
    }

    for (size_t model_id = 0; model_id < models_freq.size(); ++model_id)
    {
        if (!model_used[model_id])
        {
            delete[] models_freq[model_id];
            delete[] models_rate[model_id];
        }
    }

    cout << used_models << " distinct per-site models detected" << endl;
}

void read_site_model_file(const std::string &filename, Alignment &alignment)
{
    auto site_freq = std::vector<double>();
    auto rate_matrices = std::vector<double>();
    auto site_model = std::vector<int>();
    auto _custom_str = read_binary_site_model_file_internal(const_cast<std::string &>(filename), site_freq, rate_matrices, site_model, alignment.num_states);
    write_site_models_to_alignment(alignment, site_freq.data(), rate_matrices.data(), site_model.size());
    return;
}

void write_binary_site_model_file(const std::string &filename, Alignment &alignment)
{
    size_t nsites = alignment.getNSite();
    size_t nstates = alignment.num_states;
    size_t nrates = alignment.getNumRates();
    // 20 states = amino acid MUTSEL, 61 states = codon MUTSEL (standard genetic code)
    ASSERT(nstates == 20 || nstates == 61);
    try
    {
        ofstream out;
        out.exceptions(ios::failbit | ios::badbit);
        out.open(filename, std::ios::binary);
        IntVector pattern_index;
        alignment.getSitePatternIndex(pattern_index);

        uint64_t custom_string_length = 6; // length of "UNUSED"
        out.write(reinterpret_cast<const char *>(&custom_string_length), sizeof(uint64_t));
        out.write("UNUSED", custom_string_length);

        uint64_t num_sites = nsites;
        out.write(reinterpret_cast<const char *>(&num_sites), sizeof(uint64_t));

        for (size_t i = 0; i < nsites; ++i)
        {
            double *state_freq = alignment.ptn_state_freq[pattern_index[i]];
            out.write(reinterpret_cast<const char *>(state_freq), nstates * sizeof(double));
        }

        for (size_t i = 0; i < nsites; ++i)
        {
            double *rate_mat = alignment.ptn_rate_mat[pattern_index[i]];
            out.write(reinterpret_cast<const char *>(rate_mat), nrates * sizeof(double));
        }

        cout << "Site mutsel model printed to " << filename << endl;
    }
    catch (ios::failure)
    {
        outError(ERR_WRITE_OUTPUT, filename);
    }
}

DoubleVector computeMutselSiteRates(Alignment &alignment)
{
    ASSERT(alignment.ptn_rate_mat.size() == alignment.getNPattern());
    ASSERT(alignment.ptn_state_freq.size() == alignment.getNPattern());

    int nstates = alignment.num_states;

    size_t npattern = alignment.getNPattern();
    DoubleVector pattern_rates(npattern);
    for (size_t ptn = 0; ptn < npattern; ++ptn)
    {
        double *R = alignment.ptn_rate_mat[ptn];
        double *pi = alignment.ptn_state_freq[ptn];
        double rate = 0.0;
        int idx = 0;
        for (int i = 0; i < nstates; ++i)
        {
            for (int j = i + 1; j < nstates; ++j)
            {
                rate += R[idx] * pi[i] * pi[j];
                idx++;
            }
        }
        pattern_rates[ptn] = 2.0 * rate;
    }

    size_t nsite = alignment.getNSite();
    DoubleVector site_rates(nsite);
    for (size_t site = 0; site < nsite; ++site)
    {
        site_rates[site] = pattern_rates[alignment.getPatternID(site)];
    }
    return site_rates;
}

void computeMutselSiteFrequencyModel(Params &params, Alignment *alignment)
{
    ASSERT(params.tree_freq_file);

    // Codon data uses a different mutation process (a 4-state nucleotide
    // GTR expanded through the genetic code, see rust_mutsel_codon) than
    // amino-acid data (a single 20-state GTR, see rust_mutsel); everything
    // else -- the model name, guide tree loading, and site model file
    // writing -- is shared.
    bool is_codon = (alignment->seq_type == SEQ_CODON);
    std::vector<uint8_t> codon_nt, codon_aa;
    if (is_codon)
    {
        if (!alignment->isStandardGeneticCode() || alignment->num_states != 61)
        {
            outError("MUTSEL on codon data currently only supports the standard genetic code (61 sense codons)");
        }
        buildCodonMutationTables(alignment, codon_nt, codon_aa);
    }

    cout << endl
         << "===> COMPUTING MUTSEL MODEL BASED ON TREE FILE " << params.tree_freq_file << endl;
    PhyloTree *tree = new PhyloTree(alignment);
    tree->setParams(&params);
    bool myrooted = params.is_rooted;
    tree->readTree(params.tree_freq_file, myrooted);
    tree->setAlignment(alignment);
    tree->setRootNode(params.root);

    tree->setNumThreads(params.num_threads);

    tree->ensureNumberOfThreadsIsSet(nullptr);

    auto [branch_lengths, parent_indices] = prepare_mutsel_tree(tree);
    auto [sequences, L, N] = prepare_mutsel_alignment(alignment);

    int nstates = alignment->num_states;
    int nrates = alignment->getNumRates();

    double *site_freq = new double[(size_t)L * nstates];

    double *site_rate = new double[(size_t)L * nrates];

    // Close log file, so we can append in the mutsel library without messing up the order of log messages from IQ-TREE and mutsel library
    std::cout << std::flush;
    auto outstream = dynamic_cast<outstreambuf *>(std::cout.rdbuf());
    if (outstream)
    {
        outstream->close();
    }
    else
    {
        throw std::runtime_error("IQTREE-Logging seems to be uninitialized");
    }

    // Calls into the Rust code
    rust_set_rayon_threads(params.num_threads);
    if (is_codon)
    {
        rust_mutsel_codon(parent_indices.data(),
                          branch_lengths.data(),
                          sequences.data(),
                          L,
                          N,
                          parent_indices.size(),
                          codon_nt.data(),
                          codon_aa.data(),
                          params.model_name.c_str(),
                          params.mutsel_prior_rate_file.empty() ? nullptr : params.mutsel_prior_rate_file.c_str(),
                          params.mutsel_prior_freq_file.empty() ? nullptr : params.mutsel_prior_freq_file.c_str(),
                          verbose_mode,
                          site_freq,
                          site_rate,
                          ((string)params.out_prefix).c_str());
    }
    else
    {
        rust_mutsel(parent_indices.data(),
                    branch_lengths.data(),
                    sequences.data(),
                    L,
                    N,
                    parent_indices.size(),
                    params.model_name.c_str(),
                    params.mutsel_prior_rate_file.empty() ? nullptr : params.mutsel_prior_rate_file.c_str(),
                    params.mutsel_prior_freq_file.empty() ? nullptr : params.mutsel_prior_freq_file.c_str(),
                    verbose_mode,
                    site_freq,
                    site_rate,
                    ((string)params.out_prefix).c_str());
    }

    outstream->open(((string)params.out_prefix + ".log").c_str(), std::ios::app); // reopen log file

    write_site_models_to_alignment(*alignment, site_freq, site_rate, L);

    write_binary_site_model_file(((string)params.out_prefix + ".sitemodel").c_str(), *alignment);

    params.print_site_state_freq = WSF_NONE;

    delete[] site_freq;
    delete[] site_rate;
    delete tree;

    cout << endl
         << "===> CONTINUE ANALYSIS USING THE INFERRED MUTSEL MODEL" << endl;
}
