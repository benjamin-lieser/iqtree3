/*
 * ratesitesamples.h
 *
 * Rate "heterogeneity" of a site-specific model with several models per site, e.g. posterior
 * samples of the MUTSEL model: one category per model, all with rate 1 and weight 1/K, so the
 * likelihood of a site is the mean of the likelihoods under its K models. The kernels map
 * category c of a pattern to the c-th model of the pattern (ModelSubst::getSiteSampleStride).
 */

#ifndef RATESITESAMPLES_H
#define RATESITESAMPLES_H

#include "rateheterogeneity.h"

class RateSiteSamples : public RateHeterogeneity
{
public:
    RateSiteSamples(int num_samples, PhyloTree *tree) : RateHeterogeneity(), num_samples(num_samples)
    {
        name = "";
        full_name = "Mean over " + convertIntToString(num_samples) + " samples of the site-specific model";
        setTree(tree);
    }

    virtual int getNRate() { return num_samples; }

    virtual int getNDiscreteRate() { return num_samples; }

    virtual double getRate(int category) { return 1.0; }

    virtual double getProp(int category) { return 1.0 / num_samples; }

    virtual void writeInfo(ostream &out)
    {
        out << "Site-specific model: mean likelihood over " << num_samples << " samples per site" << endl;
    }

protected:
    int num_samples;
};

#endif
