// Rate categories for the posterior draws of a MUTSEL site model

#ifndef RATESITEDRAWS_H
#define RATESITEDRAWS_H

#include "rateheterogeneity.h"

/**
 * Every site of a MUTSEL site model has num_draws posterior draws of its model (ModelSet::num_draws),
 * which the site-specific likelihood kernel uses as rate categories. All categories have rate 1 and
 * weight 1 / num_draws, so the site likelihood is the average over the draws. Nothing is optimized.
 */
class RateSiteDraws : public RateHeterogeneity
{
public:
    RateSiteDraws(int num_draws) : RateHeterogeneity(), num_draws(num_draws)
    {
        full_name = "Equally weighted mixture of " + convertIntToString(num_draws) + " posterior draws per site";
    }

    virtual int getNRate() { return num_draws; }

    virtual int getNDiscreteRate() { return num_draws; }

    virtual double getRate(int category) { return 1.0; }

    virtual double getProp(int category) { return 1.0 / num_draws; }

    virtual void writeInfo(ostream &out)
    {
        out << full_name << endl;
    }

private:
    int num_draws;
};

#endif // RATESITEDRAWS_H
