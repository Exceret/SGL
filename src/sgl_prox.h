#ifndef SGL_PROX_HPP
#define SGL_PROX_HPP

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    /*
     * Scalar soft-thresholding:
     *
     *   sign(x) * max(abs(x) - threshold, 0)
     */
    inline double soft_threshold(
    const double x,
    const double threshold
    ) noexcept
    {
        if (x > threshold)
    {
        return x - threshold;
    }
    if (x < -threshold)
    {
        return x + threshold;
    }
    return 0.0;
}


/*
 * In-place sparse-group proximal operator for one group.
 *
 * z must be an m x 1 matrix.
 *
 * First applies the coordinate-wise L1 operation:
 *
 *   u_j = S(z_j, lambda_l1)
 *
 * Then applies the group L2 operation:
 *
 *   z <- max(1 - lambda_group * group_weight / ||u||_2, 0) * u
 *
 * The group-weight scaling is passed explicitly so that the original
 * SGL penalty convention can be preserved by the caller.
 */
inline void sparse_group_prox_inplace(
    arma::mat& z,
         const double lambda_l1,
         const double lambda_group,
         const double group_weight
) noexcept
{
    const arma::uword m = z.n_rows;
    double *z_ptr = z.memptr();
        const double group_threshold =
            lambda_group * group_weight;
        double squared_norm = 0.0;
        // Coordinate-wise L1 shrinkage.
        for (arma::uword j = 0; j < m; ++j)
        {
            const double value = soft_threshold(
                                     z_ptr[j],
                                     lambda_l1
                                 );
            z_ptr[j] = value;
            squared_norm += value * value;
        }
        // The L1 operation already produced the zero vector.
        if (squared_norm == 0.0)
        {
            return;
        }
        const double norm = std::sqrt(squared_norm);
        // Group threshold reached or exceeded.
        if (!(norm > group_threshold))
        {
            for (arma::uword j = 0; j < m; ++j)
            {
                z_ptr[j] = 0.0;
            }
            return;
        }
        const double factor =
            1.0 - group_threshold / norm;
        for (arma::uword j = 0; j < m; ++j)
        {
            z_ptr[j] *= factor;
        }
    }

} // namespace sgl

#endif