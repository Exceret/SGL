#ifndef SGL_GROUP_NORM_H
#define SGL_GROUP_NORM_H

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    /*
     * Euclidean norm of one group vector.
     *
     * z must be an m x 1 matrix.
     *
     *   ||z||_2 = sqrt(sum_j z_j^2)
     */
    inline double group_l2_norm(
    const arma::mat& z
    ) noexcept
    {
        const arma::uword m = z.n_rows;
        const double *z_ptr = z.memptr();
        double squared_norm = 0.0;
        for (arma::uword j = 0; j < m; ++j)
        {
            squared_norm += z_ptr[j] * z_ptr[j];
        }
        return std::sqrt(squared_norm);
    }

} // namespace sgl

#endif