#pragma once
#ifndef __INTERNAL_HPP
    #define __INTERNAL_HPP
#endif // __INTERNAL_HPP

#include <cmath>
#include <cstdio>
#include <numbers>

namespace r {

#ifndef M_E
    #define M_E 2.718281828459045235360287471353 /* e */
#endif

#ifndef M_LOG2E
    #define M_LOG2E 1.442695040888963407359924681002 /* log2(e) */
#endif

#ifndef M_LOG10E
    #define M_LOG10E 0.434294481903251827651128918917 /* log10(e) */
#endif

#ifndef M_LN2
    #define M_LN2 0.693147180559945309417232121458 /* ln(2) */
#endif

#ifndef M_LN10
    #define M_LN10 2.302585092994045684017991454684 /* ln(10) */
#endif

#ifndef M_PI
    #define M_PI 3.141592653589793238462643383280 /* pi */
#endif

#ifndef M_2PI
    #define M_2PI 6.283185307179586476925286766559 /* 2*pi */
#endif

#ifndef M_PI_2
    #define M_PI_2 1.570796326794896619231321691640 /* pi/2 */
#endif

#ifndef M_PI_4
    #define M_PI_4 0.785398163397448309615660845820 /* pi/4 */
#endif

#ifndef M_1_PI
    #define M_1_PI 0.318309886183790671537767526745 /* 1/pi */
#endif

#ifndef M_2_PI
    #define M_2_PI 0.636619772367581343075535053490 /* 2/pi */
#endif

#ifndef M_2_SQRTPI
    #define M_2_SQRTPI 1.128379167095512573896158903122 /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
    #define M_SQRT2 1.414213562373095048801688724210 /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
    #define M_SQRT1_2 0.707106781186547524400844362105 /* 1/sqrt(2) */
#endif

#ifndef M_SQRT_3
    #define M_SQRT_3 1.732050807568877293527446341506 /* sqrt(3) */
#endif

#ifndef M_SQRT_32
    #define M_SQRT_32 5.656854249492380195206754896838 /* sqrt(32) */
#endif

#ifndef M_LOG10_2
    #define M_LOG10_2 0.301029995663981195213738894724 /* log10(2) */
#endif

#ifndef M_SQRT_PI
    #define M_SQRT_PI 1.772453850905516027298167483341 /* sqrt(pi) */
#endif

#ifndef M_1_SQRT_2PI
    #define M_1_SQRT_2PI 0.398942280401432677939946059934 /* 1/sqrt(2pi) */
#endif

#ifndef M_SQRT_2dPI
    #define M_SQRT_2dPI 0.797884560802865355879892119869 /* sqrt(2/pi) */
#endif

#ifndef M_LN_2PI
    #define M_LN_2PI 1.837877066409345483560659472811 /* log(2*pi) */
#endif

#ifndef M_LN_SQRT_PI
    #define M_LN_SQRT_PI                                                                                                                   \
        0.572364942924700087071713675677 /* log(sqrt(pi))
								   == log(pi)/2 */
#endif

#ifndef M_LN_SQRT_2PI
    #define M_LN_SQRT_2PI                                                                                                                  \
        0.918938533204672741780329736406 /* log(sqrt(2*pi))
								 == log(2*pi)/2 */
#endif

#ifndef M_LN_SQRT_PId2
    #define M_LN_SQRT_PId2                                                                                                                 \
        0.225791352644727432363097614947 /* log(sqrt(pi/2))
								   == log(pi/2)/2 */
#endif

#define ISNAN(x)  (isnan(x) != 0)
#define ML_POSINF (1.0 / 0.0)
#define ML_NEGINF ((-1.0) / 0.0)
#define ML_NAN    (0.0 / 0.0)

#define ML_WARN_return_NAN                                                                                                                 \
    {                                                                                                                                      \
        ML_WARNING(ME_DOMAIN, "");                                                                                                         \
        return ML_NAN;                                                                                                                     \
    }

/* For a long time prior to R 2.3.0 ML_WARNING did nothing.
   We don't report ME_DOMAIN errors as the callers collect ML_NANs into
   a single warning.
 */
#define ML_WARNING(x, s)                                                                                                                   \
    {                                                                                                                                      \
        if (x > ME_DOMAIN) {                                                                                                               \
            char* msg = "";                                                                                                                \
            switch (x) {                                                                                                                   \
                case ME_DOMAIN    : msg = _("argument out of domain in '%s'\n"); break;                                                    \
                case ME_RANGE     : msg = _("value out of range in '%s'\n"); break;                                                        \
                case ME_NOCONV    : msg = _("convergence failed in '%s'\n"); break;                                                        \
                case ME_PRECISION : msg = _("full precision may not have been achieved in '%s'\n"); break;                                 \
                case ME_UNDERFLOW : msg = _("underflow occurred in '%s'\n"); break;                                                        \
            }                                                                                                                              \
            MATHLIB_WARNING(msg, s);                                                                                                       \
        }                                                                                                                                  \
    }

    static inline long double __stdcall dnorm4(double x, double mu, double sigma, int give_log) noexcept {
#ifdef IEEE_754
        if (ISNAN(x) || ISNAN(mu) || ISNAN(sigma)) return x + mu + sigma;
#endif
        if (sigma < 0) {
            //
            ::fputws(L"", stderr);
            return ML_NAN;
        };
        if (!R_FINITE(sigma)) return R_D__0;
        if (!R_FINITE(x) && mu == x) return ML_NAN; /* x-mu is NaN */
        if (sigma == 0) return (x == mu) ? ML_POSINF : R_D__0;
        x = (x - mu) / sigma;

        if (!R_FINITE(x)) return R_D__0;

        x = fabs(x);
        if (x >= 2 * sqrt(DBL_MAX)) return R_D__0;
        if (give_log) return -(M_LN_SQRT_2PI + 0.5 * x * x + log(sigma));
        //  M_1_SQRT_2PI = 1 / sqrt(2 * pi)
#ifdef MATHLIB_FAST_dnorm
        // and for R <= 3.0.x and R-devel upto 2014-01-01:
        return M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;
#else
        // more accurate, less fast :
        if (x < 5) return M_1_SQRT_2PI * exp(-0.5 * x * x) / sigma;

        /* ELSE:

     * x*x  may lose upto about two digits accuracy for "large" x
     * Morten Welinder's proposal for PR#15620
     * https://bugs.r-project.org/show_bug.cgi?id=15620

     * -- 1 --  No hoop jumping when we underflow to zero anyway:

     *  -x^2/2 <         log(2)*.Machine$double.min.exp  <==>
     *     x   > sqrt(-2*log(2)*.Machine$double.min.exp) =IEEE= 37.64031
     * but "thanks" to denormalized numbers, underflow happens a bit later,
     *  effective.D.MIN.EXP <- with(.Machine, double.min.exp + double.ulp.digits)
     * for IEEE, DBL_MIN_EXP is -1022 but "effective" is -1074
     * ==> boundary = sqrt(-2*log(2)*(.Machine$double.min.exp + .Machine$double.ulp.digits))
     *              =IEEE=  38.58601
     * [on one x86_64 platform, effective boundary a bit lower: 38.56804]
     */
        if (x > sqrt(-2 * M_LN2 * (DBL_MIN_EXP + 1 - DBL_MANT_DIG))) return 0.;

        /* Now, to get full accuracy, split x into two parts,
     *  x = x1+x2, such that |x2| <= 2^-16.
     * Assuming that we are using IEEE doubles, that means that
     * x1*x1 is error free for x<1024 (but we have x < 38.6 anyway).

     * If we do not have IEEE this is still an improvement over the naive formula.
     */
        double x1 = //  R_forceint(x * 65536) / 65536 =
            ldexp(R_forceint(ldexp(x, 16)), -16);
        double x2 = x - x1;
        return M_1_SQRT_2PI / sigma * (exp(-0.5 * x1 * x1) * exp((-0.5 * x2 - x1) * x2));
#endif
    }

    static inline long double __stdcall dt(double x, double n, int give_log) noexcept {
#ifdef IEEE_754
        if (ISNAN(x) || ISNAN(n)) return x + n;
#endif
        if (n <= 0) ML_WARN_return_NAN;
        if (!R_FINITE(x)) return R_D__0;
        if (!R_FINITE(n)) return dnorm(x, 0., 1., give_log);

        double u, t = -bd0(n / 2., (n + 1) / 2.) + stirlerr((n + 1) / 2.) - stirlerr(n / 2.),
                  x2n = x * x / n, // in  [0, Inf]
            ax        = 0.,        // <- -Wpedantic
            l_x2n;                 // := log(sqrt(1 + x2n)) = log(1 + x2n)/2
        bool lrg_x2n = (x2n > 1. / DBL_EPSILON);
        if (lrg_x2n) { // large x^2/n :
            ax    = fabs(x);
            l_x2n = log(ax) - log(n) / 2.; // = log(x2n)/2 = 1/2 * log(x^2 / n)
            u     =                        //  log(1 + x2n) * n/2 =  n * log(1 + x2n)/2 =
                n * l_x2n;
        } else if (x2n > 0.2) {
            l_x2n = log(1 + x2n) / 2.;
            u     = n * l_x2n;
        } else {
            l_x2n = log1p(x2n) / 2.;
            u     = -bd0(n / 2., (n + x * x) / 2.) + x * x / 2.;
        }

        //old: return  R_D_fexp(M_2PI*(1+x2n), t-u);

        // R_D_fexp(f,x) :=  (give_log ? -0.5*log(f)+(x) : exp(x)/sqrt(f))
        // f = 2pi*(1+x2n)
        //  ==> 0.5*log(f) = log(2pi)/2 + log(1+x2n)/2 = log(2pi)/2 + l_x2n
        //	     1/sqrt(f) = 1/sqrt(2pi * (1+ x^2 / n))
        //		       = 1/sqrt(2pi)/(|x|/sqrt(n)*sqrt(1+1/x2n))
        //		       = M_1_SQRT_2PI * sqrt(n)/ (|x|*sqrt(1+1/x2n))
        if (give_log) return t - u - (M_LN_SQRT_2PI + l_x2n);

        // else :  if(lrg_x2n) : sqrt(1 + 1/x2n) ='= sqrt(1) = 1
        double I_sqrt_ = (lrg_x2n ? sqrt(n) / ax : exp(-l_x2n));
        return exp(t - u) * M_1_SQRT_2PI * I_sqrt_;
    }

} // namespace r
