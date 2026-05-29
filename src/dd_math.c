/* Double-double arithmetic. See dd_math.h.
   Algorithms: Hida/Li/Bailey 2001 ("Library for Double-Double and Quad-Double
   Arithmetic", LBNL); the same canonical formulation cuda_math.cu uses for
   the GPU dd128 kernels. */

#include "dd_math.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── error-free transforms ──────────────────────────────────────────────── */

/* Knuth's TwoSum: returns (s, e) with s = fl(a+b) and a+b == s+e exactly. */
static inline void two_sum(double a, double b, double *s, double *e) {
    double ss = a + b;
    double bb = ss - a;
    *e = (a - (ss - bb)) + (b - bb);
    *s = ss;
}

/* Dekker's QuickTwoSum: same contract but requires |a| >= |b|. */
static inline void quick_two_sum(double a, double b, double *s, double *e) {
    double ss = a + b;
    *e = b - (ss - a);
    *s = ss;
}

/* Veltkamp split: a = hi + lo with hi having 26 mantissa bits, lo 27. */
static inline void split(double a, double *hi, double *lo) {
    /* The 2^27 + 1 constant assumes 53-bit doubles; it splits the mantissa
       cleanly so the two products in two_prod don't overflow. */
    double t = 134217729.0 * a;
    *hi = t - (t - a);
    *lo = a - *hi;
}

/* TwoProd: (p, e) with p = fl(a*b) and a*b == p+e exactly. Uses FMA when
   available (one rounding); otherwise the Dekker split (slower, six flops). */
static inline void two_prod(double a, double b, double *p, double *e) {
#if defined(FP_FAST_FMA) || defined(__FMA__) || \
    (defined(__ARM_FEATURE_FMA) && __ARM_FEATURE_FMA)
    double pp = a * b;
    *e = fma(a, b, -pp);
    *p = pp;
#else
    double a_hi, a_lo, b_hi, b_lo, pp;
    pp = a * b;
    split(a, &a_hi, &a_lo);
    split(b, &b_hi, &b_lo);
    *e = ((a_hi * b_hi - pp) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    *p = pp;
#endif
}

/* ── constructors / accessors ───────────────────────────────────────────── */

ndarray_dd_t ndarray_dd_from_double(double a) {
    ndarray_dd_t r; r.hi = a; r.lo = 0.0; return r;
}

ndarray_dd_t ndarray_dd_from_pair(double hi, double lo) {
    ndarray_dd_t r; r.hi = hi; r.lo = lo; return r;
}

double ndarray_dd_to_double(ndarray_dd_t a) {
    return a.hi;
}

/* ── unary ─────────────────────────────────────────────────────────────── */

ndarray_dd_t ndarray_dd_neg(ndarray_dd_t a) {
    ndarray_dd_t r; r.hi = -a.hi; r.lo = -a.lo; return r;
}

ndarray_dd_t ndarray_dd_abs(ndarray_dd_t a) {
    return (a.hi < 0.0) ? ndarray_dd_neg(a) : a;
}

/* ── add / sub ─────────────────────────────────────────────────────────── */

ndarray_dd_t ndarray_dd_add(ndarray_dd_t a, ndarray_dd_t b) {
    double s1, s2, t1, t2;
    two_sum(a.hi, b.hi, &s1, &s2);
    two_sum(a.lo, b.lo, &t1, &t2);
    s2 += t1;
    quick_two_sum(s1, s2, &s1, &s2);
    s2 += t2;
    quick_two_sum(s1, s2, &s1, &s2);
    return ndarray_dd_from_pair(s1, s2);
}

ndarray_dd_t ndarray_dd_sub(ndarray_dd_t a, ndarray_dd_t b) {
    return ndarray_dd_add(a, ndarray_dd_neg(b));
}

/* ── mul / div ─────────────────────────────────────────────────────────── */

ndarray_dd_t ndarray_dd_mul(ndarray_dd_t a, ndarray_dd_t b) {
    double p1, p2;
    two_prod(a.hi, b.hi, &p1, &p2);
    p2 += a.hi * b.lo + a.lo * b.hi;
    double s1, s2;
    quick_two_sum(p1, p2, &s1, &s2);
    return ndarray_dd_from_pair(s1, s2);
}

ndarray_dd_t ndarray_dd_div(ndarray_dd_t a, ndarray_dd_t b) {
    if (b.hi == 0.0 && b.lo == 0.0) {
        if (a.hi == 0.0 && a.lo == 0.0) return ndarray_dd_from_double(NAN);
        return ndarray_dd_from_double((a.hi < 0.0) ? -INFINITY : INFINITY);
    }
    /* Three-step long division: q = (q1 + q2 + q3), each correction reduces
       the residual by one double-precision step. Total error ~ 2^-104. */
    double q1 = a.hi / b.hi;
    ndarray_dd_t r = ndarray_dd_sub(a, ndarray_dd_mul(b, ndarray_dd_from_double(q1)));
    double q2 = r.hi / b.hi;
    r = ndarray_dd_sub(r, ndarray_dd_mul(b, ndarray_dd_from_double(q2)));
    double q3 = r.hi / b.hi;
    double s1, s2;
    quick_two_sum(q1, q2, &s1, &s2);
    s2 += q3;
    quick_two_sum(s1, s2, &s1, &s2);
    return ndarray_dd_from_pair(s1, s2);
}

/* ── truncate toward zero ───────────────────────────────────────────────── */

ndarray_dd_t ndarray_dd_trunc(ndarray_dd_t a) {
    double hi = trunc(a.hi);
    /* If the integer part of `a` was entirely in `hi`, the residual lives in
       `lo`. Otherwise truncating `hi` already dropped a fractional part and
       `lo` is no longer meaningful. */
    if (hi == a.hi) {
        double lo = trunc(a.lo);
        double s, e;
        quick_two_sum(hi, lo, &s, &e);
        return ndarray_dd_from_pair(s, e);
    }
    return ndarray_dd_from_pair(hi, 0.0);
}

ndarray_dd_t ndarray_dd_fmod(ndarray_dd_t a, ndarray_dd_t b) {
    if (b.hi == 0.0 && b.lo == 0.0) return ndarray_dd_from_double(NAN);
    ndarray_dd_t q = ndarray_dd_trunc(ndarray_dd_div(a, b));
    return ndarray_dd_sub(a, ndarray_dd_mul(q, b));
}

/* ── pow ───────────────────────────────────────────────────────────────── */

/* Full DD pow would need DD log + DD exp (substantial). We do integer-exponent
   binary exponentiation when feasible (full DD precision retained), and fall
   back to fp64 pow otherwise. */
ndarray_dd_t ndarray_dd_pow(ndarray_dd_t a, ndarray_dd_t b) {
    /* Detect integer exponent within int64 range with no fractional part. */
    double bd = b.hi;
    if (b.lo == 0.0 && bd == trunc(bd) && fabs(bd) < 9.2233720368547758e18) {
        long long n = (long long)bd;
        long long an = n < 0 ? -n : n;
        ndarray_dd_t base = a;
        ndarray_dd_t result = ndarray_dd_from_double(1.0);
        while (an > 0) {
            if (an & 1) result = ndarray_dd_mul(result, base);
            base = ndarray_dd_mul(base, base);
            an >>= 1;
        }
        if (n < 0) result = ndarray_dd_div(ndarray_dd_from_double(1.0), result);
        return result;
    }
    /* Fractional exponent: fall back to fp64. Precision degraded — match
       the existing __float128 path which also falls back when both args
       are non-integer (quadmath's powq is only marginally better). */
    return ndarray_dd_from_double(pow(a.hi, b.hi));
}

/* ── sqrt / rsqrt ──────────────────────────────────────────────────────── */

/**
 * @brief Double-double square root.
 *
 * Computes sqrt(a) to ~106-bit precision via a single Newton refinement
 * step starting from a fp64 seed. Identity:
 *   y' = 0.5 * (y + a / y)
 * Carried out in DD so the residual `a - y*y` is captured exactly.
 *
 * NaN propagation: returns NaN for any input with `hi < 0` or NaN.
 * Zero is preserved exactly (no division by zero).
 *
 * @param[in] a Non-negative DD input.
 * @return DD square root of @p a.
 */
ndarray_dd_t ndarray_dd_sqrt(ndarray_dd_t a) {
    if (a.hi == 0.0 && a.lo == 0.0) return a;
    if (a.hi < 0.0 || ndarray_dd_isnan(a)) {
        return ndarray_dd_from_double(NAN);
    }
    double y = sqrt(a.hi);
    /* y' = y + (a - y*y) / (2*y) — refines the seed from fp64 (53-bit) to
       full DD precision in one pass. */
    ndarray_dd_t y_dd  = ndarray_dd_from_double(y);
    ndarray_dd_t y_sq  = ndarray_dd_mul(y_dd, y_dd);
    ndarray_dd_t diff  = ndarray_dd_sub(a, y_sq);
    ndarray_dd_t denom = ndarray_dd_from_double(2.0 * y);
    ndarray_dd_t corr  = ndarray_dd_div(diff, denom);
    return ndarray_dd_add(y_dd, corr);
}

/**
 * @brief Double-double reciprocal square root, `1 / sqrt(a)`.
 *
 * Implemented as `1.0 / sqrt(a)` rather than a fused Newton step on
 * `1/sqrt` because the DD division and DD sqrt are both already
 * iteratively refined; chaining them keeps the implementation small
 * without measurably worse precision than the fused form.
 *
 * @param[in] a Positive DD input.
 * @return DD reciprocal square root of @p a; +inf for zero, NaN for
 *         negative.
 */
ndarray_dd_t ndarray_dd_rsqrt(ndarray_dd_t a) {
    ndarray_dd_t r = ndarray_dd_sqrt(a);
    return ndarray_dd_div(ndarray_dd_from_double(1.0), r);
}

/* ── DD-precision transcendentals ───────────────────────────────────────── */

/* Pre-computed DD constants, accurate to ~106 bits. Each is split so
   `hi` is the closest fp64 approximation and `lo` is the residual.
   Verified against libquadmath: `(double)ln2_hi + (double)ln2_lo` matches
   `logq(2.0Q)` to within 1 DD ULP. */
static const ndarray_dd_t DD_LN2 = {
    /* ln(2)  = 0.69314718055994530941723212145817... */
     0.6931471805599453,    /* hi: 0.693147180559945286... (closest double) */
     2.3190468138462996e-17 /* lo: residual = ln2 - hi   */
};
static const ndarray_dd_t DD_LOG2_E = {
    /* 1/ln(2) = 1.44269504088896340735992468100... */
     1.4426950408889634,
     2.0355273740931033e-17
};
static const ndarray_dd_t DD_LOG10_E = {
    /* 1/ln(10) = 0.43429448190325182765112891891... */
     0.4342944819032518,
     1.0983196502167645e-17
};

/**
 * @brief DD-precision exp(x).
 *
 * Range reduction: write x = k·ln(2) + r with k = round(x / ln(2)) and
 * |r| ≤ ln(2)/2 ≈ 0.347. Then exp(x) = 2^k · exp(r); the 2^k factor
 * is exact in fp64 (just shifts the exponent), and exp(r) is evaluated
 * via the Taylor series 1 + r + r²/2! + r³/3! + … in DD arithmetic
 * using Horner's method. The series is summed through r²⁴/24!: at the
 * worst-case |r| ≤ ln(2)/2 ≈ 0.3466 the first omitted term r²⁵/25! ≈
 * 6.8e-37 is far below DD epsilon (~2⁻¹⁰⁶ ≈ 1.2e-32), so the result
 * carries full ~32-digit DD precision. (Twenty terms — the original
 * cutoff — left r²¹/21! ≈ 4.2e-30 in the remainder, capping accuracy
 * at ~29 digits and making the GPU DD path diverge from the CPU
 * libquadmath path at the 31st digit.)
 *
 * Handles overflow (`exp(x) > DBL_MAX`) by returning +inf and underflow
 * (`exp(x) < DBL_MIN_SUBNORMAL`) by returning 0. NaN propagates.
 *
 * @param[in] a Input DD value.
 * @return exp(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_exp(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (isinf(a.hi)) return ndarray_dd_from_double(a.hi > 0 ? INFINITY : 0.0);
    /* Fast over/underflow guards — exp(±709.78…) is the fp64 edge. */
    if (a.hi >  709.7827) return ndarray_dd_from_double(INFINITY);
    if (a.hi < -745.1332) return ndarray_dd_from_double(0.0);
    /* Range reduction: k = round(x · log2_e), r = x − k · ln2. */
    double k_d = round(a.hi * 1.4426950408889634);
    int    k   = (int)k_d;
    ndarray_dd_t k_dd = ndarray_dd_from_double(k_d);
    ndarray_dd_t r    = ndarray_dd_sub(a, ndarray_dd_mul(k_dd, DD_LN2));

    /* Horner evaluation of 1 + r·(1 + r/2·(1 + r/3·(… + r/24))) */
    ndarray_dd_t result = ndarray_dd_from_double(1.0);
    for (int i = 24; i >= 1; i--) {
        /* result = 1 + (r/i) · result */
        ndarray_dd_t r_over_i = ndarray_dd_div(r, ndarray_dd_from_double((double)i));
        result = ndarray_dd_add(ndarray_dd_from_double(1.0),
                                 ndarray_dd_mul(r_over_i, result));
    }

    /* Scale by 2^k using ldexp on each limb — exact, exponent-only op. */
    result.hi = ldexp(result.hi, k);
    result.lo = ldexp(result.lo, k);
    return result;
}

/**
 * @brief DD-precision expm1(x) = exp(x) − 1.
 *
 * Near zero, computing `exp(x) − 1` directly suffers catastrophic
 * cancellation. Use the Taylor series of expm1 itself for |x| ≤ 0.5:
 *     expm1(x) = x + x²/2! + x³/3! + … = x · (1 + x/2 · (1 + x/3 · (…)))
 * which converges with 25 terms at full DD precision. For larger |x|
 * the cancellation is negligible — defer to `exp(x) − 1`.
 *
 * @param[in] a Input DD value.
 * @return exp(a) − 1 in DD precision.
 */
ndarray_dd_t ndarray_dd_expm1(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi >= 0.5 || a.hi <= -0.5) {
        return ndarray_dd_sub(ndarray_dd_exp(a), ndarray_dd_from_double(1.0));
    }
    /* Horner of x·(1 + x/2·(1 + x/3·(…))) — start at i=25 to capture
       (0.5)^25/25! ≈ 1.9e-33 < DD eps. */
    ndarray_dd_t result = ndarray_dd_from_double(1.0);
    for (int i = 25; i >= 2; i--) {
        ndarray_dd_t a_over_i = ndarray_dd_div(a, ndarray_dd_from_double((double)i));
        result = ndarray_dd_add(ndarray_dd_from_double(1.0),
                                 ndarray_dd_mul(a_over_i, result));
    }
    return ndarray_dd_mul(a, result);
}

/**
 * @brief DD-precision log(x) (natural logarithm).
 *
 * Range reduction: write x = m · 2^e via `frexp`, so m ∈ [0.5, 1). To
 * keep the substitution `u = (m − 1)/(m + 1)` small we conditionally
 * shift m into [√0.5, √2) ≈ [0.707, 1.414); then |u| ≤ 0.172. The
 * atanh-style series ln(m) = 2·(u + u³/3 + u⁵/5 + u⁷/7 + …) converges
 * about twice as fast as the plain Taylor of ln(1+y) because the
 * even-power terms vanish. Twenty-six odd terms (through u^51/51) give
 * full ~32-digit DD precision at the |u| ≤ 0.172 boundary.
 *
 * Final: log(x) = 2·Σ + e·ln(2).
 *
 * NaN / negative / zero handling:
 *   log(NaN) → NaN, log(<0) → NaN, log(0) → −inf, log(+inf) → +inf.
 *
 * @param[in] a Input DD value (must be > 0 for a finite result).
 * @return log(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_log(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi < 0.0) return ndarray_dd_from_double(NAN);
    if (a.hi == 0.0 && a.lo == 0.0) return ndarray_dd_from_double(-INFINITY);
    if (isinf(a.hi)) return ndarray_dd_from_double(INFINITY);

    /* Decompose hi = m · 2^e so m ∈ [0.5, 1). The lo limb is folded back
       in DD multiplication. */
    int    e;
    double m_hi = frexp(a.hi, &e);
    /* Re-normalize the DD pair after stripping 2^e: dd = a / 2^e. */
    ndarray_dd_t m = ndarray_dd_from_pair(m_hi, ldexp(a.lo, -e));

    /* Bring m into [sqrt(0.5), sqrt(2)) so |u| ≤ ~0.172. */
    if (m.hi < 0.7071067811865476) {
        m   = ndarray_dd_add(m, m);    /* m · 2 */
        e  -= 1;
    }

    /* u = (m − 1) / (m + 1). */
    ndarray_dd_t one = ndarray_dd_from_double(1.0);
    ndarray_dd_t u   = ndarray_dd_div(ndarray_dd_sub(m, one),
                                       ndarray_dd_add(m, one));
    ndarray_dd_t u2  = ndarray_dd_mul(u, u);

    /* 2·atanh(u) = 2·(u + u³/3 + u⁵/5 + … + u^(2N-1)/(2N-1)). For
       |u| ≤ 0.172 (the post-shift range), 2N-1 = 51 gives the worst-
       case truncated term u^51/51 ≈ 4·10⁻⁴² — well below DD epsilon.
       Use `dd_div` for the 1/k constants; `dd_from_double(1.0 / k)`
       would only have fp64 precision in the constant. */
    ndarray_dd_t sum = ndarray_dd_div(one, ndarray_dd_from_double(51.0));
    for (int k = 49; k >= 1; k -= 2) {
        ndarray_dd_t inv_k = ndarray_dd_div(one, ndarray_dd_from_double((double)k));
        sum = ndarray_dd_add(inv_k, ndarray_dd_mul(u2, sum));
    }
    /* Multiply by 2u to get the series sum. */
    ndarray_dd_t log_m = ndarray_dd_mul(u, sum);
    log_m = ndarray_dd_add(log_m, log_m);  /* · 2 */

    /* log(x) = log(m) + e · ln(2). */
    ndarray_dd_t e_dd = ndarray_dd_from_double((double)e);
    return ndarray_dd_add(log_m, ndarray_dd_mul(e_dd, DD_LN2));
}

/**
 * @brief DD-precision log1p(x) = log(1 + x).
 *
 * For |x| ≤ 0.5 the value 1 + x suffers catastrophic cancellation of
 * x's sub-fp64 information (when |x| ≲ fp64 epsilon the whole of x lands
 * in the lo limb and is rounded away by the subsequent range reduction).
 * Use instead the area-hyperbolic-tangent identity
 *     log1p(x) = 2·atanh( x / (2 + x) ),
 * with u = x / (2 + x). The divisor 2 + x stays in [1.5, 2.5] so it
 * never cancels and the DD add/divide preserve x's lo limb in full;
 * |u| ≤ 0.2 over the branch, so the odd series 2·(u + u³/3 + … + u⁵¹/51)
 * (26 odd terms) is below DD epsilon. For |x| > 0.5 there is no
 * cancellation in 1 + x, so defer to `dd_log(1 + x)` — that path also
 * covers the x ≤ −1 (→ NaN / −inf) and +inf edges.
 *
 * @param[in] a Input DD value (a > −1 for a finite result).
 * @return log(1 + a) in DD precision.
 */
ndarray_dd_t ndarray_dd_log1p(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi >= 0.5 || a.hi <= -0.5) {
        return ndarray_dd_log(ndarray_dd_add(ndarray_dd_from_double(1.0), a));
    }
    ndarray_dd_t one = ndarray_dd_from_double(1.0);
    ndarray_dd_t u   = ndarray_dd_div(a,
                           ndarray_dd_add(ndarray_dd_from_double(2.0), a));
    ndarray_dd_t u2  = ndarray_dd_mul(u, u);
    /* Same 26-odd-term atanh ladder as ndarray_dd_log; |u| ≤ 0.2 here so
       the truncated term u^53/53 is far below DD epsilon. */
    ndarray_dd_t sum = ndarray_dd_div(one, ndarray_dd_from_double(51.0));
    for (int k = 49; k >= 1; k -= 2) {
        ndarray_dd_t inv_k = ndarray_dd_div(one, ndarray_dd_from_double((double)k));
        sum = ndarray_dd_add(inv_k, ndarray_dd_mul(u2, sum));
    }
    ndarray_dd_t r = ndarray_dd_mul(u, sum);
    return ndarray_dd_add(r, r);  /* · 2 */
}

/**
 * @brief DD-precision exp2(x) = 2^x.
 *
 * Implemented as `exp(x · ln(2))` so the existing DD exp drives the
 * precision; the multiplication by `DD_LN2` is exact at DD precision
 * because `x` is the only fp64-tier input.
 *
 * @param[in] a Input DD value.
 * @return 2^a in DD precision.
 */
ndarray_dd_t ndarray_dd_exp2(ndarray_dd_t a) {
    return ndarray_dd_exp(ndarray_dd_mul(a, DD_LN2));
}

/**
 * @brief DD-precision log2(x) = log(x) / ln(2) = log(x) · log2(e).
 *
 * Special-case exact integer powers of two so `log2(2^k)` returns
 * exactly `k` (matching the GPU `tcuda_log2_fp` short-circuit and
 * the CPU libm guarantee on libquadmath builds).
 *
 * @param[in] a Input DD value (a > 0 for a finite result).
 * @return log2(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_log2(ndarray_dd_t a) {
    /* Power-of-2 short-circuit: frexp(2^k) = (0.5, k+1). */
    if (a.lo == 0.0 && isfinite(a.hi) && a.hi > 0.0) {
        int e;
        double m = frexp(a.hi, &e);
        if (m == 0.5) return ndarray_dd_from_double((double)(e - 1));
    }
    return ndarray_dd_mul(ndarray_dd_log(a), DD_LOG2_E);
}

/**
 * @brief DD-precision log10(x) = log(x) · log10(e).
 *
 * @param[in] a Input DD value (a > 0 for a finite result).
 * @return log10(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_log10(ndarray_dd_t a) {
    return ndarray_dd_mul(ndarray_dd_log(a), DD_LOG10_E);
}

/**
 * @brief DD-precision logb(x) — binary exponent of |x| as a DD integer.
 *
 * `logb(x)` is defined as `floor(log2(|x|))` for finite normal x and
 * is already integer-valued, so the result fits exactly in fp64. The
 * `lo` limb of the result is always 0.
 *
 * @param[in] a Input DD value.
 * @return logb(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_logb(ndarray_dd_t a) {
    return ndarray_dd_from_double(logb(a.hi));
}

/* ── DD-precision hyperbolic functions ──────────────────────────────────────
   sinh / cosh / tanh / asinh / acosh / atanh evaluated entirely in DD
   arithmetic by composing the DD exp / expm1 / log / log1p / sqrt
   primitives above, so each carries ~32 decimal digits — the same tier as
   libquadmath's sinhq/coshq/… on the native fp128 build. The algorithms are
   byte-identical to the device twins dd_sinh … dd_atanh in
   src/ndmath/cuda/cuda_math.cu, which keeps the CPU DD fallback and the GPU
   dd path bit-for-bit aligned. (These run only on the DD fp128 backend —
   non-quadmath platforms; the libquadmath build calls sinhq/… directly.) */

/**
 * @brief DD-precision sinh(x).
 *
 * Near zero the direct (e^x − e^−x)/2 suffers catastrophic cancellation, so
 * for |x| < 0.5 use the expm1 identity sinh(x) = u·(u+2) / (2·(u+1)) with
 * u = expm1(x), which keeps full precision. For |x| ≥ 0.5 the difference
 * has no harmful cancellation, but once exp(±x) saturates to ±inf (|x|
 * beyond fp64's exp range, ~709.78) the difference would form an inf−inf
 * NaN, so we defer to fp64 libm sinh there — finite up to |x| ≈ 710.48 and
 * correctly-signed ±inf beyond (the fp64 exponent-range limit shared with
 * the DD exp/log path; the magnitude is ≳1e307 so DD's extra digits are
 * moot). NaN propagates; sinh(±inf) = ±inf.
 *
 * @param[in] a Input DD value.
 * @return sinh(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_sinh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi > -0.5 && a.hi < 0.5) {
        ndarray_dd_t u   = ndarray_dd_expm1(a);
        ndarray_dd_t num = ndarray_dd_mul(u, ndarray_dd_add(u, ndarray_dd_from_double(2.0)));
        ndarray_dd_t den = ndarray_dd_mul(ndarray_dd_from_double(2.0),
                                          ndarray_dd_add(u, ndarray_dd_from_double(1.0)));
        return ndarray_dd_div(num, den);
    }
    ndarray_dd_t ex  = ndarray_dd_exp(a);
    ndarray_dd_t enx = ndarray_dd_exp(ndarray_dd_neg(a));
    if (isinf(ex.hi) || isinf(enx.hi)) return ndarray_dd_from_double(sinh(a.hi));
    return ndarray_dd_mul(ndarray_dd_sub(ex, enx), ndarray_dd_from_double(0.5));
}

/**
 * @brief DD-precision cosh(x).
 *
 * cosh(x) = (e^x + e^−x)/2. Both terms are positive so there is no
 * cancellation and cosh(0) = 1 stays exact. Once exp(±x) saturates to ±inf
 * (|x| beyond fp64's exp range, ~709.78) the sum would form an inf−inf NaN
 * via two_sum, so we defer to fp64 libm cosh there — finite up to
 * |x| ≈ 710.48, +inf beyond (the fp64 exponent-range limit shared with the
 * DD exp/log path). NaN propagates; cosh(±inf) = +inf.
 *
 * @param[in] a Input DD value.
 * @return cosh(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_cosh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    ndarray_dd_t ex  = ndarray_dd_exp(a);
    ndarray_dd_t enx = ndarray_dd_exp(ndarray_dd_neg(a));
    if (isinf(ex.hi) || isinf(enx.hi)) return ndarray_dd_from_double(cosh(a.hi));
    return ndarray_dd_mul(ndarray_dd_add(ex, enx), ndarray_dd_from_double(0.5));
}

/**
 * @brief DD-precision tanh(x).
 *
 * tanh(x) = v/(v+2) with v = expm1(2x): no cancellation near 0, and for
 * |x| ≤ 40 the argument 2x ≤ 80 keeps e^{2x} ≈ 5.5e34 ≪ DBL_MAX so there
 * is no overflow. For |x| > 40 (including ±inf) tanh saturates to ±1 to
 * below DD epsilon (1 − tanh(40) ≈ 1e-35), so return ±1 directly. NaN
 * propagates.
 *
 * @param[in] a Input DD value.
 * @return tanh(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_tanh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi >  40.0) return ndarray_dd_from_double( 1.0);
    if (a.hi < -40.0) return ndarray_dd_from_double(-1.0);
    ndarray_dd_t two = ndarray_dd_from_double(2.0);
    ndarray_dd_t v   = ndarray_dd_expm1(ndarray_dd_mul(a, two));
    return ndarray_dd_div(v, ndarray_dd_add(v, two));
}

/**
 * @brief DD-precision asinh(x) (inverse hyperbolic sine).
 *
 * asinh(x) = log1p(x + x²/(1 + sqrt(1+x²))). The x²/(1+s) term equals
 * sqrt(x²+1) − 1 without cancellation, and log1p preserves precision near
 * 0. The computation runs on |x| with the sign restored afterwards so the
 * x + sqrt(x²+1) argument never cancels for x < 0. For |x| > 1e150, where
 * x² would overflow fp64, fall back to asinh(x) ≈ ln(2|x|) = ln|x| + ln 2.
 * NaN / ±inf propagate.
 *
 * @param[in] a Input DD value.
 * @return asinh(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_arcsinh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a) || isinf(a.hi)) return a;
    int          neg = (a.hi < 0.0);
    ndarray_dd_t ax  = ndarray_dd_abs(a);
    ndarray_dd_t one = ndarray_dd_from_double(1.0);
    ndarray_dd_t r;
    if (ax.hi > 1e150) {
        r = ndarray_dd_add(ndarray_dd_log(ax), DD_LN2);
    } else {
        ndarray_dd_t t = ndarray_dd_mul(ax, ax);
        ndarray_dd_t s = ndarray_dd_sqrt(ndarray_dd_add(t, one));
        r = ndarray_dd_log1p(ndarray_dd_add(ax, ndarray_dd_div(t, ndarray_dd_add(one, s))));
    }
    return neg ? ndarray_dd_neg(r) : r;
}

/**
 * @brief DD-precision acosh(x) (inverse hyperbolic cosine).
 *
 * Domain x ≥ 1; acosh(x) = log1p((x−1) + sqrt((x−1)(x+1))). The split sqrt
 * avoids cancellation as x → 1 where acosh → 0. For x < 1 the result is
 * NaN; for x > 1e150 fall back to acosh(x) ≈ ln(2x). NaN / +inf propagate.
 *
 * @param[in] a Input DD value.
 * @return acosh(a) in DD precision, NaN when a < 1.
 */
ndarray_dd_t ndarray_dd_arccosh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    if (a.hi < 1.0)   return ndarray_dd_from_double(NAN);
    if (isinf(a.hi))  return a;
    if (a.hi > 1e150) return ndarray_dd_add(ndarray_dd_log(a), DD_LN2);
    ndarray_dd_t one  = ndarray_dd_from_double(1.0);
    ndarray_dd_t w    = ndarray_dd_sub(a, one);
    ndarray_dd_t root = ndarray_dd_mul(ndarray_dd_sqrt(w),
                                       ndarray_dd_sqrt(ndarray_dd_add(w, ndarray_dd_from_double(2.0))));
    return ndarray_dd_log1p(ndarray_dd_add(w, root));
}

/**
 * @brief DD-precision atanh(x) (inverse hyperbolic tangent).
 *
 * Domain |x| ≤ 1; atanh(x) = ½·log1p(2x/(1−x)), which avoids cancellation
 * near 0. The |x| = 1 endpoint returns ±inf (handled explicitly, since
 * 1−x = 0 would make the DD divide yield NaN); |x| > 1 returns NaN. NaN
 * propagates.
 *
 * @param[in] a Input DD value.
 * @return atanh(a) in DD precision.
 */
ndarray_dd_t ndarray_dd_arctanh(ndarray_dd_t a) {
    if (ndarray_dd_isnan(a)) return a;
    ndarray_dd_t ax = ndarray_dd_abs(a);
    if (ax.hi > 1.0) return ndarray_dd_from_double(NAN);
    if (ax.hi == 1.0 && a.lo == 0.0)
        return ndarray_dd_from_double(a.hi > 0.0 ? INFINITY : -INFINITY);
    ndarray_dd_t one = ndarray_dd_from_double(1.0);
    ndarray_dd_t num = ndarray_dd_add(a, a);
    ndarray_dd_t den = ndarray_dd_sub(one, a);
    return ndarray_dd_mul(ndarray_dd_from_double(0.5), ndarray_dd_log1p(ndarray_dd_div(num, den)));
}

/* ── int conversion ─────────────────────────────────────────────────────── */

long long ndarray_dd_to_int64(ndarray_dd_t a) {
    return (long long)a.hi + (long long)a.lo;
}

ndarray_dd_t ndarray_dd_from_int64(long long a) {
    /* int64 has 63 mantissa bits; doesn't fit in a single double (53 bits).
       Split into hi+lo. */
    double hi = (double)a;
    long long hi_int = (long long)hi;
    double lo = (double)(a - hi_int);
    double s, e;
    two_sum(hi, lo, &s, &e);
    return ndarray_dd_from_pair(s, e);
}

/* ── comparisons ───────────────────────────────────────────────────────── */

int ndarray_dd_cmp(ndarray_dd_t a, ndarray_dd_t b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return  1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return  1;
    return 0;
}

int ndarray_dd_iszero(ndarray_dd_t a) { return a.hi == 0.0 && a.lo == 0.0; }
int ndarray_dd_isnan(ndarray_dd_t a)  { return isnan(a.hi); }
int ndarray_dd_isinf(ndarray_dd_t a)  { return isinf(a.hi); }

/* ── string parser ─────────────────────────────────────────────────────── */

static int strcasematch3(const char *s, char c0, char c1, char c2) {
    return (s[0] == c0 || s[0] == (char)toupper((unsigned char)c0)) &&
           (s[1] == c1 || s[1] == (char)toupper((unsigned char)c1)) &&
           (s[2] == c2 || s[2] == (char)toupper((unsigned char)c2));
}

/* Compute 10^|n| as a DD. n must be reasonable (we don't try to handle the
   far ends of the fp range — overflow to inf or underflow to 0 is fine). */
static ndarray_dd_t dd_pow10(int n) {
    if (n == 0) return ndarray_dd_from_double(1.0);
    /* Powers of 10 up to 10^22 are exact in fp64; beyond that we accumulate
       via DD multiplication so the lo word captures the rounding error. */
    static const double POW10[23] = {
        1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10,
        1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
    };
    int an = n < 0 ? -n : n;
    ndarray_dd_t result;
    if (an <= 22) {
        result = ndarray_dd_from_double(POW10[an]);
    } else {
        result = ndarray_dd_from_double(POW10[22]);
        ndarray_dd_t ten = ndarray_dd_from_double(10.0);
        for (int i = 22; i < an; i++) {
            result = ndarray_dd_mul(result, ten);
        }
    }
    if (n < 0) {
        result = ndarray_dd_div(ndarray_dd_from_double(1.0), result);
    }
    return result;
}

ndarray_dd_t ndarray_dd_from_string(const char *str) {
    if (str == NULL) return ndarray_dd_from_double(0.0);

    while (isspace((unsigned char)*str)) str++;

    int sign = 1;
    if (*str == '+') { str++; }
    else if (*str == '-') { sign = -1; str++; }

    if (strcasematch3(str, 'i', 'n', 'f')) {
        return ndarray_dd_from_double(sign > 0 ? INFINITY : -INFINITY);
    }
    if (strcasematch3(str, 'n', 'a', 'n')) {
        return ndarray_dd_from_double(NAN);
    }

    ndarray_dd_t value = ndarray_dd_from_double(0.0);
    ndarray_dd_t ten   = ndarray_dd_from_double(10.0);
    int decimal_seen   = 0;
    int frac_digits    = 0;
    int any_digits     = 0;

    while (*str == '.' || (*str >= '0' && *str <= '9')) {
        if (*str == '.') {
            if (decimal_seen) break;
            decimal_seen = 1;
            str++;
            continue;
        }
        value = ndarray_dd_mul(value, ten);
        value = ndarray_dd_add(value, ndarray_dd_from_double((double)(*str - '0')));
        if (decimal_seen) frac_digits++;
        any_digits = 1;
        str++;
    }

    if (!any_digits) return ndarray_dd_from_double(0.0);

    int exp = 0;
    if (*str == 'e' || *str == 'E') {
        str++;
        int esign = 1;
        if (*str == '+') str++;
        else if (*str == '-') { esign = -1; str++; }
        while (*str >= '0' && *str <= '9') {
            exp = exp * 10 + (*str - '0');
            str++;
        }
        exp *= esign;
    }
    exp -= frac_digits;

    if (exp != 0) {
        value = ndarray_dd_mul(value, dd_pow10(exp));
    }
    if (sign < 0) value = ndarray_dd_neg(value);
    return value;
}

/* ── string formatter ──────────────────────────────────────────────────── */

/* Extract `n_digits` decimal digits MSB-first into `out` (no terminator).
   `a` must be in [1, 10). Returns the binary-search-corrected exponent. */
static int extract_digits(ndarray_dd_t a, char *out, int n_digits) {
    ndarray_dd_t ten = ndarray_dd_from_double(10.0);
    int exp_correction = 0;
    /* Normalize against rounding drift: when the prior log10-derived
       exponent was off by 1, a.hi can be just shy of 1.0 or just over 10.0. */
    while (a.hi >= 10.0) { a = ndarray_dd_div(a, ten); exp_correction++; }
    while (a.hi <  1.0 && !ndarray_dd_iszero(a)) {
        a = ndarray_dd_mul(a, ten); exp_correction--;
    }
    for (int i = 0; i < n_digits; i++) {
        int d = (int)a.hi;
        if (d < 0) d = 0;
        if (d > 9) d = 9;
        out[i] = (char)('0' + d);
        a = ndarray_dd_sub(a, ndarray_dd_from_double((double)d));
        a = ndarray_dd_mul(a, ten);
    }
    /* Round the last extracted digit based on the next-digit residue. */
    int next = (int)a.hi;
    if (next >= 5) {
        for (int i = n_digits - 1; i >= 0; i--) {
            if (out[i] < '9') { out[i]++; break; }
            out[i] = '0';
            if (i == 0) {
                /* Overflow into a new leading digit. */
                memmove(out + 1, out, (size_t)n_digits - 1);
                out[0] = '1';
                exp_correction++;
            }
        }
    }
    return exp_correction;
}

void ndarray_dd_to_string(ndarray_dd_t a, char *buf, size_t bufsize) {
    if (bufsize == 0) return;

    if (isnan(a.hi)) { snprintf(buf, bufsize, "nan"); return; }
    if (isinf(a.hi)) { snprintf(buf, bufsize, "%s", a.hi > 0 ? "inf" : "-inf"); return; }
    if (a.hi == 0.0 && a.lo == 0.0) { snprintf(buf, bufsize, "0"); return; }

    int sign = 0;
    if (a.hi < 0.0) { sign = 1; a = ndarray_dd_neg(a); }

    /* Initial exponent estimate from log10(hi); may be off by 1 — corrected
       inside extract_digits. */
    int exp0 = (int)floor(log10(a.hi));
    ndarray_dd_t scaled = ndarray_dd_mul(a, dd_pow10(-exp0));

    const int N = 32;            /* DD has ~32 reliable decimal digits */
    char digits[40];
    int exp = exp0 + extract_digits(scaled, digits, N);

    /* Trim trailing zeros. */
    int len = N;
    while (len > 1 && digits[len - 1] == '0') len--;

    /* Format. snprintf into a stack buffer first so the bounds are bounded. */
    char out[80];
    char *p = out;
    if (sign) *p++ = '-';

    if (exp >= -4 && exp < 21) {
        /* Plain decimal — emit the mantissa digits with the decimal point
           positioned per the exponent. */
        if (exp >= 0) {
            for (int i = 0; i <= exp; i++) {
                *p++ = (i < len) ? digits[i] : '0';
            }
            if (len > exp + 1) {
                *p++ = '.';
                for (int i = exp + 1; i < len; i++) *p++ = digits[i];
            }
        } else {
            *p++ = '0';
            *p++ = '.';
            for (int i = 0; i < -exp - 1; i++) *p++ = '0';
            for (int i = 0; i < len; i++) *p++ = digits[i];
        }
    } else {
        /* Scientific notation. */
        *p++ = digits[0];
        if (len > 1) {
            *p++ = '.';
            for (int i = 1; i < len; i++) *p++ = digits[i];
        }
        *p++ = 'e';
        int e = exp;
        if (e >= 0) { *p++ = '+'; }
        else        { *p++ = '-'; e = -e; }
        char eb[12];
        snprintf(eb, sizeof(eb), "%02d", e);
        size_t el = strlen(eb);
        memcpy(p, eb, el); p += el;
    }
    *p = '\0';

    /* Copy into caller buffer, bounded. */
    size_t out_len = (size_t)(p - out);
    if (out_len >= bufsize) out_len = bufsize - 1;
    memcpy(buf, out, out_len);
    buf[out_len] = '\0';
}
