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
