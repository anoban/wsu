/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/

// clang .\common\maskApi.c -Wall -Wextra -Werror -O3 -static -std=c23

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if (!defined _WIN32) || (!defined _WIN64)
    #error This source has been customized only for Win32 systems!
#endif

typedef struct RLE {
        unsigned long long h, w, m;
        unsigned*          cnts;
} RLE;

static inline int __stdcall uintCompare(_In_ const void* const a, _In_ const void* const b) {
    unsigned c = *((unsigned*) a), d = *((unsigned*) b);
    return c > d ? 1 : c < d ? -1 : 0;
}

static inline unsigned __stdcall umin(_In_ unsigned a, _In_ unsigned b) { return (a < b) ? a : b; }

static inline unsigned __stdcall umax(_In_ unsigned a, _In_ unsigned b) { return (a > b) ? a : b; }

/* Initialize RLE. */
void __stdcall rleInit(
    _Inout_ RLE* const R, unsigned long long h, unsigned long long w, unsigned long long m, _In_ const unsigned* const cnts
) {
    R->h    = h;
    R->w    = w;
    R->m    = m;
    R->cnts = !m ? NULL : malloc(sizeof(unsigned) * m);
    assert(R->cnts); //

    if (cnts)
        for (unsigned long long j = 0; j < m; j++) R->cnts[j] = cnts[j];
}

/* destroy RLE. */
void __stdcall rleFree(_Inout_ RLE* const R) {
    free(R->cnts);
    R->cnts = NULL;
}

/* Initialize RLE array. */
void __stdcall rlesInit(RLE** R, unsigned long long n) {
    *R = malloc(sizeof(RLE) * n);
    for (unsigned long long i = 0; i < n; i++) rleInit((*R) + i, 0, 0, 0, 0);
}

/* destroy RLE array. */
void rlesFree(RLE** R, unsigned long long n) {
    for (unsigned long long i = 0; i < n; i++) rleFree((*R) + i);
    free(*R);
    *R = NULL;
}

/* Encode binary masks using RLE. */
void rleEncode(RLE* R, const unsigned char* M, unsigned long long h, unsigned long long w, unsigned long long n) {
    unsigned long long i, j, k, a = w * h;
    unsigned           c, *cnts;
    unsigned char      p;
    cnts = malloc(sizeof(unsigned) * (a + 1));
    for (i = 0; i < n; i++) {
        const unsigned char* T = M + a * i;
        k = p = c = 0;
        for (j = 0; j < a; j++) {
            if (T[j] != p) {
                cnts[k++] = c;
                c         = 0;
                p         = T[j];
            }
            c++;
        }
        cnts[k++] = c;
        rleInit(R + i, h, w, k, cnts);
    }
    free(cnts);
}

/* Decode binary masks encoded via RLE. */
void rleDecode(const RLE* R, unsigned char* M, unsigned long long n) {
    unsigned long long i, j, k;
    for (i = 0; i < n; i++) {
        unsigned char v = 0;
        for (j = 0; j < R[i].m; j++) {
            for (k = 0; k < R[i].cnts[j]; k++) *(M++) = v;
            v = !v;
        }
    }
}

/* Compute union or intersection of encoded masks. */
void rleMerge(const RLE* R, RLE* M, unsigned long long n, int intersect) {
    unsigned *         cnts, c, ca, cb, cc, ct;
    int                v, va, vb, vp;
    unsigned long long i, a, b, h = R[0].h, w = R[0].w, m = R[0].m;
    RLE                A, B;
    if (n == 0) {
        rleInit(M, 0, 0, 0, 0);
        return;
    }
    if (n == 1) {
        rleInit(M, h, w, m, R[0].cnts);
        return;
    }
    cnts = malloc(sizeof(unsigned) * (h * w + 1));
    for (a = 0; a < m; a++) cnts[a] = R[0].cnts[a];
    for (i = 1; i < n; i++) {
        B = R[i];
        if (B.h != h || B.w != w) {
            h = w = m = 0;
            break;
        }
        rleInit(&A, h, w, m, cnts);
        ca = A.cnts[0];
        cb = B.cnts[0];
        v = va = vb = 0;
        m           = 0;
        a = b = 1;
        cc    = 0;
        ct    = 1;
        while (ct > 0) {
            c   = umin(ca, cb);
            cc += c;
            ct  = 0;
            ca -= c;
            if (!ca && a < A.m) {
                ca = A.cnts[a++];
                va = !va;
            }
            ct += ca;
            cb -= c;
            if (!cb && b < B.m) {
                cb = B.cnts[b++];
                vb = !vb;
            }
            ct += cb;
            vp  = v;
            if (intersect)
                v = va && vb;
            else
                v = va || vb;
            if (v != vp || ct == 0) {
                cnts[m++] = cc;
                cc        = 0;
            }
        }
        rleFree(&A);
    }
    rleInit(M, h, w, m, cnts);
    free(cnts);
}

/* Compute area of encoded masks. */
void rleArea(const RLE* R, unsigned long long n, unsigned* a) {
    unsigned long long i, j;
    for (i = 0; i < n; i++) {
        a[i] = 0;
        for (j = 1; j < R[i].m; j += 2) a[i] += R[i].cnts[j];
    }
}

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const RLE* R, double* bb, unsigned long long n) {
    unsigned long long i;
    for (i = 0; i < n; i++) {
        unsigned           h, w, x, y, xs, ys, xe, ye, xp, cc, t;
        unsigned long long j, m;
        h  = (unsigned) R[i].h;
        w  = (unsigned) R[i].w;
        m  = R[i].m;
        m  = ((unsigned long long) (m / 2)) * 2;
        xs = w;
        ys = h;
        xe = ye = 0;
        cc      = 0;
        if (!m) {
            bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0;
            continue;
        }
        for (j = 0; j < m; j++) {
            cc += R[i].cnts[j];
            t   = cc - j % 2;
            y   = t % h;
            x   = (t - y) / h;
            if (j % 2 == 0)
                xp = x;
            else if (xp < x) {
                ys = 0;
                ye = h - 1;
            }
            xs = umin(xs, x);
            xe = umax(xe, x);
            ys = umin(ys, y);
            ye = umax(ye, y);
        }
        bb[4 * i + 0] = xs;
        bb[4 * i + 2] = xe - xs + 1;
        bb[4 * i + 1] = ys;
        bb[4 * i + 3] = ye - ys + 1;
    }
}

/* Compute intersection over union between bounding boxes. */
void bbIou(double* dt, double* gt, unsigned long long m, unsigned long long n, unsigned char* iscrowd, double* o) {
    double             h, w, i, u, ga, da;
    unsigned long long g, d;
    int                crowd;
    for (g = 0; g < n; g++) {
        double* G = gt + g * 4;
        ga        = G[2] * G[3];
        crowd     = iscrowd != NULL && iscrowd[g];
        for (d = 0; d < m; d++) {
            double* D    = dt + d * 4;
            da           = D[2] * D[3];
            o[g * m + d] = 0;
            w            = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
            if (w <= 0) continue;
            h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
            if (h <= 0) continue;
            i            = w * h;
            u            = crowd ? da : da + ga - i;
            o[g * m + d] = i / u;
        }
    }
}

/* Compute intersection over union between masks. */
void rleIou(RLE* dt, RLE* gt, unsigned long long m, unsigned long long n, unsigned char* iscrowd, double* o) {
    unsigned long long g, d;
    double *           db, *gb;
    int                crowd;
    db = malloc(sizeof(double) * m * 4);
    rleToBbox(dt, db, m);
    gb = malloc(sizeof(double) * n * 4);
    rleToBbox(gt, gb, n);
    bbIou(db, gb, m, n, iscrowd, o);
    free(db);
    free(gb);
    for (g = 0; g < n; g++)
        for (d = 0; d < m; d++)
            if (o[g * m + d] > 0) {
                crowd = iscrowd != NULL && iscrowd[g];
                if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
                    o[g * m + d] = -1;
                    continue;
                }
                unsigned long long ka, kb, a, b;
                unsigned           c, ca, cb, ct, i, u;
                int                va, vb;
                ca = dt[d].cnts[0];
                ka = dt[d].m;
                va = vb = 0;
                cb      = gt[g].cnts[0];
                kb      = gt[g].m;
                a = b = 1;
                i = u = 0;
                ct    = 1;
                while (ct > 0) {
                    c = umin(ca, cb);
                    if (va || vb) {
                        u += c;
                        if (va && vb) i += c;
                    }
                    ct  = 0;
                    ca -= c;
                    if (!ca && a < ka) {
                        ca = dt[d].cnts[a++];
                        va = !va;
                    }
                    ct += ca;
                    cb -= c;
                    if (!cb && b < kb) {
                        cb = gt[g].cnts[b++];
                        vb = !vb;
                    }
                    ct += cb;
                }
                if (i == 0)
                    u = 1;
                else if (crowd)
                    rleArea(dt + d, 1, &u);
                o[g * m + d] = (double) i / (double) u;
            }
}

/* Compute non-maximum suppression between bounding masks */
void rleNms(RLE* dt, unsigned long long n, unsigned* keep, double thr) {
    unsigned long long i, j;
    double             u;
    for (i = 0; i < n; i++) keep[i] = 1;
    for (i = 0; i < n; i++)
        if (keep[i]) {
            for (j = i + 1; j < n; j++)
                if (keep[j]) {
                    rleIou(dt + i, dt + j, 1, 1, 0, &u);
                    if (u > thr) keep[j] = 0;
                }
        }
}

/* Compute non-maximum suppression between bounding boxes */
void bbNms(double* dt, unsigned long long n, unsigned* keep, double thr) {
    unsigned long long i = 0, j = 0;
    double             u = 0.000;
    for (i = 0; i < n; i++) keep[i] = 1;
    for (i = 0; i < n; i++)
        if (keep[i]) {
            for (j = i + 1; j < n; j++)
                if (keep[j]) {
                    bbIou(dt + i * 4, dt + j * 4, 1, 1, 0, &u);
                    if (u > thr) keep[j] = 0;
                }
        }
}

/* Convert polygon to encoded mask. */
void rleFrPoly(RLE* R, const double* xy, unsigned long long k, unsigned long long h, unsigned long long w) {
    /* upsample and get discrete points densely along entire boundary */
    unsigned long long j, m = 0;
    double             scale = 5;
    int *              x, *y, *u, *v;
    unsigned *         a, *b;
    x = malloc(sizeof(int) * (k + 1));
    y = malloc(sizeof(int) * (k + 1));
    for (j = 0; j < k; j++) x[j] = (int) (scale * xy[j * 2 + 0] + .5);
    x[k] = x[0];
    for (j = 0; j < k; j++) y[j] = (int) (scale * xy[j * 2 + 1] + .5);
    y[k] = y[0];
    for (j = 0; j < k; j++) m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
    u = malloc(sizeof(int) * m);
    v = malloc(sizeof(int) * m);
    m = 0;
    for (j = 0; j < k; j++) {
        int    xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
        int    flip;
        double s;
        dx   = abs(xe - xs);
        dy   = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {
            t  = xs;
            xs = xe;
            xe = t;
            t  = ys;
            ys = ye;
            ye = t;
        }
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        if (dx >= dy)
            for (d = 0; d <= dx; d++) {
                t    = flip ? dx - d : d;
                u[m] = t + xs;
                v[m] = (int) (ys + s * t + .5);
                m++;
            }
        else
            for (d = 0; d <= dy; d++) {
                t    = flip ? dy - d : d;
                v[m] = t + ys;
                u[m] = (int) (xs + s * t + .5);
                m++;
            }
    }
    /* get points along y-boundary and downsample */
    free(x);
    free(y);
    k = m;
    m = 0;
    double xd, yd;
    x = malloc(sizeof(int) * k);
    y = malloc(sizeof(int) * k);
    for (j = 1; j < k; j++)
        if (u[j] != u[j - 1]) {
            xd = (double) (u[j] < u[j - 1] ? u[j] : u[j] - 1);
            xd = (xd + .5) / scale - .5;
            if (floor(xd) != xd || xd < 0 || xd > w - 1) continue;
            yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]);
            yd = (yd + .5) / scale - .5;
            if (yd < 0)
                yd = 0;
            else if (yd > h)
                yd = h;
            yd   = ceil(yd);
            x[m] = (int) xd;
            y[m] = (int) yd;
            m++;
        }
    /* compute rle encoding given y-boundary points */
    k = m;
    a = malloc(sizeof(unsigned) * (k + 1));
    for (j = 0; j < k; j++) a[j] = (unsigned) (x[j] * (int) (h) + y[j]);
    a[k++] = (unsigned) (h * w);
    free(u);
    free(v);
    free(x);
    free(y);
    qsort(a, k, sizeof(unsigned), uintCompare);
    unsigned p = 0;
    for (j = 0; j < k; j++) {
        unsigned t  = a[j];
        a[j]       -= p;
        p           = t;
    }
    b = malloc(sizeof(unsigned) * k);
    j = m  = 0;
    b[m++] = a[j++];
    while (j < k)
        if (a[j] > 0)
            b[m++] = a[j++];
        else {
            j++;
            if (j < k) b[m - 1] += a[j++];
        }
    rleInit(R, h, w, m, b);
    free(a);
    free(b);
}

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(RLE* R, const double* bb, unsigned long long h, unsigned long long w, unsigned long long n) {
    unsigned long long i;
    for (i = 0; i < n; i++) {
        double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
        double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
        double xy[8] = { xs, ys, xs, ye, xe, ye, xe, ys };
        rleFrPoly(R + i, xy, 4, h, w);
    }
}

/* Get compressed string representation of encoded mask. */
char* rleToString(const RLE* R) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    unsigned long long i, m = R->m, p = 0;
    long               x;
    int                more;
    char*              s = malloc(sizeof(char) * m * 6);
    for (i = 0; i < m; i++) {
        x = (long) R->cnts[i];
        if (i > 2) x -= (long) R->cnts[i - 2];
        more = 1;
        while (more) {
            char c   = x & 0x1f;
            x      >>= 5;
            more     = (c & 0x10) ? x != -1 : x != 0;
            if (more) c |= 0x20;
            c      += 48;
            s[p++]  = c;
        }
    }
    s[p] = 0;
    return s;
}

/* Convert from compressed string representation of encoded mask. */
void rleFrString(RLE* R, char* s, unsigned long long h, unsigned long long w) {
    unsigned long long m = 0, p = 0, k;
    long               x;
    int                more;
    unsigned*          cnts;
    while (s[m]) m++;
    cnts = malloc(sizeof(unsigned) * m);
    m    = 0;
    while (s[p]) {
        x    = 0;
        k    = 0;
        more = 1;
        while (more) {
            char c  = s[p] - 48;
            x      |= (c & 0x1f) << 5 * k;
            more    = c & 0x20;
            p++;
            k++;
            if (!more && (c & 0x10)) x |= -1 << 5 * k;
        }
        if (m > 2) x += (long) cnts[m - 2];
        cnts[m++] = (unsigned) x;
    }
    rleInit(R, h, w, m, cnts);
    free(cnts);
}

int main(void) {
    //
    _putws(L"Hellooooooo! It's been a while since I last used C!");
    return EXIT_SUCCESS;
}
