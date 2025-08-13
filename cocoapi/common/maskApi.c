/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "maskApi.h"

uint32_t umin(_In_ uint32_t a, _In_ uint32_t b) { return (a < b) ? a : b; }

uint32_t umax(_In_ uint32_t a, _In_ uint32_t b) { return (a > b) ? a : b; }

void rleInit(_Inout_ RLE* const R, uint64_t h, uint64_t w, uint64_t m, _In_ const uint32_t* const cnts) {
    R->h    = h;
    R->w    = w;
    R->m    = m;
    R->cnts = !m ? NULL : malloc(sizeof(uint32_t) * m);

    if (cnts)
        for (uint64_t j = 0; j < m; j++) R->cnts[j] = cnts[j];
}

void rleFree(_Inout_ RLE* const R) {
    free(R->cnts);
    R->cnts = NULL;
}

void rlesInit(RLE** R, uint64_t n) {
    *R = malloc(sizeof(RLE) * n);
    for (uint64_t i = 0; i < n; i++) rleInit((*R) + i, 0, 0, 0, 0);
}

void rlesFree(RLE** R, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) rleFree((*R) + i);
    free(*R);
    *R = NULL;
}

void rleEncode(RLE* R, const uint8_t* M, uint64_t h, uint64_t w, uint64_t n) {
    uint64_t i, j, k, a = w * h;
    uint32_t c, *cnts;
    uint8_t  p;
    cnts = malloc(sizeof(uint32_t) * (a + 1));
    for (i = 0; i < n; i++) {
        const uint8_t* T = M + a * i;
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

void rleDecode(const RLE* R, uint8_t* M, uint64_t n) {
    uint64_t i, j, k;
    for (i = 0; i < n; i++) {
        uint8_t v = 0;
        for (j = 0; j < R[i].m; j++) {
            for (k = 0; k < R[i].cnts[j]; k++) *(M++) = v;
            v = !v;
        }
    }
}

void rleMerge(const RLE* R, RLE* M, uint64_t n, int intersect) {
    uint32_t *cnts, c, ca, cb, cc, ct;
    int       v, va, vb, vp;
    uint64_t  i, a, b, h = R[0].h, w = R[0].w, m = R[0].m;
    RLE       A, B;
    if (n == 0) {
        rleInit(M, 0, 0, 0, 0);
        return;
    }
    if (n == 1) {
        rleInit(M, h, w, m, R[0].cnts);
        return;
    }
    cnts = malloc(sizeof(uint32_t) * (h * w + 1));
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

void rleArea(const RLE* R, uint64_t n, uint32_t* a) {
    uint64_t i, j;
    for (i = 0; i < n; i++) {
        a[i] = 0;
        for (j = 1; j < R[i].m; j += 2) a[i] += R[i].cnts[j];
    }
}

void rleIou(RLE* dt, RLE* gt, uint64_t m, uint64_t n, uint8_t* iscrowd, double* o) {
    uint64_t g, d;
    double * db, gb;
    int      crowd;
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
                uint64_t ka, kb, a, b;
                uint32_t c, ca, cb, ct, i, u;
                int      va, vb;
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

void rleNms(RLE* dt, uint64_t n, uint32_t* keep, double thr) {
    uint64_t i, j;
    double   u;
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

void bbIou(double* dt, double* gt, uint64_t m, uint64_t n, uint8_t* iscrowd, double* o) {
    double   h, w, i, u, ga, da;
    uint64_t g, d;
    int      crowd;
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

void bbNms(double* dt, uint64_t n, uint32_t* keep, double thr) {
    uint64_t i, j;
    double   u;
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

void rleToBbox(const RLE* R, double* bb, uint64_t n) {
    uint64_t i;
    for (i = 0; i < n; i++) {
        uint32_t h, w, x, y, xs, ys, xe, ye, xp, cc, t;
        uint64_t j, m;
        h  = (uint32_t) R[i].h;
        w  = (uint32_t) R[i].w;
        m  = R[i].m;
        m  = ((uint64_t) (m / 2)) * 2;
        xs = w;
        ys = h;
        xe = ye = 0;
        cc      = 0;
        if (m == 0) {
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

void rleFrBbox(RLE* R, const double* bb, uint64_t h, uint64_t w, uint64_t n) {
    uint64_t i;
    for (i = 0; i < n; i++) {
        double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
        double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
        double xy[8] = { xs, ys, xs, ye, xe, ye, xe, ys };
        rleFrPoly(R + i, xy, 4, h, w);
    }
}

int uintCompare(_In_ const void* const a, _In_ const void* const b) {
    uint32_t c = *((uint32_t*) a), d = *((uint32_t*) b);
    return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(RLE* R, const double* xy, uint64_t k, uint64_t h, uint64_t w) {
    /* upsample and get discrete points densely along entire boundary */
    uint64_t  j, m = 0;
    double    scale = 5;
    int *     x, *y, *u, *v;
    uint32_t *a, *b;
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
    a = malloc(sizeof(uint32_t) * (k + 1));
    for (j = 0; j < k; j++) a[j] = (uint32_t) (x[j] * (int) (h) + y[j]);
    a[k++] = (uint32_t) (h * w);
    free(u);
    free(v);
    free(x);
    free(y);
    qsort(a, k, sizeof(uint32_t), uintCompare);
    uint32_t p = 0;
    for (j = 0; j < k; j++) {
        uint32_t t  = a[j];
        a[j]       -= p;
        p           = t;
    }
    b = malloc(sizeof(uint32_t) * k);
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

char* rleToString(const RLE* R) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    uint64_t i, m = R->m, p = 0;
    long     x;
    int      more;
    char*    s = malloc(sizeof(char) * m * 6);
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

void rleFrString(RLE* R, char* s, uint64_t h, uint64_t w) {
    uint64_t  m = 0, p = 0, k;
    long      x;
    int       more;
    uint32_t* cnts;
    while (s[m]) m++;
    cnts = malloc(sizeof(uint32_t) * m);
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
        cnts[m++] = (uint32_t) x;
    }
    rleInit(R, h, w, m, cnts);
    free(cnts);
}
