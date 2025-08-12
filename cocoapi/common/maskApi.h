/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
#pragma once
#include <stdint.h>

typedef struct {
        uint64_t  h, w, m;
        uint32_t* cnts;
} RLE;

/* Initialize/destroy RLE. */
void rleInit(RLE* R, uint64_t h, uint64_t w, uint64_t m, uint32_t* cnts);
void rleFree(RLE* R);

/* Initialize/destroy RLE array. */
void rlesInit(RLE** R, uint64_t n);
void rlesFree(RLE** R, uint64_t n);

/* Encode binary masks using RLE. */
void rleEncode(RLE* R, const uint8_t* mask, uint64_t h, uint64_t w, uint64_t n);

/* Decode binary masks encoded via RLE. */
void rleDecode(const RLE* R, uint8_t* mask, uint64_t n);

/* Compute union or intersection of encoded masks. */
void rleMerge(const RLE* R, RLE* M, uint64_t n, int intersect);

/* Compute area of encoded masks. */
void rleArea(const RLE* R, uint64_t n, uint32_t* a);

/* Compute intersection over union between masks. */
void rleIou(RLE* dt, RLE* gt, uint64_t m, uint64_t n, uint8_t* iscrowd, double* o);

/* Compute non-maximum suppression between bounding masks */
void rleNms(RLE* dt, uint64_t n, uint32_t* keep, double thr);

/* Compute intersection over union between bounding boxes. */
void bbIou(double* dt, double* gt, uint64_t m, uint64_t n, uint8_t* iscrowd, double* o);

/* Compute non-maximum suppression between bounding boxes */
void bbNms(double* dt, uint64_t n, uint32_t* keep, double thr);

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const RLE* R, double* bb, uint64_t n);

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(RLE* R, const double* bb, uint64_t h, uint64_t w, uint64_t n);

/* Convert polygon to encoded mask. */
void rleFrPoly(RLE* R, const double* xy, uint64_t k, uint64_t h, uint64_t w);

/* Get compressed string representation of encoded mask. */
char* rleToString(const RLE* R);

/* Convert from compressed string representation of encoded mask. */
void rleFrString(RLE* R, char* s, uint64_t h, uint64_t w);
