/* bafe/jit.c - JIT cache: SHA-256 + compile + dlopen
 *
 * SHA-256 implementation is a minimal self-contained version (FIPS 180-4).
 * Compilation uses `cc` from PATH. Caching is on-disk in
 * $BAFE_CACHE_DIR (default .bafecache).
 */
#define _POSIX_C_SOURCE 200809L
#include "bafe/jit.h"
#include "bafe/codegen.h"
#include "bafe/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>

/* ------------------------------------------------------------------ */
/* SHA-256 (FIPS 180-4) - minimal self-contained implementation       */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buffer[64];
    size_t   buflen;
} sha256_ctx;

static const uint32_t _SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
    0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
    0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
    0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
    0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
    0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

#define _ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void _sha256_block(sha256_ctx *c, const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
               ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = _ROTR(w[i-15], 7) ^ _ROTR(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = _ROTR(w[i-2], 17) ^ _ROTR(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c->state[0], b = c->state[1], cc = c->state[2], d = c->state[3];
    uint32_t e = c->state[4], f = c->state[5], g = c->state[6], h = c->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = _ROTR(e, 6) ^ _ROTR(e, 11) ^ _ROTR(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + _SHA_K[i] + w[i];
        uint32_t S0 = _ROTR(a, 2) ^ _ROTR(a, 13) ^ _ROTR(a, 22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->state[0] += a; c->state[1] += b; c->state[2] += cc; c->state[3] += d;
    c->state[4] += e; c->state[5] += f; c->state[6] += g; c->state[7] += h;
}

static void _sha256_init(sha256_ctx *c) {
    c->state[0] = 0x6a09e667; c->state[1] = 0xbb67ae85;
    c->state[2] = 0x3c6ef372; c->state[3] = 0xa54ff53a;
    c->state[4] = 0x510e527f; c->state[5] = 0x9b05688c;
    c->state[6] = 0x1f83d9ab; c->state[7] = 0x5be0cd19;
    c->bitlen = 0; c->buflen = 0;
}

static void _sha256_update(sha256_ctx *c, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    c->bitlen += (uint64_t)len * 8;
    while (len > 0) {
        size_t take = 64 - c->buflen;
        if (take > len) take = len;
        memcpy(c->buffer + c->buflen, p, take);
        c->buflen += take;
        p += take;
        len -= take;
        if (c->buflen == 64) {
            _sha256_block(c, c->buffer);
            c->buflen = 0;
        }
    }
}

static void _sha256_final(sha256_ctx *c, uint8_t out[32]) {
    /* pad: append 0x80, then zeros, then 64-bit length */
    c->buffer[c->buflen++] = 0x80;
    if (c->buflen > 56) {
        while (c->buflen < 64) c->buffer[c->buflen++] = 0;
        _sha256_block(c, c->buffer);
        c->buflen = 0;
    }
    while (c->buflen < 56) c->buffer[c->buflen++] = 0;
    uint64_t bitlen = c->bitlen;
    for (int i = 7; i >= 0; i--) c->buffer[c->buflen++] = (uint8_t)(bitlen >> (i * 8));
    _sha256_block(c, c->buffer);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (uint8_t)(c->state[i] >> 24);
        out[i*4+1] = (uint8_t)(c->state[i] >> 16);
        out[i*4+2] = (uint8_t)(c->state[i] >> 8);
        out[i*4+3] = (uint8_t)(c->state[i]);
    }
}

static void _hex(const uint8_t *bytes, int n, char *out) {
    static const char *hex = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        out[i*2]   = hex[(bytes[i] >> 4) & 0xf];
        out[i*2+1] = hex[bytes[i] & 0xf];
    }
    out[n*2] = '\0';
}

/* ------------------------------------------------------------------ */
/* Graph hashing                                                      */
/* ------------------------------------------------------------------ */

int bafe_jit_hash_graph(const bafe_graph *g, char *out, size_t out_size) {
    if (out_size < 65) return -1;
    sha256_ctx c; _sha256_init(&c);
    /* hash: n_inputs, n_outputs, n_nodes, then each node */
    _sha256_update(&c, &g->n_inputs, sizeof(g->n_inputs));
    _sha256_update(&c, &g->n_outputs, sizeof(g->n_outputs));
    _sha256_update(&c, &g->n_nodes, sizeof(g->n_nodes));
    for (int i = 0; i < g->n_nodes; i++) {
        const bafe_node *n = &g->nodes[i];
        _sha256_update(&c, n->op_name, strlen(n->op_name) + 1);
        _sha256_update(&c, &n->attrs, sizeof(n->attrs));
        _sha256_update(&c, &n->n_children, sizeof(n->n_children));
        _sha256_update(&c, n->children, sizeof(bafe_node_id) * (size_t)n->n_children);
        _sha256_update(&c, &n->shape, sizeof(n->shape));
        _sha256_update(&c, &n->dtype, sizeof(n->dtype));
        _sha256_update(&c, &n->layout, sizeof(n->layout));  /* Phase 2 */
        if (n->is_input) _sha256_update(&c, n->input_name, strlen(n->input_name) + 1);
        if (n->is_constant) _sha256_update(&c, &n->const_value, sizeof(n->const_value));
    }
    uint8_t digest[32];
    _sha256_final(&c, digest);
    _hex(digest, 32, out);
    return 0;
}

/* ------------------------------------------------------------------ */
/* JIT cache                                                          */
/* ------------------------------------------------------------------ */

static char  _cache_dir[1024] = "";
static bool  _cache_dir_set = false;
static bafe_jit_stats _stats = {0, 0, 0, 0};

#define BAFE_MAX_CACHED 256
typedef struct {
    char  hash[65];
    void *so_handle;
    bafe_kernel_fn fn;
} cache_entry;
static cache_entry _cache[BAFE_MAX_CACHED];
static int _cache_n = 0;

const char *bafe_jit_cache_dir(void) {
    if (_cache_dir_set) return _cache_dir;
    const char *env = getenv("BAFE_CACHE_DIR");
    if (env && *env) {
        strncpy(_cache_dir, env, sizeof(_cache_dir) - 1);
        _cache_dir[sizeof(_cache_dir) - 1] = '\0';
    } else {
        const char *home = getenv("HOME");
        if (home) {
            snprintf(_cache_dir, sizeof(_cache_dir), "%s/.bafecache", home);
        } else {
            strncpy(_cache_dir, ".bafecache", sizeof(_cache_dir) - 1);
        }
    }
    _cache_dir_set = true;
    return _cache_dir;
}

void bafe_jit_set_cache_dir(const char *dir) {
    if (!dir) return;
    strncpy(_cache_dir, dir, sizeof(_cache_dir) - 1);
    _cache_dir[sizeof(_cache_dir) - 1] = '\0';
    _cache_dir_set = true;
}

static void _ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) return;
    mkdir(path, 0755);
}

bafe_kernel_fn bafe_jit_get_or_compile(const bafe_graph *g,
                                        char *err_buf, size_t err_buf_size) {
    if (err_buf && err_buf_size > 0) err_buf[0] = '\0';

    char hash[65];
    if (bafe_jit_hash_graph(g, hash, sizeof(hash)) < 0) {
        if (err_buf) snprintf(err_buf, err_buf_size, "hash failed");
        return NULL;
    }

    /* in-memory cache */
    for (int i = 0; i < _cache_n; i++) {
        if (strcmp(_cache[i].hash, hash) == 0) {
            _stats.hits++;
            return _cache[i].fn;
        }
    }
    _stats.misses++;

    /* on-disk cache */
    const char *dir = bafe_jit_cache_dir();
    _ensure_dir(dir);
    char so_path[1280], c_path[1280];
    snprintf(so_path, sizeof(so_path), "%s/%s.so", dir, hash);
    snprintf(c_path,  sizeof(c_path),  "%s/%s.c",  dir, hash);

    /* check if .so exists and is loadable */
    struct stat st;
    void *handle = NULL;
    bafe_kernel_fn fn = NULL;
    bool need_compile = true;
    if (stat(so_path, &st) == 0) {
        handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            fn = (bafe_kernel_fn)dlsym(handle, "bafe_kernel");
            if (fn) {
                need_compile = false;
                _stats.hits++;  /* count on-disk cache hits */
            }
        }
    }

    if (need_compile) {
        _stats.compiles++;
        /* emit C source */
        char *src = bafe_codegen_emit_alloc(g, "bafe_kernel");
        if (!src) {
            if (err_buf) snprintf(err_buf, err_buf_size, "codegen failed");
            _stats.compile_failures++;
            return NULL;
        }
        /* write to .c file */
        FILE *f = fopen(c_path, "w");
        if (!f) {
            free(src);
            if (err_buf) snprintf(err_buf, err_buf_size, "cannot write %s", c_path);
            _stats.compile_failures++;
            return NULL;
        }
        fputs(src, f);
        fclose(f);
        free(src);

        /* compile: cc -shared -fPIC -O2 -o <so> <c> -lm */
        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "cc -shared -fPIC -O2 -std=c11 -o %s %s -lm 2>&1",
                 so_path, c_path);
        FILE *pipe = popen(cmd, "r");
        if (!pipe) {
            if (err_buf) snprintf(err_buf, err_buf_size, "popen failed");
            _stats.compile_failures++;
            return NULL;
        }
        char compile_err[1024] = {0};
        size_t ep = 0;
        char line[256];
        while (fgets(line, sizeof(line), pipe)) {
            if (ep + strlen(line) < sizeof(compile_err)) {
                strcpy(compile_err + ep, line);
                ep += strlen(line);
            }
        }
        int rc = pclose(pipe);
        if (rc != 0) {
            if (err_buf) snprintf(err_buf, err_buf_size, "compile failed: %s", compile_err);
            _stats.compile_failures++;
            return NULL;
        }

        /* dlopen */
        handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            if (err_buf) snprintf(err_buf, err_buf_size, "dlopen failed: %s", dlerror());
            _stats.compile_failures++;
            return NULL;
        }
        fn = (bafe_kernel_fn)dlsym(handle, "bafe_kernel");
        if (!fn) {
            if (err_buf) snprintf(err_buf, err_buf_size, "dlsym failed: %s", dlerror());
            dlclose(handle);
            _stats.compile_failures++;
            return NULL;
        }
    }

    /* store in memory cache */
    if (_cache_n < BAFE_MAX_CACHED) {
        strncpy(_cache[_cache_n].hash, hash, 64);
        _cache[_cache_n].hash[64] = '\0';
        _cache[_cache_n].so_handle = handle;
        _cache[_cache_n].fn = fn;
        _cache_n++;
    }
    return fn;
}

bafe_jit_stats bafe_jit_get_stats(void) {
    return _stats;
}

void bafe_jit_clear(void) {
    for (int i = 0; i < _cache_n; i++) {
        if (_cache[i].so_handle) dlclose(_cache[i].so_handle);
    }
    _cache_n = 0;
    memset(&_stats, 0, sizeof(_stats));
}

void bafe_jit_invalidate_memory_cache(void) {
    /* Close the dlopen'd handles but keep the on-disk .so files.
     * The next bafe_jit_get_or_compile call will re-optimize (picking
     * up the calibrated cost model) and may produce a different graph,
     * which either hits the on-disk cache for the SAME hash or compiles
     * a NEW .so for the new graph. */
    for (int i = 0; i < _cache_n; i++) {
        if (_cache[i].so_handle) dlclose(_cache[i].so_handle);
    }
    _cache_n = 0;
    /* Note: we do NOT reset _stats here — hits/misses/compiles are
     * cumulative across invalidations. */
}
