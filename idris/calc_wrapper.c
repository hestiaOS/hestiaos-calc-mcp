/* C wrapper for Idris RefC kernel — caller-provided fixed buffer.
 *
 * Signature: int <op>_buf(const char* input, char* out, int out_len)
 *   Returns negative on buffer too small (-1), else the written length.
 *   out is always NUL-terminated on success.
 *
 * Memory: result Value is released via idris2_removeReference after
 * extracting the string into the caller buffer. No strdup, no malloc.
 */

#include <stdio.h>
#include <string.h>

#include "cBackend.h"
#include "memoryManagement.h"
#include "idris_support.h"

/* Idris-generated function declarations (module Main, from libcalc_main.idr) */
extern Value *Main_add(Value *input);
extern Value *Main_sub(Value *input);
extern Value *Main_mul(Value *input);
extern Value *Main_div(Value *input);
extern Value *Main_intpow(Value *input);
extern Value *__mainExpression_0(void);
extern Value *idris2_trampoline(Value *closure);

static int initialized = 0;

static void init_idris(void) {
    if (initialized) return;
    char *dummy_argv[] = {"libcalc", NULL};
    idris2_setArgs(1, dummy_argv);
    Value *mainExpr = __mainExpression_0();
    idris2_trampoline(mainExpr);
    initialized = 1;
}

/* Call an Idris String->String function and write result into caller buffer.
 * Returns written length (non-negative) on success, -1 on buffer too small.
 * The result Value is released after extraction to prevent runtime heap leak.
 */
static int call_idris_buf(Value *(*fn)(Value*), const char* input,
                          char* out, int out_len) {
    if (!initialized) init_idris();
    Value *idris_str = (Value *)idris2_mkString((char *)input);
    Value *closure = fn(idris_str);
    Value *result = idris2_trampoline(closure);
    const char *cstr = ((Value_String *)result)->str;
    int needed = snprintf(out, out_len, "%s", cstr);
    /* Release the result Value — decrements refcount, frees if 0 */
    idris2_removeReference(result);
    if (needed >= out_len) return -1;
    return needed;
}

/* --- Exported buffer-based functions --- */

int add_rat_buf(const char* input, char* out, int out_len) {
    return call_idris_buf(Main_add, input, out, out_len);
}

int sub_rat_buf(const char* input, char* out, int out_len) {
    return call_idris_buf(Main_sub, input, out, out_len);
}

int mul_rat_buf(const char* input, char* out, int out_len) {
    return call_idris_buf(Main_mul, input, out, out_len);
}

int div_rat_buf(const char* input, char* out, int out_len) {
    return call_idris_buf(Main_div, input, out, out_len);
}

int intpow_rat_buf(const char* input, char* out, int out_len) {
    return call_idris_buf(Main_intpow, input, out, out_len);
}
