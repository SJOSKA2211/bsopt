let wasm;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export3(addHeapObject(e));
    }
}

let heap = new Array(128).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

const AmericanOptionsWASMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_americanoptionswasm_free(ptr >>> 0, 1));

const BlackScholesWASMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_blackscholeswasm_free(ptr >>> 0, 1));

const CrankNicolsonWASMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_cranknicolsonwasm_free(ptr >>> 0, 1));

const GreeksFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_greeks_free(ptr >>> 0, 1));

const HestonWASMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_hestonwasm_free(ptr >>> 0, 1));

const MonteCarloWASMFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_montecarlowasm_free(ptr >>> 0, 1));

export class AmericanOptionsWASM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        AmericanOptionsWASMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_americanoptionswasm_free(ptr, 0);
    }
    constructor() {
        const ret = wasm.americanoptionswasm_new();
        this.__wbg_ptr = ret >>> 0;
        AmericanOptionsWASMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Longstaff-Schwartz (LSM) implementation for American Options in WASM.
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} num_paths
     * @param {number} num_steps
     * @returns {number}
     */
    price_lsm(spot, strike, time, vol, rate, div, is_call, num_paths, num_steps) {
        const ret = wasm.americanoptionswasm_price_lsm(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, num_paths, num_steps);
        return ret;
    }
}
if (Symbol.dispose) AmericanOptionsWASM.prototype[Symbol.dispose] = AmericanOptionsWASM.prototype.free;

export class BlackScholesWASM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        BlackScholesWASMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_blackscholeswasm_free(ptr, 0);
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @returns {number}
     */
    price_call(spot, strike, time, vol, rate, div) {
        const ret = wasm.blackscholeswasm_price_call(this.__wbg_ptr, spot, strike, time, vol, rate, div);
        return ret;
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} m
     * @param {number} n
     * @returns {number}
     */
    price_american(spot, strike, time, vol, rate, div, is_call, m, n) {
        const ret = wasm.blackscholeswasm_price_american(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, m, n);
        return ret;
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} r
     * @param {number} v0
     * @param {number} kappa
     * @param {number} theta
     * @param {number} sigma
     * @param {number} rho
     * @param {boolean} is_call
     * @param {number} num_paths
     * @returns {number}
     */
    price_heston_mc(spot, strike, time, r, v0, kappa, theta, sigma, rho, is_call, num_paths) {
        const ret = wasm.blackscholeswasm_price_heston_mc(this.__wbg_ptr, spot, strike, time, r, v0, kappa, theta, sigma, rho, is_call, num_paths);
        return ret;
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @returns {Greeks}
     */
    calculate_greeks(spot, strike, time, vol, rate, div) {
        const ret = wasm.blackscholeswasm_calculate_greeks(this.__wbg_ptr, spot, strike, time, vol, rate, div);
        return Greeks.__wrap(ret);
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} num_paths
     * @returns {number}
     */
    price_monte_carlo(spot, strike, time, vol, rate, div, is_call, num_paths) {
        const ret = wasm.blackscholeswasm_price_monte_carlo(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, num_paths);
        return ret;
    }
    /**
     * @param {Float64Array} params
     * @returns {Float64Array}
     */
    batch_price_heston(params) {
        const ptr0 = passArrayF64ToWasm0(params, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_price_heston(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} num_paths
     * @param {number} num_steps
     * @returns {number}
     */
    price_american_lsm(spot, strike, time, vol, rate, div, is_call, num_paths, num_steps) {
        const ret = wasm.blackscholeswasm_price_american_lsm(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, num_paths, num_steps);
        return ret;
    }
    /**
     * @param {any} params
     * @returns {any}
     */
    batch_calculate_soa(params) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.blackscholeswasm_batch_calculate_soa(retptr, this.__wbg_ptr, addHeapObject(params));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * SIMD-accelerated batch calculation for Black-Scholes.
     * Processes 2 options at a time using f64x2 SIMD (v128) intrinsics.
     * @param {Float64Array} params
     * @returns {Float64Array}
     */
    batch_calculate_simd(params) {
        const ptr0 = passArrayF64ToWasm0(params, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_calculate_simd(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
    /**
     * @param {Float64Array} params
     * @returns {Float64Array}
     */
    batch_calculate_view(params) {
        const ptr0 = passArrayF64ToWasm0(params, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_calculate_view(this.__wbg_ptr, ptr0, len0);
        return takeObject(ret);
    }
    /**
     * @param {Float64Array} params
     * @param {number} m
     * @param {number} n
     * @returns {Float64Array}
     */
    batch_price_american(params, m, n) {
        const ptr0 = passArrayF64ToWasm0(params, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_price_american(this.__wbg_ptr, ptr0, len0, m, n);
        return takeObject(ret);
    }
    /**
     * @param {Float64Array} params
     * @param {number} num_paths
     * @returns {Float64Array}
     */
    batch_price_monte_carlo(params, num_paths) {
        const ptr0 = passArrayF64ToWasm0(params, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_price_monte_carlo(this.__wbg_ptr, ptr0, len0, num_paths);
        return takeObject(ret);
    }
    /**
     * Highly optimized batch calculation using SIMD, Rayon, and manual prefetching.
     * @param {Float64Array} spots
     * @param {Float64Array} strikes
     * @param {Float64Array} times
     * @param {Float64Array} vols
     * @param {Float64Array} rates
     * @param {Float64Array} divs
     * @param {Float64Array} are_calls
     * @returns {Float64Array}
     */
    batch_calculate_soa_compact(spots, strikes, times, vols, rates, divs, are_calls) {
        const ptr0 = passArrayF64ToWasm0(spots, wasm.__wbindgen_export);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(strikes, wasm.__wbindgen_export);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(times, wasm.__wbindgen_export);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF64ToWasm0(vols, wasm.__wbindgen_export);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passArrayF64ToWasm0(rates, wasm.__wbindgen_export);
        const len4 = WASM_VECTOR_LEN;
        const ptr5 = passArrayF64ToWasm0(divs, wasm.__wbindgen_export);
        const len5 = WASM_VECTOR_LEN;
        const ptr6 = passArrayF64ToWasm0(are_calls, wasm.__wbindgen_export);
        const len6 = WASM_VECTOR_LEN;
        const ret = wasm.blackscholeswasm_batch_calculate_soa_compact(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4, ptr5, len5, ptr6, len6);
        return takeObject(ret);
    }
    constructor() {
        const ret = wasm.blackscholeswasm_new();
        this.__wbg_ptr = ret >>> 0;
        BlackScholesWASMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} price
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @returns {number}
     */
    solve_iv(price, spot, strike, time, rate, div, is_call) {
        const ret = wasm.blackscholeswasm_solve_iv(this.__wbg_ptr, price, spot, strike, time, rate, div, is_call);
        return ret;
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @returns {number}
     */
    price_put(spot, strike, time, vol, rate, div) {
        const ret = wasm.blackscholeswasm_price_put(this.__wbg_ptr, spot, strike, time, vol, rate, div);
        return ret;
    }
}
if (Symbol.dispose) BlackScholesWASM.prototype[Symbol.dispose] = BlackScholesWASM.prototype.free;

export class CrankNicolsonWASM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CrankNicolsonWASMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_cranknicolsonwasm_free(ptr, 0);
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} m
     * @param {number} n
     * @returns {number}
     */
    price_american(spot, strike, time, vol, rate, div, is_call, m, n) {
        const ret = wasm.cranknicolsonwasm_price_american(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, m, n);
        return ret;
    }
    constructor() {
        const ret = wasm.americanoptionswasm_new();
        this.__wbg_ptr = ret >>> 0;
        CrankNicolsonWASMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) CrankNicolsonWASM.prototype[Symbol.dispose] = CrankNicolsonWASM.prototype.free;

export class Greeks {
    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(Greeks.prototype);
        obj.__wbg_ptr = ptr;
        GreeksFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        GreeksFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_greeks_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get delta() {
        const ret = wasm.__wbg_get_greeks_delta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set delta(arg0) {
        wasm.__wbg_set_greeks_delta(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get gamma() {
        const ret = wasm.__wbg_get_greeks_gamma(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set gamma(arg0) {
        wasm.__wbg_set_greeks_gamma(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get vega() {
        const ret = wasm.__wbg_get_greeks_vega(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set vega(arg0) {
        wasm.__wbg_set_greeks_vega(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get theta() {
        const ret = wasm.__wbg_get_greeks_theta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set theta(arg0) {
        wasm.__wbg_set_greeks_theta(this.__wbg_ptr, arg0);
    }
    /**
     * @returns {number}
     */
    get rho() {
        const ret = wasm.__wbg_get_greeks_rho(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set rho(arg0) {
        wasm.__wbg_set_greeks_rho(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Greeks.prototype[Symbol.dispose] = Greeks.prototype.free;

export class HestonWASM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        HestonWASMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_hestonwasm_free(ptr, 0);
    }
    /**
     * Price European call using Carr-Madan with Simpson's Rule in WASM
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} r
     * @param {number} v0
     * @param {number} kappa
     * @param {number} theta
     * @param {number} sigma
     * @param {number} rho
     * @returns {number}
     */
    price_call(spot, strike, time, r, v0, kappa, theta, sigma, rho) {
        const ret = wasm.hestonwasm_price_call(this.__wbg_ptr, spot, strike, time, r, v0, kappa, theta, sigma, rho);
        return ret;
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} r
     * @param {number} v0
     * @param {number} kappa
     * @param {number} theta
     * @param {number} sigma
     * @param {number} rho
     * @param {boolean} is_call
     * @param {number} num_paths
     * @returns {number}
     */
    price_monte_carlo(spot, strike, time, r, v0, kappa, theta, sigma, rho, is_call, num_paths) {
        const ret = wasm.hestonwasm_price_monte_carlo(this.__wbg_ptr, spot, strike, time, r, v0, kappa, theta, sigma, rho, is_call, num_paths);
        return ret;
    }
    constructor() {
        const ret = wasm.americanoptionswasm_new();
        this.__wbg_ptr = ret >>> 0;
        HestonWASMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) HestonWASM.prototype[Symbol.dispose] = HestonWASM.prototype.free;

export class MonteCarloWASM {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        MonteCarloWASMFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_montecarlowasm_free(ptr, 0);
    }
    /**
     * @param {number} spot
     * @param {number} strike
     * @param {number} time
     * @param {number} vol
     * @param {number} rate
     * @param {number} div
     * @param {boolean} is_call
     * @param {number} num_paths
     * @param {boolean} antithetic
     * @returns {number}
     */
    price_european(spot, strike, time, vol, rate, div, is_call, num_paths, antithetic) {
        const ret = wasm.montecarlowasm_price_european(this.__wbg_ptr, spot, strike, time, vol, rate, div, is_call, num_paths, antithetic);
        return ret;
    }
    constructor() {
        const ret = wasm.americanoptionswasm_new();
        this.__wbg_ptr = ret >>> 0;
        MonteCarloWASMFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
}
if (Symbol.dispose) MonteCarloWASM.prototype[Symbol.dispose] = MonteCarloWASM.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg_Error_52673b7de5a0ca89 = function(arg0, arg1) {
        const ret = Error(getStringFromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg___wbindgen_boolean_get_dea25b33882b895b = function(arg0) {
        const v = getObject(arg0);
        const ret = typeof(v) === 'boolean' ? v : undefined;
        return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
    };
    imports.wbg.__wbg___wbindgen_debug_string_adfb662ae34724b6 = function(arg0, arg1) {
        const ret = debugString(getObject(arg1));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_in_0d3e1e8f0c669317 = function(arg0, arg1) {
        const ret = getObject(arg0) in getObject(arg1);
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_function_8d400b8b1af978cd = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'function';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_object_ce774f3490692386 = function(arg0) {
        const val = getObject(arg0);
        const ret = typeof(val) === 'object' && val !== null;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_string_704ef9c8fc131030 = function(arg0) {
        const ret = typeof(getObject(arg0)) === 'string';
        return ret;
    };
    imports.wbg.__wbg___wbindgen_is_undefined_f6b95eab589e0269 = function(arg0) {
        const ret = getObject(arg0) === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_jsval_loose_eq_766057600fdd1b0d = function(arg0, arg1) {
        const ret = getObject(arg0) == getObject(arg1);
        return ret;
    };
    imports.wbg.__wbg___wbindgen_number_get_9619185a74197f95 = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'number' ? obj : undefined;
        getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
    };
    imports.wbg.__wbg___wbindgen_string_get_a2a31e16edf96e42 = function(arg0, arg1) {
        const obj = getObject(arg1);
        const ret = typeof(obj) === 'string' ? obj : undefined;
        var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
        var len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_call_3020136f7a2d6e44 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_call_abb4ff46ce38be40 = function() { return handleError(function (arg0, arg1) {
        const ret = getObject(arg0).call(getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_crypto_86f2631e91b51511 = function(arg0) {
        const ret = getObject(arg0).crypto;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_done_62ea16af4ce34b24 = function(arg0) {
        const ret = getObject(arg0).done;
        return ret;
    };
    imports.wbg.__wbg_getRandomValues_b3f15fcbfabb0f8b = function() { return handleError(function (arg0, arg1) {
        getObject(arg0).getRandomValues(getObject(arg1));
    }, arguments) };
    imports.wbg.__wbg_get_6b7bd52aca3f9671 = function(arg0, arg1) {
        const ret = getObject(arg0)[arg1 >>> 0];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_get_af9dab7e9603ea93 = function() { return handleError(function (arg0, arg1) {
        const ret = Reflect.get(getObject(arg0), getObject(arg1));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_get_with_ref_key_1dc361bd10053bfe = function(arg0, arg1) {
        const ret = getObject(arg0)[getObject(arg1)];
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_instanceof_ArrayBuffer_f3320d2419cd0355 = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof ArrayBuffer;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Uint8Array_da54ccc9d3e09434 = function(arg0) {
        let result;
        try {
            result = getObject(arg0) instanceof Uint8Array;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_isArray_51fd9e6422c0a395 = function(arg0) {
        const ret = Array.isArray(getObject(arg0));
        return ret;
    };
    imports.wbg.__wbg_iterator_27b7c8b35ab3e86b = function() {
        const ret = Symbol.iterator;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_length_22ac23eaec9d8053 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_length_d45040a40c570362 = function(arg0) {
        const ret = getObject(arg0).length;
        return ret;
    };
    imports.wbg.__wbg_msCrypto_d562bbe83e0d4b91 = function(arg0) {
        const ret = getObject(arg0).msCrypto;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_1ba21ce319a06297 = function() {
        const ret = new Object();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_25f239778d6112b9 = function() {
        const ret = new Array();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_6421f6084cc5bc5a = function(arg0) {
        const ret = new Uint8Array(getObject(arg0));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_from_slice_9a48ef80d2a51f94 = function(arg0, arg1) {
        const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_no_args_cb138f77cf6151ee = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_with_length_aa5eaf41d35235e5 = function(arg0) {
        const ret = new Uint8Array(arg0 >>> 0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_next_138a17bbf04e926c = function(arg0) {
        const ret = getObject(arg0).next;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_next_3cfe5c0fe2a4cc53 = function() { return handleError(function (arg0) {
        const ret = getObject(arg0).next();
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_node_e1f24f89a7336c2e = function(arg0) {
        const ret = getObject(arg0).node;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_process_3975fd6c72f520aa = function(arg0) {
        const ret = getObject(arg0).process;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_prototypesetcall_dfe9b766cdc1f1fd = function(arg0, arg1, arg2) {
        Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
    };
    imports.wbg.__wbg_randomFillSync_f8c153b79f285817 = function() { return handleError(function (arg0, arg1) {
        getObject(arg0).randomFillSync(takeObject(arg1));
    }, arguments) };
    imports.wbg.__wbg_require_b74f47fc2d022fd6 = function() { return handleError(function () {
        const ret = module.require;
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_set_3f1d0b984ed272ed = function(arg0, arg1, arg2) {
        getObject(arg0)[takeObject(arg1)] = takeObject(arg2);
    };
    imports.wbg.__wbg_set_7df433eea03a5c14 = function(arg0, arg1, arg2) {
        getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_769e6b65d6557335 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_60cf02db4de8e1c1 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_08f5a74c69739274 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_a8924b26aa92d024 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addHeapObject(ret);
    };
    imports.wbg.__wbg_subarray_845f2f5bce7d061a = function(arg0, arg1, arg2) {
        const ret = getObject(arg0).subarray(arg1 >>> 0, arg2 >>> 0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_value_57b7b035e117f7ee = function(arg0) {
        const ret = getObject(arg0).value;
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_versions_4e31226f5e8dc909 = function(arg0) {
        const ret = getObject(arg0).versions;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_cb9088102bce6b30 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
        const ret = getArrayU8FromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_clone_ref = function(arg0) {
        const ret = getObject(arg0);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('bsopt_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
