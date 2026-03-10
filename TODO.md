# SIMD Vectorization via Structure of Arrays (SoA)

To maximize performance, the `SynapsePool` should be reorganized. By grouping all `c0` values together, then all `c1` values, and so on, the CPU can load multiple synapses into a single register (e.g., 8 `f32` values in a 256-bit AVX register) and process the plasticity logic in parallel.

Optimized SynapsePool Structure

```zig
const SynapsePool = struct {
    weights: [MAX_SYNAPSES]f32 = undefined,
    sources: [MAX_SYNAPSES]u32 = undefined,
    targets: [MAX_SYNAPSES]u32 = undefined,
    
    // SoA Layout: Each coefficient has its own contiguous array
    c0: [MAX_SYNAPSES]f32 = undefined,
    c1: [MAX_SYNAPSES]f32 = undefined,
    c2: [MAX_SYNAPSES]f32 = undefined,
    c3: [MAX_SYNAPSES]f32 = undefined,
    c4: [MAX_SYNAPSES]f32 = undefined,
    
    act_syn: usize = 0,

    pub fn addConnection(self: *SynapsePool, src: u32, dst: u32, w: f32, c: [5]f32) void {
        if (self.act_syn >= MAX_SYNAPSES) return;
        const i = self.act_syn;
        self.weights[i] = w;
        self.sources[i] = src;
        self.targets[i] = dst;
        self.c0[i] = c[0];
        self.c1[i] = c[1];
        self.c2[i] = c[2];
        self.c3[i] = c[3];
        self.c4[i] = c[4];
        self.act_syn += 1;
    }
};

```

---

Vectorized Plasticity Update

Using `@Vector` in Zig allows the compiler to use SIMD instructions (AVX, NEON) to update multiple synapses at once.

```zig
pub fn applyPlasticityVectorized(res: *Reservoir, pool: *SynapsePool) void {
    const simd_len = std.simd.suggestVectorLength(f32) orelse 1;
    var i: usize = 0;

    while (i + simd_len <= pool.act_syn) : (i += simd_len) {
        // Load weights and coefficients into vectors
        var w_vec: @Vector(simd_len, f32) = pool.weights[i..][0..simd_len].*;
        const c0: @Vector(simd_len, f32) = pool.c0[i..][0..simd_len].*;
        const c1: @Vector(simd_len, f32) = pool.c1[i..][0..simd_len].*;
        const c2: @Vector(simd_len, f32) = pool.c2[i..][0..simd_len].*;
        const c3: @Vector(simd_len, f32) = pool.c3[i..][0..simd_len].*;
        const c4: @Vector(simd_len, f32) = pool.c4[i..][0..simd_len].*;

        // Gather pre and post states
        var pre_vec: @Vector(simd_len, f32) = undefined;
        var post_vec: @Vector(simd_len, f32) = undefined;
        
        inline for (0..simd_len) |j| {
            pre_vec[j] = res.prev_states[pool.sources[i + j]];
            post_vec[j] = res.states[pool.targets[i + j]];
        }

        // Parallel computation of Delta W
        const delta = c0 * (c1 * pre_vec * post_vec + c2 * pre_vec + c3 * post_vec + c4);
        
        // Update and store back
        pool.weights[i..][0..simd_len].* = w_vec + delta;
    }

    // Handle remaining synapses (scalar fallback)
    for (i..pool.act_syn) |idx| {
        const pre = res.prev_states[pool.sources[idx]];
        const post = res.states[pool.targets[idx]];
        const delta = pool.c0[idx] * (pool.c1[idx]*pre*post + pool.c2[idx]*pre + pool.c3[idx]*post + pool.c4[idx]);
        pool.weights[idx] += delta;
    }
}

```

Key Performance Benefits

* **Cache Locality:** Loading `c0[i..i+8]` is a single cache-line-friendly operation. In your previous `[5 * MAX_SYNAPSES]` layout, the CPU had to jump across memory to find the next `c0`.
* **Instruction Density:** One `fadd` or `fmul` instruction now processes multiple synapses.
* **Branch Prediction:** Eliminating the inner `inline for` loop in the hot path allows the CPU to pipeline the arithmetic more effectively.

Would you like me to help you implement a "Worker Thread" pattern to run these fitness evaluations in parallel across your CPU cores?
