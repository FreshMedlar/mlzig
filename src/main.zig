const std = @import("std");
const mlzig = @import("mlzig");
const analysis = @import("analysis.zig");
const tests = @import("tests.zig");

const NEURON_COUNT: usize = 10000;
const COMPRESSED_SIZE: usize = 50;
const MAX_SYNAPSES: usize = 1000000;

const Genotype = struct {
    compressed_coeffs: [5][COMPRESSED_SIZE]f32,
};

const Reservoir = struct {
    states: [NEURON_COUNT]f32 = undefined,
    prev_states: [NEURON_COUNT]f32 = undefined,
    leaks: [NEURON_COUNT]f32 = undefined,

    active_indices: [NEURON_COUNT]u32 = undefined,
    active_neuron_count: usize = 0,
    active_mask: [NEURON_COUNT / 8]u8 = [_]u8{0} ** (NEURON_COUNT / 8),
    input_sums: [NEURON_COUNT]f32 = undefined,

    pub fn init(self: *Reservoir) void {
        @memset(&self.states, 0.0);
        @memset(&self.prev_states, 0.0);
        @memset(&self.leaks, 0.8);
        @memset(&self.active_indices, 0);
        self.active_neuron_count = 0;
        @memset(&self.active_mask, 0);
        @memset(&self.input_sums, 0.0);
    }

    // pub fn clone(res: *Reservoir) Reservoir {
    //     return .{
    //         .states = [_]f32{0.0} ** NEURON_COUNT,
    //         .leaks = [_]f32{0.8} ** NEURON_COUNT,
    //     };
    // }

    // optimize to reset only firt active_neuron_count?
    pub fn reset(self: *Reservoir) void {
        @memset(&self.states, 0.0);
        @memset(&self.prev_states, 0.0);
        @memset(&self.input_sums, 0.0);
    }

    pub fn markActive(self: *Reservoir, idx: u32) void {
        const byte_idx = idx / 8;
        const bit_idx = @as(u3, @intCast(idx % 8));
        const mask = @as(u8, 1) << bit_idx;

        // Check the bitmask to avoid duplicate entries in active_indices
        if (self.active_mask[byte_idx] & mask == 0) {
            self.active_mask[byte_idx] |= mask;
            self.active_indices[self.active_neuron_count] = idx;
            self.active_neuron_count += 1;
        }
    }
};

const SynapsePool = struct {
    weights: [MAX_SYNAPSES]f32 = undefined,
    sources: [MAX_SYNAPSES]u32 = undefined,
    targets: [MAX_SYNAPSES]u32 = undefined,
    coeffs: [MAX_SYNAPSES * 5]f32 = undefined,
    act_syn: usize = 0,

    pub fn addConnection(self: *SynapsePool, res: *Reservoir, src: u32, dst: u32, w: f32, c: [5]f32) void {
        if (self.act_syn >= MAX_SYNAPSES) return;

        const idx = self.act_syn;
        self.weights[idx] = w;
        self.sources[idx] = src;
        self.targets[idx] = dst;

        const c_offset = idx * 5;
        inline for (0..5) |i| {
            self.coeffs[c_offset + i] = c[i];
        }
        self.act_syn += 1;

        res.markActive(src);
        res.markActive(dst);
    }

    pub fn pruneConnection(self: *SynapsePool, index: usize) void {
        if (index >= self.act_syn) return;

        const last_idx = self.act_syn - 1;
        self.weights[index] = self.weights[last_idx];
        self.sources[index] = self.sources[last_idx];
        self.targets[index] = self.targets[last_idx];

        const dst_offset = index * 5;
        const src_offset = last_idx * 5;
        inline for (0..5) |i| {
            self.coeffs[dst_offset + i] = self.coeffs[src_offset + i];
        }

        self.act_syn -= 1;
    }

    // TODO
    pub fn rndInit(self: *SynapsePool) void {
        @memset(&self.weights, 0.0);
        @memset(&self.sources, 0);
        @memset(&self.targets, 0);
        @memset(&self.coeffs, 0.0);
        self.act_syn = 0;
    }
};

const WorkerContext = struct {
    worker_pool: *SynapsePool,
    worker_res: *Reservoir,
    worker_readout: *Readout,
    base_pool: *const SynapsePool,
    base_res: *const Reservoir,
    base_readout: *const Readout,
};

const Readout = struct {
    weights: [NEURON_COUNT]f32,
    bias: f32,

    pub fn init(self: *Readout) void {
        @memset(&self.weights, 0.0);
        self.bias = 0.0;
    }
};

pub fn computeReadout(res: *const Reservoir, readout: *const Readout) f32 {
    var sum: f32 = readout.bias;

    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        sum += res.states[idx] * readout.weights[idx];
    }
    return sum;
}

pub fn updateReservoir(res: *Reservoir) void {
    for (&res.states, res.leaks) |*state, leak| {
        state.* += (state.*) * leak;
    }
}

pub fn forward(res: *Reservoir, pool: *SynapsePool) void {
    // we save res.states for the weights update
    // note that we may need to save the state before injecting the input
    @memcpy(&res.prev_states, &res.states);

    // prepare n input for n active_neurons
    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        res.input_sums[idx] = 0.0;
    }

    // we cycle through all the synapses
    for (pool.weights[0..pool.act_syn], pool.sources[0..pool.act_syn], pool.targets[0..pool.act_syn]) |w, src, dst| {
        res.input_sums[dst] += w * res.states[src];
    }

    // beware race conditions if modifying this code: the algo must remain order independent,
    // the loops must be separate, or use a buffer

    // we compute the actual per-neuron input
    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        const input = res.input_sums[idx];
        const leak = res.leaks[idx];

        const activated = std.math.tanh(input);
        res.states[idx] = ((1.0 - leak) * res.states[idx]) + (leak * activated);
    }
}

pub fn applyPlasticity(res: *Reservoir, pool: *SynapsePool) void {
    for (0..pool.act_syn) |i| {
        const src = pool.sources[i];
        const dst = pool.targets[i];
        const pre = res.prev_states[src];
        const post = res.states[dst];

        const c = pool.coeffs[i * 5 .. i * 5 + 5];
        const delta = c[0] * (c[1] * pre * post + c[2] * pre + c[3] * post + c[4]);

        pool.weights[i] += delta;

        // may need to clamp weights for divergence
    }
}

pub fn initialize(res: *Reservoir, pool: *SynapsePool, active_count: u16) void {
    const default_coeffs = [_]f32{0.0} ** 5;

    // we mark the N as active
    for (0..active_count) |i| {
        res.markActive(@intCast(i));
    }

    // create the synapses
    for (0..active_count) |n| {
        for (0..active_count) |m| {
            pool.addConnection(res, @intCast(n), @intCast(m), 0.1, default_coeffs);
        }
    }
}

// make this function generic
pub fn expand_genome(geno: [50]f32, pheno: *[1000]f32) void {
    for (0..1000) |i| {
        var val: f32 = 0.0;
        const x = @as(f32, @floatFromInt(i)) / 1000.0;

        for (geno, 0..) |amp, freq| {
            const f = @as(f32, @floatFromInt(freq + 1));
            val += amp * std.math.cos(f * std.math.pi * x);
        }
        pheno[i] = val;
    }
}

pub fn fitness(
    ctx: *WorkerContext,
    seed: u64,
    input_data: []const f32,
    target_data: []const f32,
    std_dev: f32,
) f32 {
    // generate perturbations here
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    // SYNAPSES TODO we may not neet to copy all the coefficients
    @memcpy(&ctx.worker_pool.weights, &ctx.base_pool.weights);
    @memcpy(&ctx.worker_pool.sources, &ctx.base_pool.sources);
    @memcpy(&ctx.worker_pool.targets, &ctx.base_pool.targets);
    @memcpy(&ctx.worker_pool.coeffs, &ctx.base_pool.coeffs);
    ctx.worker_pool.act_syn = ctx.base_pool.act_syn;
    // RESERVOIR
    @memcpy(&ctx.worker_res.states, &ctx.base_res.states);
    @memcpy(&ctx.worker_res.prev_states, &ctx.base_res.prev_states);
    @memcpy(&ctx.worker_res.leaks, &ctx.base_res.leaks);
    @memcpy(&ctx.worker_res.active_indices, &ctx.base_res.active_indices);
    ctx.worker_res.active_neuron_count = ctx.base_res.active_neuron_count;
    @memcpy(&ctx.worker_res.active_mask, &ctx.base_res.active_mask);
    @memcpy(&ctx.worker_res.input_sums, &ctx.base_res.input_sums);
    // READOUT
    @memcpy(&ctx.worker_readout.weights, &ctx.base_readout.weights);
    ctx.worker_readout.bias = ctx.base_readout.bias;
    ctx.worker_res.reset();

    for (0..ctx.worker_pool.act_syn * 5) |i| {
        const perturbation = random.floatNorm(f32) * std_dev;
        ctx.worker_pool.coeffs[i] += perturbation;
    }
    for (0..ctx.worker_readout.weights.len) |i| {
        const perturbation = random.floatNorm(f32) * std_dev;
        ctx.worker_readout.weights[i] += perturbation;
    }

    // SIMULATION
    var total_sq_error: f32 = 0.0;
    for (input_data, 0..) |val, i| {
        ctx.worker_res.states[0] = val;

        forward(ctx.worker_res, ctx.worker_pool);
        applyPlasticity(ctx.worker_res, ctx.worker_pool);

        const prediction = computeReadout(ctx.worker_res, ctx.worker_readout);
        const error_val = prediction - target_data[i];
        total_sq_error += error_val * error_val;
    }

    return total_sq_error / @as(f32, @floatFromInt(input_data.len));
}

// washout plasticity may be a problem, also the washout should be on the training data
// otherwise initial performance will be low
pub fn washout(res: *Reservoir, pool: *SynapsePool, input_data: []const f32) void {
    for (input_data) |val| {
        res.states[0] = val;
        forward(res, pool);
        applyPlasticity(res, pool);
    }
}

pub fn runEvolution(
    pool: *std.Thread.Pool,
    comptime population: usize,
    contexts: []WorkerContext,
    input_data: []const f32,
    target_data: []const f32,
    std_dev: f32,
) !void {
    var prng = std.Random.DefaultPrng.init(42);

    var wg = std.Thread.WaitGroup{};
    const num_workers = contexts.len;
    const chunk_size = population / num_workers;
    var scores: [population]f32 = undefined;
    var seeds: [population]u64 = undefined;

    // generate seeds
    for (&seeds) |*s| {
        s.* = prng.random().int(u64);
    }

    // start workers
    var start_idx: usize = 0;
    for (0..num_workers) |w| {
        const end_idx = if (w == num_workers - 1) population else start_idx + chunk_size;
        // spawnWg increments the counter and adds the task to the pool
        pool.spawnWg(&wg, evaluateBatch, .{
            &contexts[w],
            seeds[start_idx..end_idx],
            scores[start_idx..end_idx],
            input_data,
            target_data,
            std_dev,
        });

        start_idx = end_idx;
    }

    // This blocks the Main Thread until all 100 individuals are evaluated
    wg.wait();

    // compare fitness scores, pick 25% best seeds, expand and apply to base genome
    std.debug.print("All 100 evaluated. Selection starting...\n", .{});
    const winners: usize = comptime population/4;
    std.mem.sort(f32, &scores, {}, std.sort.desc(f32));
    const elite: [winners]f32 = undefined;
    @memcpy(&elite, scores[0..winners]);

    const winners_weights: [winners]f32 = undefined;
    var sum: f32 = 0.0;
    for (0..winners) |i| { 
        winners_weights[i] = std.math.log(winners+0.5) - std.math.log(i + 1);
        sum+=winners_weights[i];
    }
    for (0..winners_weights) |i| { winners_weights[i] = winners_weights[i] / sum; }

    // expand the weights 
    var out_mean: []f64 = 0.0;
    for (0..winners) |i| {
        const log_weight = winners_weights[i];
        const params = 0.0; // TODO should be the evolved weights
        for (0..winners) |j| {
            out_mean[j] += log_weight * params[j];
        }
    }
}

pub fn evaluateBatch(
    ctx: *WorkerContext,
    seeds: []const u64,
    scores: []f32,
    input_data: []const f32,
    target_data: []const f32,
    std_dev: f32,
) void {
    for (seeds, 0..) |seed, i| {
        scores[i] = fitness(ctx, seed, input_data, target_data, std_dev);
    }
}

// --------------------------------------------------------------------------

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // TEST DATA
    const params = tests.MGParams{
        .tau = 17,
        .beta = 0.2,
        .gamma = 0.1,
    };
    var mg = try tests.MackeyGlass.init(allocator, params, 1.2);
    defer mg.deinit();

    // DATA GENERATION
    const sequence_len = 500;
    const input_data = try allocator.alloc(f32, sequence_len);
    defer allocator.free(input_data);
    const target_data = try allocator.alloc(f32, sequence_len);
    defer allocator.free(target_data);

    // warm-up
    for (0..100) |_| { _ = mg.next(); }
    // gen
    var current_val = mg.next();
    for (0..sequence_len) |i| {
        const next_val = mg.next();
        input_data[i] = current_val;
        target_data[i] = next_val;
        current_val = next_val;
    }

    // PARAMETER DEFINITION
    const reservoir = try allocator.create(Reservoir);
    defer allocator.destroy(reservoir);
    reservoir.init();

    const synapses = try allocator.create(SynapsePool);
    defer allocator.destroy(synapses);
    synapses.rndInit();

    const readout = try allocator.create(Readout);
    defer allocator.destroy(readout);
    readout.init();

    initialize(reservoir, synapses, 10);

    // threads pool
    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 6 });
    defer pool.deinit();

    // create workers
    var contexts: [6]WorkerContext = undefined;
    for (0..6) |i| {
        contexts[i] = .{
            .worker_pool = try allocator.create(SynapsePool),
            .worker_res = try allocator.create(Reservoir),
            .worker_readout = try allocator.create(Readout),
            .base_pool = synapses,
            .base_res = reservoir,
            .base_readout = readout,
        };
    }

    defer {
        for (contexts) |ctx| {
            allocator.destroy(ctx.worker_pool);
            allocator.destroy(ctx.worker_res);
            allocator.destroy(ctx.worker_readout);
        }
    }

    const std_dev: f32 = 0.05;
    try runEvolution(&pool, 36, &contexts, input_data, target_data, std_dev);
}














