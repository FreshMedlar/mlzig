const std = @import("std");
const analysis = @import("analysis.zig");
const tests = @import("tests.zig");

const print = std.debug.print;

const POPULATION: usize = 280;
const GENERATIONS: usize = 1000;
const NEURON_COUNT: usize = 1000;
const COMPRESSED_SIZE: usize = 100;
const READOUT_COMPRESSED_SIZE: usize = 100;
const SPARSITY: usize = 50; // connections per neuron (10% of 500)
const MAX_SYNAPSES: usize = NEURON_COUNT * SPARSITY;
const NUM_WORKERS: usize = 8;

const SEQUENCE_LEN: usize = 200;
const MAX_PREDICT_STEPS: usize = 10;
const CHAR_SET_SIZE: usize = 65;

const Genotype = struct {
    coeffs: [5][COMPRESSED_SIZE]f32,
    readout_coeffs: [READOUT_COMPRESSED_SIZE]f32,

    pub fn init() Genotype {
        var g: Genotype = undefined;
        @memset(std.mem.asBytes(&g.coeffs), 0);
        @memset(&g.readout_coeffs, 0.0);
        return g;
    }
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

    pub fn reset(self: *Reservoir) void {
        @memset(&self.states, 0.0);
        @memset(&self.prev_states, 0.0);
        @memset(&self.input_sums, 0.0);
    }

    pub fn markActive(self: *Reservoir, idx: u32) void {
        const byte_idx = idx / 8;
        const bit_idx = @as(u3, @intCast(idx % 8));
        const mask = @as(u8, 1) << bit_idx;

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

    pub fn addConnection(self: *SynapsePool, res: *Reservoir, src: u32, dst: u32, w: f32) void {
        if (self.act_syn >= MAX_SYNAPSES) return;
        const idx = self.act_syn;
        self.weights[idx] = w;
        self.sources[idx] = src;
        self.targets[idx] = dst;
        self.act_syn += 1;

        res.markActive(src);
        res.markActive(dst);
    }

    pub fn rndInit(self: *SynapsePool) void {
        @memset(&self.weights, 0.0);
        @memset(&self.sources, 0);
        @memset(&self.targets, 0);
        @memset(&self.coeffs, 0.0);
        self.act_syn = 0;
    }
};

const WorkerContext = struct {
    allocator: std.mem.Allocator,
    worker_pool: *SynapsePool,
    worker_res: *Reservoir,
    base_pool: *const SynapsePool,
    base_res: *const Reservoir,
    base_genotype: *Genotype,
    worker_geno: *Genotype,
    readout_weights: *[CHAR_SET_SIZE][NEURON_COUNT]f32,
};

pub fn forward(res: *Reservoir, pool: *SynapsePool) void {
    @memcpy(&res.prev_states, &res.states);

    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        res.input_sums[idx] = 0.0;
    }

    for (pool.weights[0..pool.act_syn], pool.sources[0..pool.act_syn], pool.targets[0..pool.act_syn]) |w, src, dst| {
        res.input_sums[dst] += w * res.states[src];
    }

    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        const input = res.input_sums[idx];
        const safe_in = std.math.clamp(input, -50.0, 50.0);
        const leak = res.leaks[idx];
        const activated = std.math.tanh(safe_in);
        res.states[idx] = ((1.0 - leak) * res.states[idx]) + (leak * activated);
    }
}

pub fn applyPlasticity(res: *Reservoir, pool: *SynapsePool) void {
    for (0..pool.act_syn) |i| {
        const src = pool.sources[i];
        const dst = pool.targets[i];
        const pre = res.prev_states[src];
        const post = res.states[dst];

        const c_idx = i * 5;
        const c0 = pool.coeffs[c_idx + 0];
        const c1 = pool.coeffs[c_idx + 1];
        const c2 = pool.coeffs[c_idx + 2];
        const c3 = pool.coeffs[c_idx + 3];
        const c4 = pool.coeffs[c_idx + 4];

        const delta = c0 * (c1 * pre * post + c2 * pre + c3 * post + c4);
        pool.weights[i] += delta;
        pool.weights[i] = std.math.clamp(pool.weights[i], -2.0, 2.0);
    }
}

// --- Improvement 1: Sparse random connectivity ---
pub fn initialize(res: *Reservoir, pool: *SynapsePool) void {
    // Mark all as active
    for (0..NEURON_COUNT) |i| {
        res.markActive(@intCast(i));
    }

    // Sparse random connections: each neuron connects to SPARSITY random targets
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..NEURON_COUNT) |n| {
        for (0..SPARSITY) |_| {
            const target = random.uintLessThan(u32, NEURON_COUNT);
            // Small initial weight scaled by 1/sqrt(SPARSITY) for stable dynamics
            const w: f32 = (random.float(f32) - 0.5) * 2.0 / @as(f32, @sqrt(@as(f32, @floatFromInt(SPARSITY))));
            pool.addConnection(res, @intCast(n), target, w);
        }
    }
}

pub fn expand_genome(geno: *const Genotype, pheno_coeffs: []f32, sources: []const u32, targets: []const u32, act_syn: usize) void {
    var cx: [10][NEURON_COUNT]f32 = undefined;
    for (0..10) |f| {
        const freq = @as(f32, @floatFromInt(f + 1));
        for (0..NEURON_COUNT) |i| {
            const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(NEURON_COUNT));
            cx[f][i] = std.math.cos(freq * std.math.pi * x);
        }
    }

    var temp: [10][NEURON_COUNT]f32 = undefined;
    inline for (0..5) |t| {
        for (0..10) |fx| {
            for (0..NEURON_COUNT) |dst| {
                var s: f32 = 0.0;
                for (0..10) |fy| {
                    s += geno.coeffs[t][fx * 10 + fy] * cx[fy][dst];
                }
                temp[fx][dst] = s;
            }
        }

        for (0..act_syn) |i| {
            const src = sources[i];
            const dst = targets[i];
            var val: f32 = 0.0;
            for (0..10) |fx| {
                val += cx[fx][src] * temp[fx][dst];
            }
            pheno_coeffs[i * 5 + t] = val;
        }
    }
}

// --- Improvement 2: Expand compressed readout coefficients into full readout weights ---
pub fn expand_readout(coeffs: []const f32, out_weights: *[CHAR_SET_SIZE][NEURON_COUNT]f32) void {
    // 2D cosine basis: 10 frequencies for character dim × 10 frequencies for neuron dim = 100 coefficients
    const NUM_FREQ: usize = 10;

    // Precompute neuron-axis cosines
    var cos_neuron: [NUM_FREQ][NEURON_COUNT]f32 = undefined;
    for (0..NUM_FREQ) |f| {
        const freq = @as(f32, @floatFromInt(f + 1));
        for (0..NEURON_COUNT) |n| {
            const x = @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(NEURON_COUNT));
            cos_neuron[f][n] = std.math.cos(freq * std.math.pi * x);
        }
    }

    // Precompute character-axis cosines
    var cos_char: [NUM_FREQ][CHAR_SET_SIZE]f32 = undefined;
    for (0..NUM_FREQ) |f| {
        const freq = @as(f32, @floatFromInt(f + 1));
        for (0..CHAR_SET_SIZE) |c| {
            const x = @as(f32, @floatFromInt(c)) / @as(f32, @floatFromInt(CHAR_SET_SIZE));
            cos_char[f][c] = std.math.cos(freq * std.math.pi * x);
        }
    }

    // Expand: w[c][n] = sum over fc, fn of coeff[fc*10+fn] * cos_char[fc][c] * cos_neuron[fn][n]
    for (0..CHAR_SET_SIZE) |c| {
        for (0..NEURON_COUNT) |n| {
            var val: f32 = 0.0;
            for (0..NUM_FREQ) |fc| {
                for (0..NUM_FREQ) |fn_idx| {
                    val += coeffs[fc * NUM_FREQ + fn_idx] * cos_char[fc][c] * cos_neuron[fn_idx][n];
                }
            }
            out_weights[c][n] = val;
        }
    }
}

const Seed = struct {
    seed: u64,
    score: f32,
    accuracy: f32,
};

fn ascScore(context: void, a: Seed, b: Seed) bool {
    _ = context;
    return a.score < b.score;
}

pub fn fitness(ctx: *WorkerContext, seed_data: *Seed, data_chunk: []const usize, sigma: f32, cur_steps: usize) void {
    const active_syn = ctx.base_pool.act_syn;
    const active_nrn = ctx.base_res.active_neuron_count;

    var prng = std.Random.DefaultPrng.init(seed_data.seed);
    const random = prng.random();

    // Genotype Copy and Mutation (plasticity coefficients)
    @memcpy(&ctx.worker_geno.coeffs, &ctx.base_genotype.coeffs);
    for (0..5) |t| {
        for (0..COMPRESSED_SIZE) |c| {
            const epsilon = random.floatNorm(f32);
            ctx.worker_geno.coeffs[t][c] += epsilon * sigma;
        }
    }

    // Mutate compressed readout coefficients
    @memcpy(&ctx.worker_geno.readout_coeffs, &ctx.base_genotype.readout_coeffs);
    for (0..READOUT_COMPRESSED_SIZE) |i| {
        const epsilon = random.floatNorm(f32);
        ctx.worker_geno.readout_coeffs[i] += epsilon * sigma;
    }

    // Expand compressed readout into full weight matrix
    expand_readout(&ctx.worker_geno.readout_coeffs, ctx.readout_weights);

    // Synapse Base Copy
    @memcpy(ctx.worker_pool.weights[0..active_syn], ctx.base_pool.weights[0..active_syn]);
    @memcpy(ctx.worker_pool.sources[0..active_syn], ctx.base_pool.sources[0..active_syn]);
    @memcpy(ctx.worker_pool.targets[0..active_syn], ctx.base_pool.targets[0..active_syn]);
    ctx.worker_pool.act_syn = active_syn;

    expand_genome(ctx.worker_geno, &ctx.worker_pool.coeffs, ctx.worker_pool.sources[0..active_syn], ctx.worker_pool.targets[0..active_syn], active_syn);

    // Res Base Copy
    @memcpy(&ctx.worker_res.leaks, &ctx.base_res.leaks);
    @memcpy(&ctx.worker_res.active_indices, &ctx.base_res.active_indices);
    ctx.worker_res.active_neuron_count = active_nrn;
    @memcpy(&ctx.worker_res.active_mask, &ctx.base_res.active_mask);
    ctx.worker_res.reset();

    var total_loss: f32 = 0.0;
    var total_correct: usize = 0;
    var total_tests: usize = 0;

    var last_predicted: usize = data_chunk[0];

    // SIMULATION & EVALUATION
    for (0..SEQUENCE_LEN) |t| {
        const is_autoregressive = t >= (SEQUENCE_LEN - cur_steps) and t > 0;
        const current_char = if (is_autoregressive) last_predicted else data_chunk[t];

        for (0..CHAR_SET_SIZE) |c| {
            ctx.worker_res.states[c] = if (c == current_char) 1.0 else 0.0;
        }

        forward(ctx.worker_res, ctx.worker_pool);
        applyPlasticity(ctx.worker_res, ctx.worker_pool);

        const target_class = data_chunk[t + 1];
        var logits: [CHAR_SET_SIZE]f32 = undefined;
        var max_logit: f32 = -1e9;
        var best_c: usize = 0;

        for (0..CHAR_SET_SIZE) |c| {
            var val: f32 = 0.0;
            for (0..NEURON_COUNT) |n| {
                val += ctx.worker_res.states[n] * ctx.readout_weights[c][n];
            }
            logits[c] = val;
            if (val > max_logit) {
                max_logit = val;
                best_c = c;
            }
        }

        last_predicted = best_c;
        if (best_c == target_class) total_correct += 1;

        var sum_exp: f32 = 0.0;
        for (0..CHAR_SET_SIZE) |c| sum_exp += std.math.exp(logits[c] - max_logit);
        const prob = std.math.exp(logits[target_class] - max_logit) / sum_exp;

        const safe_prob = std.math.clamp(prob, 1e-7, 1.0);
        total_loss -= std.math.log(f32, std.math.e, safe_prob);
        total_tests += 1;
    }

    // L2 regularization
    const l2_lambda: f32 = 0.001;
    var l2_penalty: f32 = 0.0;

    for (0..READOUT_COMPRESSED_SIZE) |i| {
        const w = ctx.worker_geno.readout_coeffs[i];
        l2_penalty += w * w;
    }
    for (0..5) |t| {
        for (0..COMPRESSED_SIZE) |c| {
            const w = ctx.worker_geno.coeffs[t][c];
            l2_penalty += w * w;
        }
    }

    const unpenalized_loss = total_loss / @as(f32, @floatFromInt(total_tests));
    seed_data.score = unpenalized_loss + (l2_lambda * l2_penalty);
    seed_data.accuracy = @as(f32, @floatFromInt(total_correct)) / @as(f32, @floatFromInt(total_tests));
}

pub fn evaluateBatch(ctx: *WorkerContext, seeds: []Seed, data_chunk: []const usize, sigma: f32, cur_steps: usize) void {
    for (seeds) |*s| {
        fitness(ctx, s, data_chunk, sigma, cur_steps);
    }
}

const EsState = struct {
    sigma: f32,
    prev_best: f32,
    stagnation_count: usize,
};

const Trainer = struct {
    allocator: std.mem.Allocator,
    pool: std.Thread.Pool,
    contexts: []WorkerContext,
    es_state: EsState,
    base_genotype: *Genotype,
    dataset: *const Dataset,

    pub fn init(allocator: std.mem.Allocator, num_workers: usize, base_pool: *const SynapsePool, base_res: *const Reservoir, dataset: *const Dataset) !*Trainer {
        const self = try allocator.create(Trainer);
        self.allocator = allocator;
        try self.pool.init(.{ .allocator = allocator, .n_jobs = @as(u32, @intCast(num_workers)) });

        self.base_genotype = try allocator.create(Genotype);
        self.base_genotype.* = Genotype.init();
        self.dataset = dataset;

        self.contexts = try allocator.alloc(WorkerContext, num_workers);
        for (0..num_workers) |i| {
            self.contexts[i] = .{
                .allocator = allocator,
                .worker_pool = try allocator.create(SynapsePool),
                .worker_res = try allocator.create(Reservoir),
                .base_pool = base_pool,
                .base_res = base_res,
                .base_genotype = self.base_genotype,
                .worker_geno = try allocator.create(Genotype),
                .readout_weights = try allocator.create([CHAR_SET_SIZE][NEURON_COUNT]f32),
            };
            self.contexts[i].worker_res.init();
            self.contexts[i].worker_pool.rndInit();
        }

        self.es_state = EsState{ .sigma = 0.05, .prev_best = 1e9, .stagnation_count = 0 };
        return self;
    }

    pub fn deinit(self: *Trainer) void {
        for (self.contexts) |ctx| {
            self.allocator.destroy(ctx.worker_pool);
            self.allocator.destroy(ctx.worker_res);
            self.allocator.destroy(ctx.worker_geno);
            self.allocator.destroy(ctx.readout_weights);
        }
        self.allocator.free(self.contexts);
        self.allocator.destroy(self.base_genotype);
        self.pool.deinit();
        self.allocator.destroy(self);
    }

    // --- Improvement 6: Sample text generation ---
    fn generateSample(self: *Trainer, base_pool_ptr: *const SynapsePool, base_res_ptr: *const Reservoir) void {
        const ctx = &self.contexts[0];
        const active_syn = base_pool_ptr.act_syn;

        // Copy base genotype (no mutation)
        @memcpy(&ctx.worker_geno.coeffs, &self.base_genotype.coeffs);
        @memcpy(&ctx.worker_geno.readout_coeffs, &self.base_genotype.readout_coeffs);

        expand_readout(&ctx.worker_geno.readout_coeffs, ctx.readout_weights);

        @memcpy(ctx.worker_pool.weights[0..active_syn], base_pool_ptr.weights[0..active_syn]);
        @memcpy(ctx.worker_pool.sources[0..active_syn], base_pool_ptr.sources[0..active_syn]);
        @memcpy(ctx.worker_pool.targets[0..active_syn], base_pool_ptr.targets[0..active_syn]);
        ctx.worker_pool.act_syn = active_syn;

        expand_genome(ctx.worker_geno, &ctx.worker_pool.coeffs, ctx.worker_pool.sources[0..active_syn], ctx.worker_pool.targets[0..active_syn], active_syn);

        @memcpy(&ctx.worker_res.leaks, &base_res_ptr.leaks);
        @memcpy(&ctx.worker_res.active_indices, &base_res_ptr.active_indices);
        ctx.worker_res.active_neuron_count = base_res_ptr.active_neuron_count;
        @memcpy(&ctx.worker_res.active_mask, &base_res_ptr.active_mask);
        ctx.worker_res.reset();

        // Seed with a space character
        var current_char: usize = self.dataset.char_to_idx[' '];
        var output_buf: [100]u8 = undefined;

        for (0..100) |i| {
            for (0..CHAR_SET_SIZE) |c| {
                ctx.worker_res.states[c] = if (c == current_char) 1.0 else 0.0;
            }

            forward(ctx.worker_res, ctx.worker_pool);
            applyPlasticity(ctx.worker_res, ctx.worker_pool);

            var max_logit: f32 = -1e9;
            var best_c: usize = 0;
            for (0..CHAR_SET_SIZE) |c| {
                var val: f32 = 0.0;
                for (0..NEURON_COUNT) |n| {
                    val += ctx.worker_res.states[n] * ctx.readout_weights[c][n];
                }
                if (val > max_logit) {
                    max_logit = val;
                    best_c = c;
                }
            }

            output_buf[i] = self.dataset.idx_to_char[best_c];
            current_char = best_c;
        }

        print("  Sample: \"{s}\"\n", .{&output_buf});
    }

    pub fn train(
        self: *Trainer,
        data_text: []const usize,
    ) !void {
        var cur_steps: usize = 1;
        print("Starting training with Curriculum. Initial steps: {d}\n", .{cur_steps});
        print("Total evolved parameters: {d} (plasticity) + {d} (readout) = {d}\n", .{ 5 * COMPRESSED_SIZE, READOUT_COMPRESSED_SIZE, 5 * COMPRESSED_SIZE + READOUT_COMPRESSED_SIZE });

        for (0..GENERATIONS) |gen| {
            const gen_start = std.time.milliTimestamp();

            var prng = std.Random.DefaultPrng.init(@as(u64, @intCast(gen_start)) + gen);
            var wg = std.Thread.WaitGroup{};

            var seeds: [POPULATION]Seed = undefined;
            for (&seeds) |*s| {
                s.seed = prng.random().int(u64);
                s.score = 0;
                s.accuracy = 0;
            }

            // --- Improvement 4: Random data sampling ---
            const max_offset = data_text.len - SEQUENCE_LEN - MAX_PREDICT_STEPS;
            const start_offset = prng.random().uintLessThan(usize, max_offset);
            const data_chunk = data_text[start_offset .. start_offset + SEQUENCE_LEN + MAX_PREDICT_STEPS];

            var start_idx: usize = 0;
            const chunk_size = POPULATION / NUM_WORKERS;

            for (0..NUM_WORKERS) |w| {
                const end_idx = if (w == NUM_WORKERS - 1) POPULATION else start_idx + chunk_size;
                self.pool.spawnWg(&wg, evaluateBatch, .{ &self.contexts[w], seeds[start_idx..end_idx], data_chunk, self.es_state.sigma, cur_steps });
                start_idx = end_idx;
            }

            wg.wait();

            const winners = POPULATION / 4;
            std.mem.sort(Seed, &seeds, {}, ascScore);
            const best = seeds[0];

            // --- Improvement 3: Rank-based weighted recombination ---
            var utilities: [POPULATION / 4]f32 = undefined;
            var util_sum: f32 = 0.0;
            for (0..winners) |i| {
                const rank_val = std.math.log(f32, std.math.e, @as(f32, @floatFromInt(winners)) + 0.5) - std.math.log(f32, std.math.e, @as(f32, @floatFromInt(i + 1)));
                utilities[i] = @max(0.0, rank_val);
                util_sum += utilities[i];
            }
            // Normalize
            for (0..winners) |i| {
                utilities[i] /= util_sum;
            }

            for (0..winners) |i| {
                const seed = seeds[i].seed;
                var loop_prng = std.Random.DefaultPrng.init(seed);
                const random = loop_prng.random();
                const weight = utilities[i];

                for (0..5) |t| {
                    for (0..COMPRESSED_SIZE) |c| {
                        const epsilon = random.floatNorm(f32);
                        self.base_genotype.coeffs[t][c] += epsilon * self.es_state.sigma * weight;
                    }
                }
                for (0..READOUT_COMPRESSED_SIZE) |c| {
                    const epsilon = random.floatNorm(f32);
                    self.base_genotype.readout_coeffs[c] += epsilon * self.es_state.sigma * weight;
                }
            }

            // --- Improvement 5: Adaptive sigma ---
            if (best.score < self.es_state.prev_best) {
                self.es_state.sigma *= 0.95;
                self.es_state.stagnation_count = 0;
            } else {
                self.es_state.stagnation_count += 1;
                if (self.es_state.stagnation_count >= 5) {
                    self.es_state.sigma *= 1.1;
                    self.es_state.stagnation_count = 0;
                }
            }
            self.es_state.sigma = std.math.clamp(self.es_state.sigma, 0.005, 0.15);
            self.es_state.prev_best = best.score;

            // --- Improvement 6: Logging with timing ---
            const gen_end = std.time.milliTimestamp();
            const gen_ms = gen_end - gen_start;

            print("Gen {d} | Steps: {d} | CE: {d:.4} | Acc: {d:.2}% | Sigma: {d:.4} | {d}ms\n", .{ gen, cur_steps, best.score, best.accuracy * 100.0, self.es_state.sigma, gen_ms });

            // Sample text every 50 generations
            if (gen % 50 == 0 and gen > 0) {
                self.generateSample(self.contexts[0].base_pool, self.contexts[0].base_res);
            }

            // Curriculum check
            if (best.accuracy > 0.45 or (cur_steps == 1 and best.accuracy > 0.25)) {
                if (cur_steps < MAX_PREDICT_STEPS) {
                    cur_steps += 1;
                    print("--> CURRICULUM UPGRADE: Now predicting {d} steps ahead.\n", .{cur_steps});
                    self.es_state.sigma = 0.05;
                }
            }
        }
    }
};

const Dataset = struct {
    text: []usize,
    char_to_idx: [256]usize,
    idx_to_char: [CHAR_SET_SIZE]u8,

    pub fn init(allocator: std.mem.Allocator, raw: []const u8) !Dataset {
        var presents = [_]bool{false} ** 256;
        for (raw) |c| presents[c] = true;

        var char_to_idx = [_]usize{0} ** 256;
        var idx_to_char = [_]u8{0} ** CHAR_SET_SIZE;
        var idx: usize = 0;
        for (0..256) |c| {
            if (presents[c]) {
                if (idx < CHAR_SET_SIZE) {
                    char_to_idx[c] = idx;
                    idx_to_char[idx] = @intCast(c);
                }
                idx += 1;
            }
        }

        const text = try allocator.alloc(usize, raw.len);
        for (raw, 0..) |c, i| text[i] = char_to_idx[c];

        return Dataset{
            .text = text,
            .char_to_idx = char_to_idx,
            .idx_to_char = idx_to_char,
        };
    }

    pub fn deinit(self: *Dataset, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var file = try std.fs.cwd().openFile("tinyshakespeare.txt", .{});
    const file_size = try file.getEndPos();
    const raw_data = try file.readToEndAlloc(allocator, file_size);
    file.close();
    defer allocator.free(raw_data);

    var dataset = try Dataset.init(allocator, raw_data);
    defer dataset.deinit(allocator);

    print("Loaded Dataset: {d} bytes. Unique Characters: {d}\n", .{ dataset.text.len, CHAR_SET_SIZE });
    if (dataset.text.len < SEQUENCE_LEN + MAX_PREDICT_STEPS) {
        print("Dataset too small!\n", .{});
        return;
    }

    const reservoir = try allocator.create(Reservoir);
    defer allocator.destroy(reservoir);
    reservoir.init();

    const synapses = try allocator.create(SynapsePool);
    defer allocator.destroy(synapses);
    synapses.rndInit();

    print("Initializing reservoir connections (sparse, {d} per neuron)...\n", .{SPARSITY});
    initialize(reservoir, synapses);
    print("Synapses created: {d}\n", .{synapses.act_syn});

    var trainer = try Trainer.init(allocator, NUM_WORKERS, synapses, reservoir, &dataset);
    defer trainer.deinit();

    try trainer.train(dataset.text);

    print("Evolution Finished.\n", .{});
}
