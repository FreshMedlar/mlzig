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
    active_neuron_count:usize = 0,
    active_mask: [NEURON_COUNT/8]u8 = [_]u8{0} ** (NEURON_COUNT / 8),
    input_sums: [NEURON_COUNT]f32 = undefined, 

    pub fn init() Reservoir {
        return .{
            .states = [_]f32{0.0} ** NEURON_COUNT,
            .leaks = [_]f32{0.8} ** NEURON_COUNT,
        };
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

    pub fn rndInit() SynapsePool {
        
    }
};

const Readout = struct {
    weights: [NEURON_COUNT]f32, 
    bias: f32,

    pub fn initReadout() Readout {
        return .{
            .weights = [_]f32{0.0} ** NEURON_COUNT,
            .bias = 0.0,
        };
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
    for (
        pool.weights[0..pool.act_syn],
        pool.sources[0..pool.act_syn], 
        pool.targets[0..pool.act_syn]
    ) |w, src, dst| {
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
        const delta = c[0] * (c[1]*pre*post + c[2]*pre + c[3]*post + c[4]);

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
    allocator: std.mem.Allocator,
    original_pool: *const SynapsePool, 
    original_res: *const Reservoir,
    readout: *const Readout,
    perturbation: []const f32,
    input_data: []const f32,
    target_data: []const f32
) !f32 {
    var worker_pool = try allocator.create(SynapsePool);
    defer allocator.destroy(worker_pool);
    worker_pool.* = original_pool.*;

    for (perturbation, 0..) |p, i| {
        worker_pool.coeffs[i] += p;
    }

    var worker_res = try allocator.create(Reservoir);
    defer allocator.destroy(worker_res);
    worker_res.* = original_res.*;
    worker_res.reset();

    var total_sq_error: f32 = 0.0;
    for(input_data, 0..) |val, i| {
        worker_res.states[0] = val;
        forward(worker_res, worker_pool);
        applyPlasticity(worker_res, worker_pool);

        const prediction = computeReadout(worker_res, readout);
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

pub fn generateNoise(buffer: []f32, std_dev: f32) void {
        
}

pub fn evaluateOffsprint(base_genome: []const f32, perturbation: []const f32) f32 {
    // generate random from seed

    // use fitness function to evaluate
    
    // return fitness score and seed 
}

pub fn runEvolution(epochs: usize, population: usize, initial_genome: []f32) void {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    // generate random seed for thread pool

    // start workers

    // compare fitness scores, pick best seeds, expand and apply to base genome
}

// --------------------------------------------------------------------------

var reservoir = Reservoir{};
var synapses = SynapsePool{};

pub fn main() !void {

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = tests.MGParams{
        .tau = 17,
        .beta = 0.2,
        .gamma = 0.1,
    };
    
    var mg = try tests.MackeyGlass.init(allocator, params, 1.2);
    defer mg.deinit();

    std.debug.print("Step, Value\n", .{});
    initialize(&reservoir, &synapses, 10);
    
    for (0..999) |i| {
        // input
        reservoir.states[0] = mg.next();

        forward(&reservoir, &synapses);
        applyPlasticity(&reservoir, &synapses);

        if (i % 100 == 0) {
            for (0..5) |n| {std.debug.print("{d:.4}", .{reservoir.states[n]});}
            std.debug.print("\n", .{});
        }
    }
}












