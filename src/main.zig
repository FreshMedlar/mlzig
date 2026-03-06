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
    biases: [NEURON_COUNT]f32 = undefined,
    leaks: [NEURON_COUNT]f32 = undefined,
    
    active_indices: [NEURON_COUNT]u32 = undefined,
    active_neuron_count:usize = 0,
    active_mask: [NEURON_COUNT/8]u8 = [_]u8{0} ** (NEURON_COUNT / 8),
    input_sums: [NEURON_COUNT]f32 = undefined, 

    pub fn init() Reservoir {
        return .{
            .states = [_]f32{0.0} ** NEURON_COUNT,
            .biases = [_]f32{0.0} ** NEURON_COUNT,
            .leaks = [_]f32{0.8} ** NEURON_COUNT,
        };
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

    pub fn addConnection(self: *SynapsePool, src: u32, dst: u32, w: f32, c: [5]f32) void {
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

pub fn updateReservoir(res: *Reservoir) void {
    for (&res.states, res.biases, res.leaks) |*state, bias, leak| {
        state.* += (bias - state.*) * leak;
    }
}

pub fn forward(res: *Reservoir, pool: *SynapsePool) void {
    // we save res.states for the weights update
    // note that we may need to save the state before injecting the input
    @memcpy(res.prev_states, res.states);

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
        const bias = res.biases[idx];

        const activated = std.math.tanh(input + bias);
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

pub fn addConnection(res: *Reservoir, pool: *SynapsePool, src: u32, dst: u32, w: f32, c: [5]f32) void {
    // 1. Add the synapse to the pool
    pool.addConnection(src, dst, w, c);

    // 2. Register these neurons as active
    // This adds them to the active_indices list if they aren't there already
    res.markActive(src);
    res.markActive(dst);
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
            addConnection(res, pool, @intCast(n), @intCast(m), 0.1, default_coeffs);
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
    original_pool: *const SynapsePool, 
    perturbation: [5000]f32,
    input_data: []f32
) f32 {
    var worker_pool = original_pool.*;

    for (perturbation, 0..) |p, i| {
        worker_pool.coeffs[i] += p;
    }

    var worker_res = Reservoir.init();

    var total_error: f32 = 0.0;
    for(input_data) |val| {
        worker_res.states[0] = val;
        forward(&worker_res, &worker_pool);
        applyPlasticity(&worker_res, &worker_pool);

        // TODO readout
        total_error += 0.01;
    }

    return total_error;
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
//    for (0..20) |i| {
//        const val = mg.next();
//        std.debug.print("{d}, {d:.6}\n", .{ i, val });
//    }

    initialize(&reservoir, &synapses, 10);
    
    for (0..99) |i| {
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












