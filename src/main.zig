const std = @import("std");
const mlzig = @import("mlzig");
const analysis = @import("analysis.zig");
const tests = @import("tests.zig");

const NEURON_COUNT: usize = 10000;
const MAX_SYNAPSES: usize = 1000000;

const Reservoir = struct {
    states: [NEURON_COUNT]f32 = undefined,
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
    active_count: usize = 0,

    pub fn addConnection(self: *SynapsePool, src: u32, dst: u32, w: f32) void {
        if (self.active_count >= MAX_SYNAPSES) return;

        const idx = self.active_count;
        self.weights[idx] = w;
        self.sources[idx] = src;
        self.targets[idx] = dst;
        self.active_count += 1;
    }

    pub fn pruneConnection(self: *SynapsePool, index: usize) void {
        if (index >= self.active_count) return;

        const last_idx = self.active_count - 1;
        self.weights[index] = self.weights[last_idx];
        self.sources[index] = self.sources[last_idx];
        self.targets[index] = self.targets[last_idx];

        self.active_count -= 1;
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
    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        res.input_sums[idx] = 0.0;
    }

    for (
        pool.weights[0..pool.active_count],
        pool.sources[0..pool.active_count],
        pool.targets[0..pool.active_count]
    ) |w, src, dst| {
        res.input_sums[dst] += w * res.states[src];
    }

    for (res.active_indices[0..res.active_neuron_count]) |idx| {
        const input = res.input_sums[idx];
        const leak = res.leaks[idx];
        const bias = res.biases[idx];

        const activated = std.math.tanh(input + bias);
        res.states[idx] = ((1.0 - leak) * res.states[idx]) + (leak * activated);
    }
} 

pub fn addConnection(res: *Reservoir, pool: *SynapsePool, src: u32, dst: u32, w: f32) void {
    // 1. Add the synapse to the pool
    pool.addConnection(src, dst, w);

    // 2. Register these neurons as active
    // This adds them to the active_indices list if they aren't there already
    res.markActive(src);
    res.markActive(dst);
}

pub fn initialize(res: *Reservoir, pool: *SynapsePool, active_count: u16) void {
    // we mark the N as active 
    for (0..active_count) |i| {
        res.markActive(@intCast(i));
    }
    // create the synapses
    for (0..active_count) |n| {
        for (0..active_count) |m| {
            addConnection(res, pool, @intCast(n), @intCast(m), 0.1);
        }
    }
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
    
    for (0..99999) |i| {
        // input
        reservoir.states[0] = mg.next();

        forward(&reservoir, &synapses);
        if (i % 100 == 0) {
            for (0..5) |n| {std.debug.print("{d:.4}", .{reservoir.states[n]});}
            std.debug.print("\n", .{});
        }
    }
}












