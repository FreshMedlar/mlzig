const std = @import("std");

pub fn checkStability(comptime neuron_count: usize, pool: anytype) f32 {
    var sums = [_]f32{0.0} ** neuron_count;
    for (pool.weights[0..pool.active_count], pool.targets[0..pool.active_count]) |w, dst| {
        sums[dst] += @abs(w);
    }
    return std.mem.max(f32, &sums);
}


