const std = @import("std");

pub const MGParams = struct {
    beta: f32 = 0.2,
    gamma: f32 = 0.1,
    n: f32 = 10.0,
    tau: usize = 17, // Standard chaos threshold
    dt: f32 = 1.0,   // Time step
};

pub const MackeyGlass = struct {
    params: MGParams,
    history: std.ArrayList(f32),
    current_x: f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, params: MGParams, seed_x: f32) !MackeyGlass {
        var history = std.ArrayListUnmanaged(f32){};
        // Fill history with initial seed value
        try history.appendNTimes(allocator, seed_x, params.tau + 1);
        return .{
            .params = params,
            .history = history,
            .current_x = seed_x,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MackeyGlass) void {
        self.history.deinit(self.allocator);
    }

    pub fn next(self: *MackeyGlass) f32 {
        const x_delayed = self.history.items[0]; // x(t - tau)
        
        // Calculate derivative: dx/dt
        const p = self.params;
        const numerator = p.beta * x_delayed;
        const denominator = 1.0 + std.math.pow(f32, x_delayed, p.n);
        const dxdt = (numerator / denominator) - (p.gamma * self.current_x);

        // Update state: x(t+1) = x(t) + dt * dxdt
        self.current_x += p.dt * dxdt;

        // Update history buffer (Slide window)
        _ = self.history.orderedRemove(0);
        self.history.append(self.allocator, self.current_x) catch unreachable;

        return self.current_x;
    }
};
