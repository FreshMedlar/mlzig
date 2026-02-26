const std = @import("std");
const mx = @import("matrix.zig");
const neur = @import("neuron.zig");

pub const Genome = struct {
    const FreqCount = 10; // The "few numbers" (Knobs)
    
    // These are the only things you mutate/evolve
    coeffs_a: [FreqCount]f32 = [_]f32{0} ** FreqCount,
    coeffs_b: [FreqCount]f32 = [_]f32{0} ** FreqCount,
    coeffs_c: [FreqCount]f32 = [_]f32{0} ** FreqCount,
    coeffs_d: [FreqCount]f32 = [_]f32{0} ** FreqCount,
    coeffs_e: [FreqCount]f32 = [_]f32{0} ** FreqCount,

    
    /// Inflates the small coefficients into a large array of parameters
    pub fn inflate(self: Genome, out: []f32) void {
        const n = out.len;
        // we loop
        for (out, 0..) |*val, i| {
            var sum: f32 = 0;
            const x = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
            
            // This is a simplified IDFT (sum of sines/cosines)
            for (self.coeffs_a, 0..) |coeff, freq| {
                const f = @as(f32, @floatFromInt(freq + 1));
                sum += coeff * std.math.cos(2.0 * std.math.pi * f * x);
            }
            val.* = sum;
        }
    }
};

pub const Population = struct {
    const NeuronCount = 1000;
    neurons: [NeuronCount]neur.Neuron = undefined,
    genome: Genome,

    pub fn applyGenome(self: *Population) void {
        
        var temp_params: [NeuronCount]f32 = undefined;
        self.genome.inflate(&temp_params);

        for (&self.neurons, 0..) |*n, i| n.a = temp_params[i];


    }
    pub fn init(self: *Population) void {
        for (&self.neurons) |*n| {
            n.* = neur.Neuron {
                .weight = 1.0,
                .a = 0.01, .b = 0.01, .c = 0.01, .d = 0.01,
                .lr = 0.01,
                .input = 0.0,
                .output = 0.0,
            };
        }
    }

    pub fn updateAll (self: *Population) void {
        for (&self.neurons) |*n| {
            n.updateWeight();
        }
    }

    pub fn forward(self: *Population) void {

    }
};



















