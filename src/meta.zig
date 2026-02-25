const std = @import("std");
const mx = @import("matrix.zig");
const neur = @import("neuron.zig");

pub const Population = struct {
    const NeuronCount = 1000;
    neurons: [NeuronCount]neur.Neuron = undefined,

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
};

fn meta_model() void {
    
    var pop = Population{};

    pop.init();
    pop.updateAll();
}


