const std = @import("std");


pub const Neuron = struct {
    weight: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    lr: f32,
    input: f32,
    output: f32,

    pub fn updateWeight(self: *Neuron) void {
        self.weight += self.lr * ((self.a*self.input*self.output) + (self.b*self.input) + (self.c*self.output) + self.d);
    } 

    pub fn forward(self: *Neuron, input: f32) f32 {
        self.input = input;
        self.output = input * self.weight;
        return self.output;
    }

};
