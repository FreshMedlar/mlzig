const std = @import("std");
const matrix = @import("matrix.zig");
const math = @import("./math/math.zig");

pub fn forward(
    input: anytype,
    weights: anytype,
) matrix.StaticMatrix(@TypeOf(weights.*).Type, @TypeOf(input.*).row_count, @TypeOf(weights.*).col_count) {
    var output = math.matMulStatic(@TypeOf(input), @TypeOf(weights), input, weights);
    math.relu(@TypeOf(output), &output);

    return output;
}
