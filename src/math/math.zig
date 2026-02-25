const std = @import("std");
const matrix = @import("matrix.zig");

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;


pub fn matMulStatic(
    comptime A: type, 
    comptime B: type, 
    a: *const A, 
    b: *const B
) matrix.StaticMatrix(A.Type, A.row_count, B.col_count) {
    // Compile-time safety check
    comptime {
        if (A.col_count != B.row_count) {
            @compileError("Matrix dimensions do not match for multiplication!");
        }
    }

    const T = A.Type;
    var res = matrix.StaticMatrix(T, A.row_count, B.col_count){};

    for (0..A.row_count) |i| {
        for (0..B.col_count) |j| {
            var sum: i32 = 0; // Accumulator to prevent overflow
            for (0..A.col_count) |k| {
                sum += @as(i32, a.get(i, k)) * @as(i32, b.get(k, j));
            }
            // Logic to cast/clamp back to T goes here
            res.set(i, j, @intCast(std.math.clamp(sum, -128, 127)));
        }
    }
    return res;
}


// -----------------------------------------------------------------------------------------------


const expectEqualSlices = std.testing.expectEqualSlices;

test "Static Matrix i8 Multiplication" {
    const Mat2x3 = matrix.StaticMatrix(i8, 2, 3);
    const Mat3x2 = matrix.StaticMatrix(i8, 3, 2);

    var m1 = Mat2x3{};
    m1.set(0, 0, 1); m1.set(0, 1, 2); m1.set(0, 2, 3);
    m1.set(1, 0, 4); m1.set(1, 1, 5); m1.set(1, 2, 6);

    var m2 = Mat3x2{};
    m2.set(0, 0, 7); m2.set(0, 1, 8);
    m2.set(1, 0, 9); m2.set(1, 1, 10);
    m2.set(2, 0, 11); m2.set(2, 1, 12);

    // Call with pointers
    const result = matMulStatic(Mat2x3, Mat3x2, &m1, &m2);

    // Row 0, Col 0: (1*7 + 2*9 + 3*11) = 7 + 18 + 33 = 58
    try expectEqual(@as(i8, 58), result.get(0, 0));
    // Row 1, Col 1: (4*8 + 5*10 + 6*12) = 32 + 50 + 72 = 154 -> Clamped to 127
    try expectEqual(@as(i8, 127), result.get(1, 1));
}

test "Compile-time dimension check" {
    // This test demonstrates how Zig handles mismatched dimensions.
    // Uncommenting the lines below will cause the build to fail.
    // const M1 = matrix.StaticMatrix(i8, 2, 2);
    // const M2 = matrix.StaticMatrix(i8, 3, 3);
    // _ = matMulStatic(M1, M2, &M1{}, &M2{}); 
}






























