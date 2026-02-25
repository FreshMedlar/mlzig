const std = @import("std");
const matrix = @import("matrix.zig");

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const c = @cImport({@cInclude("cblas.h");
});

pub fn matMul(
   m: i32, n: i32, k: i32,
   alpha: anytype,
   a: []const @TypeOf(alpha),
   b: []const @TypeOf(alpha),
   beta: @TypeOf(alpha),
   res: []@TypeOf(alpha),
   ) void {
    const T = @TypeOf(alpha);

    switch (T) {
        f32 => {
            c.cblas_sgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
                m, n, k, alpha, a.ptr, k, b.ptr, n, beta, res.ptr, n);
        },
        f64 => {
            c.cblas_dgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans,
                m, n, k, alpha, a.ptr, k, b.ptr, n, beta, res.ptr, n);
        },
        i8 => {
            matMulStatic(a, b);
        },
        else => @compileError("Unsupported type for BLAS: " ++ @typeName(T)),
    }
}

pub fn matMulStatic(
    comptime A: type, 
    comptime B: type, 
    a: A, 
    b: B
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

test "matMul - f32 (sgemm) validation" {
    // Matrix A (2x3)
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    // Matrix B (3x2)
    // [ 7.0,  8.0 ]
    // [ 9.0,  10.0 ]
    // [ 11.0, 12.0 ]
    const b = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    // Result C (2x2) = A * B
    // [ (1*7 + 2*9 + 3*11), (1*8 + 2*10 + 3*12) ]  => [ 58, 64 ]
    // [ (4*7 + 5*9 + 6*11), (4*8 + 5*10 + 6*12) ]  => [139, 154]
    var res = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const expected = [_]f32{ 58.0, 64.0, 139.0, 154.0 };

    // Call your dispatch function
    // Parameters: m=2, n=2, k=3
    matMul(2, 2, 3, @as(f32, 1.0), &a, &b, @as(f64, 0.0), &res);

    try expectEqualSlices(f32, &expected, &res);
}

test "matMul - f64 (dgemm) validation" {
    const a = [_]f64{ 2.0, 0.0, 0.0, 2.0 }; // 2x2 Identity * 2
    const b = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var res = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const expected = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    matMul(2, 2, 2, @as(f64, 1.0), &a, &b, @as(f64, 0.0), &res);

    try expectEqualSlices(f64, &expected, &res);
}

test "matMul - i8 (Manual)" {
    const m, const n, const k = .{ 2, 2, 2 };
    
    // Using small values to prevent i8 overflow in your current code
    const a = [_]i8{ 1, 2, 3, 4 };
    const b = [_]i8{ 1, 0, 0, 1 }; // Identity matrix
    var res = [_]i8{ 0, 0, 0, 0 };

    matMul(m, n, k, @as(i8, 1), &a, &b, @as(i8, 0), &res);

    try expectEqual(res[0], 1);
    try expectEqual(res[1], 2);
    try expectEqual(res[2], 3);
    try expectEqual(res[3], 4);
}
