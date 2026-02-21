const std = @import("std");
const expect = std.testing.expect;

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
        else => @compileError("Unsupported type for BLAS: " ++ @typeName(T)),
    }
}

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
    matMul(2, 2, 3, 1.0, &a, &b, 0.0, &res);

    try expectEqualSlices(f32, &expected, &res);
}

test "matMul - f64 (dgemm) validation" {
    const a = [_]f64{ 2.0, 0.0, 0.0, 2.0 }; // 2x2 Identity * 2
    const b = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var res = [_]f64{ 0.0, 0.0, 0.0, 0.0 };
    const expected = [_]f64{ 2.0, 4.0, 6.0, 8.0 };

    matMul(2, 2, 2, 1.0, &a, &b, 0.0, &res);

    try expectEqualSlices(f64, &expected, &res);
}
