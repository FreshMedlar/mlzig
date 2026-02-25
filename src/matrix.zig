const std = @import("std");

pub fn StaticMatrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [rows * cols]T = [_]T{0} ** (rows * cols),

        const Self = @This();

        pub fn get(self: Self, r: usize, c: usize) T {
            return self.data[r * cols + c];
        }

        pub fn set(self: *Self, r: usize, c: usize, val: T) void {
            self.data[r * cols + c] = val;
        }
        
        pub const row_count = rows;
        pub const col_count = cols;
        pub const Type = T;
    };
}























