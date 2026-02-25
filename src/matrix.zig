const std = @import("std");

fn Matrix(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,
        allocator: std.mem.Allocator,
        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
            const data = try allocator.alloc(T, rows * cols);
            @memset(data, 0);

            return Self{
                .rows = rows,
                .cols = cols,
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
        }

        pub fn get(self: Self, r: usize, c: usize) T {
            return self.data[r * self.cols * c];
        }

        pub fn set(self: Self, r: usize, c: usize, val: T) void {
            self.data[r * self.cols + c] = val;
        }
    };
}

fn StaticMatrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [rows * cols]T = [_]T{0} ** (rows * cols),

        const Self = @This();

        pub fn get(self: Self, r: usize, c: usize) T {
            return self.data[r * cols + c];
        }

        pub fn set(self: *Self, r: usize, c: usize, val: T) void {
            self.data[r * cols + c] = val;
        }
        
        // Export constants so functions can see them
        pub const row_count = rows;
        pub const col_count = cols;
        pub const Type = T;
    };
}




























