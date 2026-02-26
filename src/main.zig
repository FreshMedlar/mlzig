const std = @import("std");
const mlzig = @import("mlzig");
const meta = @import("meta.zig");

pub fn main() !void {
    var pop = meta.Population{}; 
    pop.init();
    std.debug.print("Done!\n", .{});



}

