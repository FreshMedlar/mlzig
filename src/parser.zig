const std = @import("std");
const fs = std.fs;
const io = std.io;
const print = std.debug.print;

pub const Reader = struct {
    length: usize = 0,
    vocab_len: usize = 0,


    pub fn read(self: *Reader, name: []const u8) !void {
        var file = try fs.cwd().openFile(name, .{});
        defer file.close();

        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        const file_size = (try file.stat()).size;
        const buffer = try file.readToEndAlloc(allocator, file_size);

        self.length = try std.unicode.utf8CountCodepoints(buffer);
        var vocab_map = std.AutoHashMap(u21, void).init(allocator);
        defer vocab_map.deinit();

        var utf8_view = try std.unicode.Utf8View.init(buffer);
        var iterator = utf8_view.iterator();

        while (iterator.nextCodepoint()) |codepoint| {
            try vocab_map.put(codepoint, {});
        }
        self.vocab_len = vocab_map.count();
        print("The text is {d} char long, with {d} different chars\n", .{self.length, self.vocab_len});
    }
};

