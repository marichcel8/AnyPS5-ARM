constant uint ELEMENT_BYTES [[function_constant(0)]];
constant uint TRANSFORM [[function_constant(1)]];
constant uint TEXEL_BYTES [[function_constant(2)]];

inline bool agc_locate(constant AgcDetileParams& p, uint2 gid, thread uint& x, thread uint& y, thread uint& offset) {
    x = gid.x + p.columnBegin;
    y = gid.y + p.rowBegin;
    if (x >= p.width || y >= p.height) return false;
    offset = agc_tiled_offset(p, x, y, ELEMENT_BYTES);
    return offset >= p.rangeBegin && offset < p.rangeEnd;
}

kernel void aps5_tiled_to_array2d(constant AgcDetileParams& p [[buffer(0)]], texture2d_array<uint, access::write> destination [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    const uint4 value = agc_transform(agc_load_element(p.tiled, offset, ELEMENT_BYTES), TRANSFORM);
    destination.write(value, uint2(x, y), p.destinationSlice);
}

kernel void aps5_tiled_to_array1d(constant AgcDetileParams& p [[buffer(0)]], texture1d_array<uint, access::write> destination [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    const uint4 value = agc_transform(agc_load_element(p.tiled, offset, ELEMENT_BYTES), TRANSFORM);
    destination.write(value, x, p.destinationSlice);
}

kernel void aps5_tiled_to_volume(constant AgcDetileParams& p [[buffer(0)]], texture3d<uint, access::write> destination [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    const uint4 value = agc_transform(agc_load_element(p.tiled, offset, ELEMENT_BYTES), TRANSFORM);
    destination.write(value, uint3(x, y, p.destinationSlice));
}

kernel void aps5_array2d_to_tiled(constant AgcDetileParams& p [[buffer(0)]], texture2d_array<uint, access::read> source [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    agc_store_element(p.tiled, offset, ELEMENT_BYTES, agc_untransform(source.read(uint2(x, y), p.destinationSlice), TRANSFORM));
}

kernel void aps5_volume_to_tiled(constant AgcDetileParams& p [[buffer(0)]], texture3d<uint, access::read> source [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    agc_store_element(p.tiled, offset, ELEMENT_BYTES, agc_untransform(source.read(uint3(x, y, p.destinationSlice)), TRANSFORM));
}

kernel void aps5_tiled_to_linear(constant AgcDetileParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    const uint4 value = agc_transform(agc_load_element(p.tiled, offset, ELEMENT_BYTES), TRANSFORM);
    agc_store_element(p.linear, y * p.linearPitchBytes + x * TEXEL_BYTES, TEXEL_BYTES, value);
}

kernel void aps5_linear_to_tiled(constant AgcDetileParams& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint x, y, offset;
    if (!agc_locate(p, gid, x, y, offset)) return;
    agc_store_element(p.tiled, offset, ELEMENT_BYTES, agc_load_element(p.linear, y * p.linearPitchBytes + x * ELEMENT_BYTES, ELEMENT_BYTES));
}
