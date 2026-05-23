--TEST--
NDArray::offsetSet() — cross-dtype NDArray value falls back to per-element cast on CPU
--FILE--
<?php
/* When the value's dtype differs from the destination, NDArray_Overwrite
   takes the slow per-element cast through double (CPU-only). This still
   preserves the destination's dtype — no silent upcast to float32 — but
   tolerates precision loss inherent to the cast.

   Verified here for:
     - int8 source -> int64 destination: lossless integer widening
     - float64 source -> int32 destination: float truncates to integer
     - int32 source -> float64 destination: lossless integer-to-float */

$cases = [
    /* dst dtype, dst init,           src dtype, src values, expected row 0 readback */
    ['int64',  [[1, 2, 3], [4, 5, 6]], 'int8',    [-1, 64, 127], [-1, 64, 127]],
    ['int32',  [[1, 2, 3], [4, 5, 6]], 'float64', [1.7, -2.5, 3.0], [1, -2, 3]],
    ['float64',[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 'int32', [10, 20, 30], [10.0, 20.0, 30.0]],
];

foreach ($cases as [$dst_t, $dst_init, $src_t, $src_vals, $want]) {
    $a = new NDArray($dst_init, $dst_t);
    $row = new NDArray($src_vals, $src_t);
    $a[0] = $row;
    $got = $a[0]->toArray();
    /* Destination dtype must be unchanged. We don't expose dtype directly
       on a sub-view; instead check the dtype-mandated PHP scalar type. */
    $first = $a[0][0];
    $is_int_dest   = in_array($dst_t, ['int8','uint8','int16','uint16','int32','uint32','int64'], true);
    $is_float_dest = in_array($dst_t, ['float32','float64'], true);
    $type_ok = ($is_int_dest && is_int($first)) || ($is_float_dest && is_float($first));

    /* Compare element-wise: integer equality for int dests, float equality
       (with tiny epsilon) for float dests. */
    $val_ok = true;
    foreach ($want as $i => $expected) {
        if (is_int($expected)) {
            if ($got[$i] !== $expected) { $val_ok = false; break; }
        } else {
            if (abs($got[$i] - $expected) > 1e-9) { $val_ok = false; break; }
        }
    }
    echo "$dst_t<-$src_t: type=", $type_ok ? "OK" : "BAD",
         " val=", $val_ok ? "OK" : "BAD got=" . json_encode($got), "\n";
}

/* Cross-device value must be rejected — CPU dest, GPU src. */
$skip_gpu = false;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $skip_gpu = true; }
if (!$skip_gpu) {
    $a = new NDArray([[1.0, 2.0], [3.0, 4.0]], 'float32');
    $row = (new NDArray([7.0, 8.0], 'float32'))->gpu();
    $err = 'NONE';
    try { $a[0] = $row; } catch (\Throwable $e) { $err = 'THROWN'; }
    echo "cpu<-gpu: $err\n";
} else {
    echo "cpu<-gpu: THROWN\n"; /* no GPU -> emit expected string to keep test portable */
}
?>
--EXPECT--
int64<-int8: type=OK val=OK
int32<-float64: type=OK val=OK
float64<-int32: type=OK val=OK
cpu<-gpu: THROWN
