--TEST--
NumPower::setDevice(false) coerces to 0 and either succeeds or throws cleanly
--FILE--
<?php
/* Boolean false coerces to int 0 in PHP loose mode. On a CUDA build
   with device 0 visible, setDevice(false) succeeds. On a CPU-only
   build, setDevice(false) throws the "CUDA not enabled" error. Either
   way, false must NOT segfault and must NOT silently no-op (the latter
   was the pre-fix behavior on a CPU build). */
$cuda_compiled = true;
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { $cuda_compiled = false; }

try {
    NumPower::setDevice(false);
    if ($cuda_compiled) {
        echo "ok\n";  /* succeeded on CUDA build, that's correct */
    } else {
        echo "FAIL: no throw on no-CUDA build\n";
    }
} catch (\Error $e) {
    $msg = $e->getMessage();
    if ($cuda_compiled) {
        /* Unexpected — false → 0 should be valid on any CUDA box. */
        echo "FAIL: unexpected throw on CUDA build: $msg\n";
    } else {
        /* Expected: CUDA not enabled. */
        $ok = strpos($msg, 'CUDA not enabled') !== false
            || strpos($msg, 'No GPU device available') !== false;
        echo $ok ? "ok\n" : "FAIL: $msg\n";
    }
}
?>
--EXPECT--
ok
