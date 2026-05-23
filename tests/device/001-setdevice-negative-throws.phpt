--TEST--
NumPower::setDevice() rejects negative ids on every build
--FILE--
<?php
/* setDevice's range check used to be `id >= 0 && id > numDevices-1`, which
   silently accepted -1, -2, .... The fix changes the predicate to
   `id < 0 || id >= numDevices` so any negative id throws regardless of
   whether CUDA is present. */

$cuda_compiled = true;
try {
    (new NDArray([1.0]))->gpu();
} catch (\Error $e) {
    $cuda_compiled = false;
}

foreach ([-1, -2, PHP_INT_MIN] as $bad_id) {
    try {
        NumPower::setDevice($bad_id);
        echo "FAIL: setDevice($bad_id) returned without throwing\n";
    } catch (\Error $e) {
        $msg = $e->getMessage();
        if ($cuda_compiled) {
            /* The range check fires before cudaSetDevice. */
            if (strpos($msg, 'does not exist') !== false) {
                echo "ok: $bad_id\n";
            } else {
                echo "FAIL: $bad_id message: $msg\n";
            }
        } else {
            /* No-CUDA build throws the unavailable-runtime error. */
            if (strpos($msg, 'CUDA not enabled') !== false ||
                strpos($msg, 'No GPU device available') !== false) {
                echo "ok: $bad_id\n";
            } else {
                echo "FAIL: $bad_id message: $msg\n";
            }
        }
    }
}
?>
--EXPECT--
ok: -1
ok: -2
ok: -9223372036854775808
