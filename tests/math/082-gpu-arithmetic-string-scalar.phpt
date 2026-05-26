--TEST--
GPU arithmetic accepts string scalars and keeps the op on GPU for fp128 / int64 / uint64
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* The string-scalar path must:
   1. Encode the string in the peer NDArray's dtype with full precision
      (NDArray_EncodeZvalToDtype: strtoflt128 / strtoull / strtoll).
   2. Migrate the freshly-built CPU 0-D scalar to GPU before the typed
      kernel runs (so the binop stays on device).
   3. Produce the same result as the equivalent CPU path. */

/* float128 GPU + string fp128 scalar stays on GPU and matches CPU. */
$f_cpu = new NDArray(['1.5e30', '2.5e30'], 'float128');
$f_gpu = $f_cpu->gpu();
$r_cpu = NumPower::add($f_cpu, '1e30');
$r_gpu = NumPower::add($f_gpu, '1e30');
if (!$r_gpu->isGPU()) { echo "fp128+str not on GPU\n"; }
if ((string)$r_cpu[0] !== (string)$r_gpu[0]) {
    echo "fp128 CPU/GPU disagree: cpu=", (string)$r_cpu[0], " gpu=", (string)$r_gpu[0], "\n";
}
echo "fp128 add: ", (string)$r_gpu[0], "\n";

/* uint64 GPU + string uint64 scalar — native GPU kernel keeps values precise. */
$u_cpu = new NDArray(['18446744073709551610', '5'], 'uint64');
$u_gpu = $u_cpu->gpu();
$r_cpu = NumPower::add($u_cpu, '5');
$r_gpu = NumPower::add($u_gpu, '5');
if (!$r_gpu->isGPU()) { echo "u64+str not on GPU\n"; }
if ((string)$r_cpu[0] !== (string)$r_gpu[0]) {
    echo "u64 CPU/GPU disagree: cpu=", (string)$r_cpu[0], " gpu=", (string)$r_gpu[0], "\n";
}
echo "u64 add: ", (string)$r_gpu[0], "\n";

/* int64 GPU + string int64 scalar — native GPU kernel. */
$i_cpu = new NDArray(['1000000000000000000', '-1000000000000000000'], 'int64');
$i_gpu = $i_cpu->gpu();
$r_cpu = NumPower::subtract($i_cpu, '500000000000000000');
$r_gpu = NumPower::subtract($i_gpu, '500000000000000000');
if (!$r_gpu->isGPU()) { echo "i64-str not on GPU\n"; }
if ((string)$r_cpu[0] !== (string)$r_gpu[0]) {
    echo "i64 CPU/GPU disagree: cpu=", (string)$r_cpu[0], " gpu=", (string)$r_gpu[0], "\n";
}
echo "i64 sub: ", (string)$r_gpu[0], "\n";

/* Operator overload on GPU NDArrays accepts string scalars. */
$u_gpu = (new NDArray(['18446744073709551600'], 'uint64'))->gpu();
$r = $u_gpu + 10;
if (!$r->isGPU()) { echo "u64 + 10 not on GPU\n"; }
echo "u64 op+10: ", (string)$r[0], "\n";

/* Reverse direction: scalar + NDArray. */
$f_gpu = (new NDArray(['1e29'], 'float128'))->gpu();
$r = NumPower::add('5e29', $f_gpu);
if (!$r->isGPU()) { echo "str + fp128 not on GPU\n"; }
echo "str+fp128: ", (string)$r[0], "\n";

/* Multiply / mod / pow string scalars all route through the same dispatch. */
$u_gpu = (new NDArray(['1000', '2000'], 'uint64'))->gpu();
$r = NumPower::multiply($u_gpu, '1000000000000000');
if (!$r->isGPU()) { echo "u64 * not on GPU\n"; }
echo "u64 mul: ", (string)$r[0], "\n";

$i_gpu = (new NDArray(['10000000000'], 'int64'))->gpu();
$r = NumPower::mod($i_gpu, '7');
echo "i64 mod: ", (string)$r[0], "\n";

$i_gpu = (new NDArray([10], 'int64'))->gpu();
$r = NumPower::pow($i_gpu, '15');
echo "i64 pow: ", (string)$r[0], "\n";
?>
--EXPECT--
fp128 add: 2500000000000000000000000000000
u64 add: 18446744073709551615
i64 sub: 500000000000000000
u64 op+10: 18446744073709551610
str+fp128: 600000000000000000000000000000
u64 mul: 1000000000000000000
i64 mod: 4
i64 pow: 1000000000000000
