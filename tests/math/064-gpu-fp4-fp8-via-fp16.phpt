--TEST--
GPU float4 / float8 arithmetic stays on GPU (casts through float16 on GPU, no CPU staging)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* float4 (E2M1) and float8 (E4M3) compute on GPU goes through float16 (both
   formats fit losslessly: fp4 has 8 positive values up to 6, fp8 has ~3
   mantissa bits with max 240 — well below float16's mantissa-10/exp-5 range).

   The result dtype is preserved as the source dtype thanks to the cast-back
   step. Values are quantised to the format's representable set on each
   write — this is inherent to the format, not a GPU artifact. */

/* fp4 same-dtype binop */
$a = NumPower::array(['1', '2', '3', '4'], 'float4')->gpu();
$b = NumPower::array(['0.5', '1', '1.5', '2'], 'float4')->gpu();
$r = $a + $b;
echo "fp4 a+b isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";

$r = $a * $b;
echo "fp4 a*b isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";

/* fp8 same-dtype binop */
$a = NumPower::array(['1', '2', '3', '4'], 'float8')->gpu();
$b = NumPower::array(['0.5', '0.5', '0.5', '0.5'], 'float8')->gpu();
$r = $a + $b;
echo "fp8 a+b isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";

$r = $a - $b;
echo "fp8 a-b isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";

/* fp4 + scalar (weak-scalar makes scalar fp4) */
$a = NumPower::array(['1', '2', '3', '4'], 'float4')->gpu();
$r = $a + 1;
echo "fp4 a+1 isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";

/* fp8 / 2 promotes to float (PyTorch true division). Result is fp32 on GPU. */
$a = NumPower::array(['8', '16', '32'], 'float8')->gpu();
$r = $a / 2;
echo "fp8 / 2 isGPU=", $r->isGPU(), " val=", json_encode($r->cpu()->toArray()), "\n";
?>
--EXPECT--
fp4 a+b isGPU=1 val=[1.5,3,4,6]
fp4 a*b isGPU=1 val=[0.5,2,4,6]
fp8 a+b isGPU=1 val=[1.5,2.5,3.5,4.5]
fp8 a-b isGPU=1 val=[0.5,1.5,2.5,3.5]
fp4 a+1 isGPU=1 val=[2,3,4,4]
fp8 / 2 isGPU=1 val=[4,8,16]
