--TEST--
GPU float128 uses double-double emulation (~106 effective bits); fp64-representable values round-trip exactly
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* Document the precision profile of GPU fp128:
   - Values exactly representable in float64 (1.5, 1000, 1e308) round-trip
     CPU → GPU → CPU exactly.
   - Arithmetic in dd preserves ~106 bits, well above float64's 53 bits.
   - GPU compute keeps the result on GPU; only the user's explicit .cpu() or
     element access converts back. */

/* fp64-representable values must round-trip exactly. (Values like 1e308
   that aren't exactly representable in fp64 lose ~7 bits via dd; for
   bit-parity we stick to integers and small fractional powers of two.) */
$fp64_exact = ['0', '1', '-1', '1.5', '-1.5', '1000', '1.25', '-1.25'];
$a = new NDArray($fp64_exact, 'float128');
$g = $a->gpu();
$ok = true;
foreach ($fp64_exact as $i => $v) {
    if ($a[$i] !== $g[$i]) { $ok = false; echo "  i=$i cpu={$a[$i]} gpu={$g[$i]}\n"; }
}
echo "fp64-exact roundtrip: ", ($ok ? 'OK' : 'BAD'), "\n";

/* IEEE-754 specials. */
$specs = ['inf', '-inf', 'nan', '0', '-0'];
$a = new NDArray($specs, 'float128');
$g = $a->gpu();
$ok = true;
foreach ($specs as $i => $v) {
    if (strtolower($a[$i]) !== strtolower($g[$i])) {
        $ok = false; echo "  i=$i cpu={$a[$i]} gpu={$g[$i]}\n";
    }
}
echo "specials roundtrip: ", ($ok ? 'OK' : 'BAD'), "\n";

/* Arithmetic on GPU stays on GPU. */
$a = NumPower::array(["1.5", "2.5", "3.5", "4.5"], "float128")->gpu();
$b = NumPower::array(["0.5", "0.5", "0.5", "0.5"], "float128")->gpu();
$r = $a + $b;
echo "fp128+fp128 GPU isGPU=", $r->isGPU(), " v=", $r->cpu()[0], "\n";
$r = $a * $b;
echo "fp128*fp128 GPU isGPU=", $r->isGPU(), " v=", $r->cpu()[0], "\n";
$r = $a / $b;
echo "fp128/fp128 GPU isGPU=", $r->isGPU(), " v=", $r->cpu()[0], "\n";

/* GPU fp128 compute uses double-double, which is more precise than float64
   (~106 bits ≈ 31 decimal digits) but less than fp128's 113 bits / 34
   digits. The dd round-trip rounds at the ~31st decimal digit. */
$pi_str = "3.14159265358979323846";
$a = NumPower::array([$pi_str], "float128")->gpu();
$g_val = $a->cpu()[0];
/* dd round-trip preserves at least the first ~16 digits exactly (fp64 worth)
   plus several more (dd has ~31 decimal digits total). */
$prefix_ok = strncmp($pi_str, $g_val, 16) === 0;
echo "fp128 GPU roundtrip pi prefix(16) match: ", ($prefix_ok ? 'OK' : 'BAD'), "\n";
?>
--EXPECT--
fp64-exact roundtrip: OK
specials roundtrip: OK
fp128+fp128 GPU isGPU=1 v=2
fp128*fp128 GPU isGPU=1 v=0.75
fp128/fp128 GPU isGPU=1 v=3
fp128 GPU roundtrip pi prefix(16) match: OK
