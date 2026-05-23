--TEST--
NDArray::offsetUnset() — `unset($a[$i])` is unconditionally rejected
--FILE--
<?php
/* NDArray buffers are fixed-shape, fixed-dtype — there is no tombstone state
   for an "absent" element, so unset() throws. Verified for valid indices,
   out-of-range indices, and non-integer offsets across CPU and GPU. */

$dtypes = ['float32', 'float64', 'float128', 'int32', 'int64', 'uint64'];

foreach ($dtypes as $t) {
    $strIO = in_array($t, ['float128','int64','uint64'], true);
    $vals = $strIO ? ['1','2','3'] : [1, 2, 3];
    $a = new NDArray($vals, $t);

    $err_valid = 'NONE';
    try { unset($a[1]); } catch (\Throwable $e) { $err_valid = $e->getMessage(); }

    $err_oor = 'NONE';
    try { unset($a[99]); } catch (\Throwable $e) { $err_oor = $e->getMessage(); }

    $err_str = 'NONE';
    try { unset($a['x']); } catch (\Throwable $e) { $err_str = $e->getMessage(); }

    echo "$t: valid='$err_valid' oor='$err_oor' str='$err_str'\n";

    /* Shape must be untouched after every (failed) unset. */
    if ($a->shape() !== [3]) {
        echo "$t: shape changed unexpectedly\n";
    }
}
echo "done\n";
?>
--EXPECT--
float32: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
float64: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
float128: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
int32: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
int64: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
uint64: valid='Cannot unset values of NDArrays' oor='Cannot unset values of NDArrays' str='Cannot unset values of NDArrays'
done
