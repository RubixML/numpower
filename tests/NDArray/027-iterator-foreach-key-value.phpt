--TEST--
NDArray Iterator: foreach ($a as $k =&gt; $v) pairs keys and values correctly
--FILE--
<?php
/* foreach ($a as $key => $value) drives the Iterator interface in the
   sequence rewind / valid / key / current / next. We assert the pairing:
     - for 1-D source, $key is the axis-0 index and $value is the dtype scalar
     - for 2-D source, $value is an NDArray and shape() matches axis-0[k] */

$a = new NDArray([10.5, 20.5, 30.5, 40.5], 'float64');
$pairs = [];
foreach ($a as $k => $v) {
    $pairs[] = "$k=>$v";
}
echo "1D float64: ", implode(',', $pairs), "\n";

$b = new NDArray(['1', '2', '3'], 'uint64');
$pairs = [];
foreach ($b as $k => $v) {
    $pairs[] = "$k=>" . (is_string($v) ? "'$v'" : "?$v");
}
echo "1D uint64:  ", implode(',', $pairs), "\n";

$c = NumPower::array([[1, 2, 3], [4, 5, 6]], 'int32');
$pairs = [];
foreach ($c as $k => $row) {
    if (!($row instanceof NDArray)) { $pairs[] = "$k=>NOT_NDARRAY"; continue; }
    $pairs[] = "$k=>" . json_encode($row->toArray());
}
echo "2D int32:   ", implode(',', $pairs), "\n";

/* foreach without the $k => binding only yields values. */
$vs = [];
foreach ($a as $v) { $vs[] = (string)$v; }
echo "without key: ", implode(',', $vs), "\n";
?>
--EXPECT--
1D float64: 0=>10.5,1=>20.5,2=>30.5,3=>40.5
1D uint64:  0=>'1',1=>'2',2=>'3'
2D int32:   0=>[1,2,3],1=>[4,5,6]
without key: 10.5,20.5,30.5,40.5
