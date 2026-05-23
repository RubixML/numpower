--TEST--
NDArray Iterator: explicit driver on GPU across all dtypes (parity with CPU)
--SKIPIF--
<?php
try { (new NDArray([1.0]))->gpu(); } catch (\Error $e) { die('skip ' . $e->getMessage()); }
?>
--FILE--
<?php
/* NDArray::current() must produce the same dtype-correct PHP scalar regardless
   of whether the source array lives on CPU RAM or GPU VRAM. The fp128 path
   takes an extra detour on GPU (storage is double-double, ScalarToZval
   reassembles it before formatting) so we exercise both 14 dtypes here. */

$dtypes = [
    'float4'   => 'float',  'float8'  => 'float',  'float16' => 'float',
    'float32'  => 'float',  'float64' => 'float',
    'float128' => 'string',
    'int8'     => 'int',    'uint8'   => 'int',
    'int16'    => 'int',    'uint16'  => 'int',
    'int32'    => 'int',    'uint32'  => 'int',
    'int64'    => 'int',
    'uint64'   => 'string',
];

foreach ($dtypes as $t => $want_type) {
    $str_io = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals   = $str_io ? ['1','2','3','4'] : [1,2,3,4];
    $g      = (new NDArray($vals, $t))->gpu();

    $g->rewind();
    $collected_keys = [];
    $collected_vals = [];
    $type_ok = true;
    while ($g->valid()) {
        $k = $g->key();
        $v = $g->current();
        $collected_keys[] = $k;
        $collected_vals[] = is_string($v) ? $v : (string)$v;
        if ($want_type === 'int'    && !is_int($v))    { $type_ok = false; }
        if ($want_type === 'float'  && !is_float($v))  { $type_ok = false; }
        if ($want_type === 'string' && !is_string($v)) { $type_ok = false; }
        $g->next();
    }
    $past_end_valid = $g->valid();
    $past_end_curr  = $g->current();

    printf("%s: keys=[%s] vals=[%s] type=%s past=valid:%s,current:%s\n",
        $t,
        implode(',', $collected_keys),
        implode(',', $collected_vals),
        $type_ok ? 'OK' : 'BAD',
        $past_end_valid ? 'true' : 'false',
        $past_end_curr === null ? 'null' : gettype($past_end_curr));
}
?>
--EXPECT--
float4: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
float8: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
float16: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
float32: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
float64: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
float128: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
int8: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
uint8: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
int16: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
uint16: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
int32: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
uint32: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
int64: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
uint64: keys=[0,1,2,3] vals=[1,2,3,4] type=OK past=valid:false,current:null
