--TEST--
NDArray Iterator: explicit rewind/valid/key/current/next on CPU across all dtypes
--FILE--
<?php
/* Drives the Iterator interface methods directly (not via foreach) to verify
   that each one returns the right thing on every supported dtype. current()
   on a 1-D source must yield the dtype-correct PHP scalar:
     - int for int8..int64 and uint8..uint32
     - string for uint64 / float128 (carry full precision via decimal string)
     - float for float4..float64
   key() must increment from 0; valid() must flip to false past the end. */

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
    /* String inputs preserve precision for the str-IO dtypes. */
    $str_io = in_array($t, ['float4','float8','float16','float128','int64','uint64'], true);
    $vals   = $str_io ? ['1','2','3','4'] : [1,2,3,4];
    $a      = new NDArray($vals, $t);

    $a->rewind();
    $collected_keys = [];
    $collected_vals = [];
    $type_ok = true;
    while ($a->valid()) {
        $k = $a->key();
        $v = $a->current();
        $collected_keys[] = $k;
        $collected_vals[] = is_string($v) ? $v : (string)$v;
        if ($want_type === 'int'    && !is_int($v))    { $type_ok = false; }
        if ($want_type === 'float'  && !is_float($v))  { $type_ok = false; }
        if ($want_type === 'string' && !is_string($v)) { $type_ok = false; }
        $a->next();
    }
    /* Past the end: valid() is false; current() is null per Iterator contract. */
    $past_end_valid = $a->valid();
    $past_end_curr  = $a->current();

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
