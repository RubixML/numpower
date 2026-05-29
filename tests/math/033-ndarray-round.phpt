--TEST--
NumPower::round
--FILE--
<?php
/* round uses round-half-to-even (banker's rounding), matching PyTorch
   `torch.round` and NumPy `np.round`: -156.5 rounds to the even neighbour
   -156 (not -157, which the legacy half-away `roundf` kernel produced). */
$a = NumPower::array([[-156.50, 150.525435], [0, -39.151414]]);
print_r(NumPower::round($a, precision: 0)->toArray());
print_r(NumPower::round($a[0], precision: 1)->toArray());
print_r(NumPower::round([[0.12],[-0.513124]], precision: 2)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => -156
            [1] => 151
        )

    [1] => Array
        (
            [0] => 0
            [1] => -39
        )

)
Array
(
    [0] => -156.5
    [1] => 150.5
)
Array
(
    [0] => Array
        (
            [0] => 0.11999999731779
        )

    [1] => Array
        (
            [0] => -0.50999999046326
        )

)