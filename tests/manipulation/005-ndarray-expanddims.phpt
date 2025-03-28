--TEST--
NDArray::expandDims
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([[1, 2, 3, 4]]);
$b = nd::array([[5, 6], [7, 8]]);
$c = nd::array([[[5, 6], [7, 8]], [[5, 6], [7, 8]]]);
$d = nd::array([1, 2, 3, 4]);

print_r(nd::expandDims(1, 0)->toArray());

print_r(nd::expandDims([[1, 2, 3, 4]], 0)->toArray());
print_r(nd::expandDims($a, -1)->toArray());
print_r(nd::expandDims($a, -2)->toArray());
print_r(nd::expandDims($a, 1)->toArray());

print_r(nd::expandDims($b, 0)->toArray());
print_r(nd::expandDims($b, -1)->toArray());
print_r(nd::expandDims($b, -2)->toArray());
print_r(nd::expandDims($b, 1)->toArray());

print_r(nd::expandDims($c, 0)->toArray());
print_r(nd::expandDims($c, -1)->toArray());
print_r(nd::expandDims($c, -2)->toArray());
print_r(nd::expandDims($c,  1)->toArray());

print_r(nd::expandDims($c, [0, -1, 1])->toArray());
print_r(nd::expandDims($c, [0, -1])->toArray());
print_r(nd::expandDims($c, [2, 1, 0])->toArray());
?>
--EXPECT--
Array
(
    [0] => 1
)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 1
                    [1] => 2
                    [2] => 3
                    [3] => 4
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 1
                )

            [1] => Array
                (
                    [0] => 2
                )

            [2] => Array
                (
                    [0] => 3
                )

            [3] => Array
                (
                    [0] => 4
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 1
                    [1] => 2
                    [2] => 3
                    [3] => 4
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 1
                    [1] => 2
                    [2] => 3
                    [3] => 4
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 5
                    [1] => 6
                )

            [1] => Array
                (
                    [0] => 7
                    [1] => 8
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 5
                )

            [1] => Array
                (
                    [0] => 6
                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => 7
                )

            [1] => Array
                (
                    [0] => 8
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 5
                    [1] => 6
                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => 7
                    [1] => 8
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => 5
                    [1] => 6
                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => 7
                    [1] => 8
                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                    [1] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                    [1] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                        )

                    [1] => Array
                        (
                            [0] => 6
                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => 7
                        )

                    [1] => Array
                        (
                            [0] => 8
                        )

                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                        )

                    [1] => Array
                        (
                            [0] => 6
                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => 7
                        )

                    [1] => Array
                        (
                            [0] => 8
                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                    [1] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

        )

    [1] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => 5
                            [1] => 6
                        )

                    [1] => Array
                        (
                            [0] => 7
                            [1] => 8
                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 5
                                        )

                                    [1] => Array
                                        (
                                            [0] => 6
                                        )

                                )

                            [1] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 7
                                        )

                                    [1] => Array
                                        (
                                            [0] => 8
                                        )

                                )

                        )

                    [1] => Array
                        (
                            [0] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 5
                                        )

                                    [1] => Array
                                        (
                                            [0] => 6
                                        )

                                )

                            [1] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 7
                                        )

                                    [1] => Array
                                        (
                                            [0] => 8
                                        )

                                )

                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => Array
                                (
                                    [0] => 5
                                )

                            [1] => Array
                                (
                                    [0] => 6
                                )

                        )

                    [1] => Array
                        (
                            [0] => Array
                                (
                                    [0] => 7
                                )

                            [1] => Array
                                (
                                    [0] => 8
                                )

                        )

                )

            [1] => Array
                (
                    [0] => Array
                        (
                            [0] => Array
                                (
                                    [0] => 5
                                )

                            [1] => Array
                                (
                                    [0] => 6
                                )

                        )

                    [1] => Array
                        (
                            [0] => Array
                                (
                                    [0] => 7
                                )

                            [1] => Array
                                (
                                    [0] => 8
                                )

                        )

                )

        )

)
Array
(
    [0] => Array
        (
            [0] => Array
                (
                    [0] => Array
                        (
                            [0] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 5
                                            [1] => 6
                                        )

                                    [1] => Array
                                        (
                                            [0] => 7
                                            [1] => 8
                                        )

                                )

                            [1] => Array
                                (
                                    [0] => Array
                                        (
                                            [0] => 5
                                            [1] => 6
                                        )

                                    [1] => Array
                                        (
                                            [0] => 7
                                            [1] => 8
                                        )

                                )

                        )

                )

        )

)