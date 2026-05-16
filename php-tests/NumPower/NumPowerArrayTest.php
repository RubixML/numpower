<?php

declare(strict_types=1);

namespace Numpower\Tests\NumPower;

use NDArray;
use NumPower;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;
use Throwable;

class NumPowerArrayTest extends TestCase
{
    public function testDefaultDtypeIsFloat64(): void
    {
        $nd = NumPower::array([1, 2.45, 9.2234567890987654]);

        $this->assertEquals(1, $nd[0]);
        $this->assertEquals(2.45, $nd[1]);
        $this->assertEquals(9.2234567890987655, $nd[2]);
    }

    public function testFloat32Dtype(): void
    {
        $nd = NumPower::array([1, 2.45, 3.1234567890987654], 'float32');

        $this->assertEquals(1, $nd[0]);
        $this->assertEquals(2.450000047683716, $nd[1]);
        $this->assertEquals(3.1234567165374756, $nd[2]);
    }

    public function testFloat64Dtype(): void
    {
        $nd = NumPower::array([1, 2.45, 9.2234567890987654], 'float64');

        $this->assertEquals(1, $nd[0]);
        $this->assertEquals(2.45, $nd[1]);
        $this->assertEquals(9.2234567890987655, $nd[2]);
    }

    public function testFloat32VsFloat64Precision(): void
    {
        $f32 = NumPower::array([2.45], 'float32');
        $f64 = NumPower::array([2.45], 'float64');

        $this->assertNotEquals($f32[0], $f64[0]);
        $this->assertEquals(2.450000047683716, $f32[0]);
        $this->assertEquals(2.45, $f64[0]);
    }

    public function testFloat32_2D(): void
    {
        $nd = NumPower::array([[1, 2.45, 3.1234567890987654], [1, 2.45, 3.1234567890987654]], 'float32');

        $this->assertEquals(2.450000047683716, $nd[0][1]);
        $this->assertEquals(3.1234567165374756, $nd[0][2]);
        $this->assertEquals(2.450000047683716, $nd[1][1]);
        $this->assertEquals(3.1234567165374756, $nd[1][2]);
    }

    public function testFloat64_2D(): void
    {
        $nd = NumPower::array([[1, 2.45, 9.2234567890987654], [1, 2.45, 9.2234567890987654]], 'float64');

        $this->assertEquals(2.45, $nd[0][1]);
        $this->assertEquals(9.2234567890987655, $nd[0][2]);
        $this->assertEquals(2.45, $nd[1][1]);
        $this->assertEquals(9.2234567890987655, $nd[1][2]);
    }

    #[DataProvider('validDtypeProvider')]
    public function testValidDtypesDoNotThrow(string $dtype): void
    {
        $nd = NumPower::array([1, 2, 3], $dtype);
        $this->assertEquals(1, $nd[0]);
        $this->assertEquals(2, $nd[1]);
        $this->assertEquals(3, $nd[2]);
    }

    public static function validDtypeProvider(): array
    {
        return [
            'float32'  => ['float32'],
            'float64'  => ['float64'],
            'int8'     => ['int8'],
            'uint8'    => ['uint8'],
            'int16'    => ['int16'],
            'uint16'   => ['uint16'],
            'int32'    => ['int32'],
            'uint32'   => ['uint32'],
            'int64'    => ['int64'],
            'uint64'   => ['uint64'],
        ];
    }

    public function testInvalidDtypeThrowsError(): void
    {
        $this->expectException(Throwable::class);
        $this->expectExceptionMessage("Invalid data type 'badtype'.");

        NumPower::array([1, 2, 3], 'badtype');
    }

    public function testInt32BoundaryValues(): void
    {
        $nd = NumPower::array([0, 1, -2147483648, 2147483647], 'int32');

        $this->assertEquals(0, $nd[0]);
        $this->assertEquals(1, $nd[1]);
        $this->assertEquals(-2147483648, $nd[2]);
        $this->assertEquals(2147483647, $nd[3]);
    }

    public function testUint8BoundaryValues(): void
    {
        $nd = NumPower::array([0, 127, 255], 'uint8');

        $this->assertEquals(0, $nd[0]);
        $this->assertEquals(127, $nd[1]);
        $this->assertEquals(255, $nd[2]);
    }

    public function testNDArrayInputIsReturnedUnchanged(): void
    {
        $a = NumPower::array([1, 2, 3], 'int32');
        $b = NumPower::array($a);

        $this->assertSame($a, $b);
    }

    public function testNDArrayInputDtypeParamIgnored(): void
    {
        $a = NumPower::array([1, 2, 3], 'int32');
        $b = NumPower::array($a, 'float64');

        $this->assertSame($a, $b);
    }
}
