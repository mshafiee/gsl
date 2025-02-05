package blas

import (
	"math"
	"testing"
)

func TestSdsdot(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	alpha := float32(2.0)

	expected := float32(1.0*4.0 + 2.0*5.0 + 3.0*6.0 + 2.0)
	result, err := Sdsdot(alpha, x, y)
	if err != nil {
		t.Fatalf("Sdsdot failed: %v", err)
	}

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}

	// Test with different strides.
	xStride2 := newVectorFloat(3, 2, []float32{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	yStride2 := newVectorFloat(3, 2, []float32{4.0, 0.0, 5.0, 0.0, 6.0, 0.0})

	resultStride, err := Sdsdot(alpha, xStride2, yStride2)
	if err != nil {
		t.Fatalf("Sdsdot with stride failed: %v", err)
	}
	if resultStride != expected {
		t.Errorf("Expected %f, but got %f (with stride)", expected, resultStride)
	}

	// Test with unequal vector sizes.
	xUnequal := newVectorFloat(2, 1, []float32{1.0, 2.0})
	yUnequal := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	_, errUnequal := Sdsdot(alpha, xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Sdsdot with unequal sizes should have returned an error")
	}
}

func TestDsdot(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})

	expected := float64(1.0*4.0 + 2.0*5.0 + 3.0*6.0)
	result, err := Dsdot(x, y)
	if err != nil {
		t.Fatalf("Dsdot failed: %v", err)
	}

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}

	// Test with different strides.
	xStride2 := newVectorFloat(3, 2, []float32{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	yStride2 := newVectorFloat(3, 2, []float32{4.0, 0.0, 5.0, 0.0, 6.0, 0.0})

	resultStride, err := Dsdot(xStride2, yStride2)
	if err != nil {
		t.Fatalf("Dsdot with stride failed: %v", err)
	}
	if resultStride != expected {
		t.Errorf("Expected %f, but got %f (with stride)", expected, resultStride)
	}

	// Test with unequal vector sizes.
	xUnequal := newVectorFloat(2, 1, []float32{1.0, 2.0})
	yUnequal := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	_, errUnequal := Dsdot(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Dsdot with unequal sizes should have returned an error")
	}
}

func TestSdot(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})

	expected := float32(1.0*4.0 + 2.0*5.0 + 3.0*6.0)
	result, err := Sdot(x, y)
	if err != nil {
		t.Fatalf("Sdot failed: %v", err)
	}

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}

	// Test with different strides.
	xStride2 := newVectorFloat(3, 2, []float32{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	yStride2 := newVectorFloat(3, 2, []float32{4.0, 0.0, 5.0, 0.0, 6.0, 0.0})

	resultStride, err := Sdot(xStride2, yStride2)
	if err != nil {
		t.Fatalf("Sdot with stride failed: %v", err)
	}
	if resultStride != expected {
		t.Errorf("Expected %f, but got %f (with stride)", expected, resultStride)
	}
	// Test with unequal vector sizes.
	xUnequal := newVectorFloat(2, 1, []float32{1.0, 2.0})
	yUnequal := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	_, errUnequal := Sdot(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Sdot with unequal sizes should have returned an error")
	}
}

func TestDdot(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})

	expected := float64(1.0*4.0 + 2.0*5.0 + 3.0*6.0)
	result, err := Ddot(x, y)
	if err != nil {
		t.Fatalf("Ddot failed: %v", err)
	}

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}

	// Test with different strides.
	xStride2 := newVector(3, 2, []float64{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	yStride2 := newVector(3, 2, []float64{4.0, 0.0, 5.0, 0.0, 6.0, 0.0})

	resultStride, err := Ddot(xStride2, yStride2)
	if err != nil {
		t.Fatalf("Ddot with stride failed: %v", err)
	}
	if resultStride != expected {
		t.Errorf("Expected %f, but got %f (with stride)", expected, resultStride)
	}
	// Test with unequal vector sizes.
	xUnequal := newVector(2, 1, []float64{1.0, 2.0})
	yUnequal := newVector(3, 1, []float64{4.0, 5.0, 6.0})
	_, errUnequal := Ddot(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Ddot with unequal sizes should have returned an error")
	}
}

func TestCdotu(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})
	y := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})

	expected := GslComplexFloat{Data: [2]float32{
		1.0*5.0 - 2.0*6.0 + 3.0*7.0 - 4.0*8.0, // Real part
		1.0*6.0 + 2.0*5.0 + 3.0*8.0 + 4.0*7.0, // Imaginary part
	}}

	result, err := Cdotu(x, y)
	if err != nil {
		t.Fatalf("Cdotu failed: %v", err)
	}

	if result.Data[0] != expected.Data[0] || result.Data[1] != expected.Data[1] {
		t.Errorf("Expected %v, but got %v", expected, result)
	}

	// Test with unequal vector sizes.
	xUnequal := newVectorComplexFloat(1, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
	})
	yUnequal := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})
	_, errUnequal := Cdotu(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Cdotu with unequal sizes should have returned an error")
	}
}

func TestCdotc(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})
	y := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})

	expected := GslComplexFloat{Data: [2]float32{
		1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0, // Real part
		1.0*6.0 - 2.0*5.0 + 3.0*8.0 - 4.0*7.0, // Imaginary part
	}}

	result, err := Cdotc(x, y)
	if err != nil {
		t.Fatalf("Cdotc failed: %v", err)
	}

	if result.Data[0] != expected.Data[0] || result.Data[1] != expected.Data[1] {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
	// Test with unequal vector sizes.
	xUnequal := newVectorComplexFloat(1, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}}})
	yUnequal := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})
	_, errUnequal := Cdotc(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Cdotc with unequal sizes should have returned an error")
	}
}

func TestZdotu(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})
	y := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})

	expected := GslComplex{Data: [2]float64{
		1.0*5.0 - 2.0*6.0 + 3.0*7.0 - 4.0*8.0, // Real part
		1.0*6.0 + 2.0*5.0 + 3.0*8.0 + 4.0*7.0, // Imaginary part
	}}

	result, err := Zdotu(x, y)
	if err != nil {
		t.Fatalf("Zdotu failed: %v", err)
	}

	if result.Data[0] != expected.Data[0] || result.Data[1] != expected.Data[1] {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
	// Test with unequal vector sizes.
	xUnequal := newVectorComplex(1, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
	})
	yUnequal := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})
	_, errUnequal := Zdotu(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Zdotu with unequal sizes should have returned an error")
	}
}

func TestZdotc(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})
	y := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})

	expected := GslComplex{Data: [2]float64{
		1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0, // Real part
		1.0*6.0 - 2.0*5.0 + 3.0*8.0 - 4.0*7.0, // Imaginary part
	}}

	result, err := Zdotc(x, y)
	if err != nil {
		t.Fatalf("Zdotc failed: %v", err)
	}

	if result.Data[0] != expected.Data[0] || result.Data[1] != expected.Data[1] {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
	// Test with unequal vector sizes.
	xUnequal := newVectorComplex(1, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
	})
	yUnequal := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})
	_, errUnequal := Zdotc(xUnequal, yUnequal)
	if errUnequal == nil {
		t.Errorf("Zdotc with unequal sizes should have returned an error")
	}
}

func TestSnrm2(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, -2.0, 3.0})
	expected := float32(math.Sqrt(float64(1.0*1.0 + 2.0*2.0 + 3.0*3.0)))
	result := Snrm2(x)
	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestDnrm2(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, -2.0, 3.0})
	expected := 3.7416573867739413
	result := Dnrm2(x)

	if math.Abs(result-expected) > 1e-15 {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestScnrm2(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{3.0, -4.0}},
		{Data: [2]float32{1.0, 2.0}},
	})
	expected := float32(math.Sqrt(float64(3.0*3.0 + 4.0*4.0 + 1.0*1.0 + 2.0*2.0)))
	result := Scnrm2(x)

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestDznrm2(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{3.0, -4.0}},
		{Data: [2]float64{1.0, 2.0}},
	})
	expected := math.Sqrt(float64(3.0*3.0 + 4.0*4.0 + 1.0*1.0 + 2.0*2.0))
	result := Dznrm2(x)

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestSasum(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, -2.0, 3.0})
	expected := float32(math.Abs(1.0) + math.Abs(-2.0) + math.Abs(3.0))
	result := Sasum(x)
	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestDasum(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, -2.0, 3.0})
	expected := math.Abs(1.0) + math.Abs(-2.0) + math.Abs(3.0)
	result := Dasum(x)
	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestScasum(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{3.0, -4.0}},
		{Data: [2]float32{1.0, 2.0}},
	})
	expected := float32(math.Abs(3.0) + math.Abs(-4.0) + math.Abs(1.0) + math.Abs(2.0))
	result := Scasum(x)

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestDzasum(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{3.0, -4.0}},
		{Data: [2]float64{1.0, 2.0}},
	})
	expected := math.Abs(3.0) + math.Abs(-4.0) + math.Abs(1.0) + math.Abs(2.0)
	result := Dzasum(x)

	if result != expected {
		t.Errorf("Expected %f, but got %f", expected, result)
	}
}

func TestIsamax(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, -5.0, 3.0})
	expected := 1
	result := Isamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}

	x = newVectorFloat(0, 1, []float32{}) // Empty vector
	expected = 0
	result = Isamax(x)
	if result != expected {
		t.Errorf("Expected index %d for empty vector, but got %d", expected, result)
	}
	x = newVectorFloat(5, 1, []float32{1.0, -5.0, 3.0, -5.0, 4.0})
	expected = 1
	result = Isamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
}

func TestIdamax(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, -5.0, 3.0})
	expected := 1
	result := Idamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}

	x = newVector(0, 1, []float64{}) // Empty vector
	expected = 0
	result = Idamax(x)
	if result != expected {
		t.Errorf("Expected index %d for empty vector, but got %d", expected, result)
	}
	x = newVector(5, 1, []float64{1.0, -5.0, 3.0, -5.0, 4.0})
	expected = 1
	result = Idamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
}

func TestIcamax(t *testing.T) {
	x := newVectorComplexFloat(3, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{-3.0, 4.0}}, // Magnitude: 5
		{Data: [2]float32{2.0, -1.0}},
	})

	expected := 1 // Index of {-3.0, 4.0} which has the largest magnitude (3+4=7).
	result := Icamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
	x = newVectorComplexFloat(5, 2, []GslComplexFloat{
		{Data: [2]float32{1.0, 1.0}},
		{Data: [2]float32{0.0, 0.0}},
		{Data: [2]float32{-3.0, 4.0}}, // Magnitude: 5
		{Data: [2]float32{0.0, 0.0}},
		{Data: [2]float32{2.0, -1.0}},
		{Data: [2]float32{0.0, 0.0}},
		{Data: [2]float32{0.0, 6.0}},
		{Data: [2]float32{0.0, 0.0}},
		{Data: [2]float32{1.0, 1.0}},
		{Data: [2]float32{0.0, 0.0}},
	})

	expected = 1 // Index of {-3.0, 4.0} which has the largest magnitude (3+4=7).
	result = Icamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
}

func TestIzamax(t *testing.T) {
	x := newVectorComplex(3, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{-3.0, 4.0}}, // Magnitude: 5
		{Data: [2]float64{2.0, -1.0}},
	})

	expected := 1 // Index of {-3.0, 4.0} which has the largest magnitude (3+4=7).
	result := Izamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
	x = newVectorComplex(5, 2, []GslComplex{
		{Data: [2]float64{1.0, 1.0}},
		{Data: [2]float64{0.0, 0.0}},
		{Data: [2]float64{-3.0, 4.0}}, // Magnitude: 5
		{Data: [2]float64{0.0, 0.0}},
		{Data: [2]float64{2.0, -1.0}},
		{Data: [2]float64{0.0, 0.0}},
		{Data: [2]float64{0.0, 6.0}},
		{Data: [2]float64{0.0, 0.0}},
		{Data: [2]float64{1.0, 1.0}},
		{Data: [2]float64{0.0, 0.0}},
	})

    expected = 1 // Index of {-3.0, 4.0} which has the largest magnitude (3+4=7).
	result = Izamax(x)
	if result != expected {
		t.Errorf("Expected index %d, but got %d", expected, result)
	}
}

func TestSswap(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	expectedX := []float32{4.0, 5.0, 6.0}
	expectedY := []float32{1.0, 2.0, 3.0}

	err := Sswap(x, y)
	if err != nil {
		t.Fatalf("Sswap failed: %v", err)
	}

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expectedX[i] {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expectedX[i], x.Data[i])
		}
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestDswap(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})
	expectedX := []float64{4.0, 5.0, 6.0}
	expectedY := []float64{1.0, 2.0, 3.0}

	err := Dswap(x, y)
	if err != nil {
		t.Fatalf("Dswap failed: %v", err)
	}

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expectedX[i] {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expectedX[i], x.Data[i])
		}
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestCswap(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})
	y := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})

	expectedX := []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	}
	expectedY := []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	}

	err := Cswap(x, y)
	if err != nil {
		t.Fatalf("Cswap failed: %v", err)
	}

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expectedX[i] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestZswap(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})
	y := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})

	expectedX := []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	}
	expectedY := []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	}

	err := Zswap(x, y)
	if err != nil {
		t.Fatalf("Zswap failed: %v", err)
	}

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expectedX[i] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestScopy(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})

	expectedY := []float32{1.0, 2.0, 3.0} // y should become a copy of x.

	err := Scopy(x, y)
	if err != nil {
		t.Fatalf("Scopy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestDcopy(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})

	expectedY := []float64{1.0, 2.0, 3.0} // y should become a copy of x.

	err := Dcopy(x, y)
	if err != nil {
		t.Fatalf("Dcopy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestCcopy(t *testing.T) {
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})
	y := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})

	expectedY := []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	}

	err := Ccopy(x, y)
	if err != nil {
		t.Fatalf("Ccopy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestZcopy(t *testing.T) {
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})
	y := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})

	expectedY := []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	}

	err := Zcopy(x, y)
	if err != nil {
		t.Fatalf("Zcopy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestSaxpy(t *testing.T) {
	alpha := float32(2.0)
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	expectedY := []float32{6.0, 9.0, 12.0} // y = 2*x + y

	err := Saxpy(alpha, x, y)
	if err != nil {
		t.Fatalf("Saxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
	x2 := newVectorFloat(3, 2, []float32{1.0, 0, 2.0, 0, 3.0, 0})
	y2 := newVectorFloat(3, 2, []float32{4.0, 0, 5.0, 0, 6.0, 0})
	expectedY = []float32{6.0, 0, 9.0, 0, 12.0, 0}
	err = Saxpy(alpha, x2, y2)
	if err != nil {
		t.Fatalf("Saxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y2.Data[i*uint(y2.Stride)] != expectedY[i*2] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i*2], y.Data[i*uint(y.Stride)])
		}
	}
}

func TestDaxpy(t *testing.T) {
	alpha := 2.0
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})
	expectedY := []float64{6.0, 9.0, 12.0} // y = 2*x + y

	err := Daxpy(alpha, x, y)
	if err != nil {
		t.Fatalf("Daxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
	x2 := newVector(3, 2, []float64{1.0, 0, 2.0, 0, 3.0, 0})
	y2 := newVector(3, 2, []float64{4.0, 0, 5.0, 0, 6.0, 0})
	expectedY = []float64{6.0, 0, 9.0, 0, 12.0, 0}
	err = Daxpy(alpha, x2, y2)
	if err != nil {
		t.Fatalf("Daxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y2.Data[i*uint(y2.Stride)] != expectedY[i*2] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i*2], y.Data[i*uint(y2.Stride)])
		}
	}
}

func TestCaxpy(t *testing.T) {
	alpha := GslComplexFloat{Data: [2]float32{2.0, 1.0}} // 2 + i
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})
	y := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{5.0, 6.0}},
		{Data: [2]float32{7.0, 8.0}},
	})

	// Expected result:
	// y[0] = (2+i)*(1+2i) + (5+6i) = (2 + 4i + i - 2) + (5 + 6i) = 0 + 5i + 5 + 6i = 5 + 11i
	// y[1] = (2+i)*(3+4i) + (7+8i) = (6 + 8i + 3i - 4) + (7 + 8i) = 2 + 11i + 7 + 8i = 9 + 19i
	expectedY := []GslComplexFloat{
		{Data: [2]float32{5.0, 11.0}},
		{Data: [2]float32{9.0, 19.0}},
	}

	err := Caxpy(alpha, x, y)
	if err != nil {
		t.Fatalf("Caxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i].Data[0] != expectedY[i].Data[0] || y.Data[i].Data[1] != expectedY[i].Data[1] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestZaxpy(t *testing.T) {
	alpha := GslComplex{Data: [2]float64{2.0, 1.0}} // 2 + i
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})
	y := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{5.0, 6.0}},
		{Data: [2]float64{7.0, 8.0}},
	})

	// Expected result:
	// y[0] = (2+i)*(1+2i) + (5+6i) = (2 + 4i + i - 2) + (5 + 6i) = 0 + 5i + 5 + 6i = 5 + 11i
	// y[1] = (2+i)*(3+4i) + (7+8i) = (6 + 8i + 3i - 4) + (7 + 8i) = 2 + 11i + 7 + 8i = 9 + 19i
	expectedY := []GslComplex{
		{Data: [2]float64{5.0, 11.0}},
		{Data: [2]float64{9.0, 19.0}},
	}

	err := Zaxpy(alpha, x, y)
	if err != nil {
		t.Fatalf("Zaxpy failed: %v", err)
	}

	for i := uint(0); i < y.Size; i++ {
		if y.Data[i].Data[0] != expectedY[i].Data[0] || y.Data[i].Data[1] != expectedY[i].Data[1] {
			t.Errorf("Expected y[%d] to be %v, but got %v", i, expectedY[i], y.Data[i])
		}
	}
}

func TestSrotg(t *testing.T) {
	a := float32(3.0)
	b := float32(4.0)
	c, s, _, _ := Srotg(a, b)
	expC := float32(a / 5.0)
	expS := float32(b / 5.0)
	if math.Abs(float64(c-expC)) > 1e-6 || math.Abs(float64(s-expS)) > 1e-6 {
		t.Errorf("Expected c = %f, s= %f, but got, c= %f, s = %f", expC, expS, c, s)
	}
}

func TestDrotg(t *testing.T) {
	a := 3.0
	b := 4.0
	c, s, _, _ := Drotg(a, b)
	expC := a / 5.0
	expS := b / 5.0

	if math.Abs(c-expC) > 1e-15 || math.Abs(s-expS) > 1e-15 {
		t.Errorf("Expected c = %f, s= %f, but got, c= %f, s = %f", expC, expS, c, s)
	}
}

func TestSrot(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})
	c := float32(0.6) // cos(theta)
	s := float32(0.8) // sin(theta)

	// Manually compute the expected result of the rotation
	expectedX := make([]float32, 3)
	expectedY := make([]float32, 3)
	for i := 0; i < 3; i++ {
		expectedX[i] = c*x.Data[i] + s*y.Data[i]
		expectedY[i] = c*y.Data[i] - s*x.Data[i]
	}
	err := Srot(x, y, c, s)
	if err != nil {
		t.Fatalf("Srot failed %v", err)
	}
	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expectedX[i] {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expectedX[i], x.Data[i])
		}
		if y.Data[i] != expectedY[i] {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestDrot(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})
	c := 0.6 // cos(theta)
	s := 0.8 // sin(theta)

	// Manually compute the expected result of the rotation
	expectedX := make([]float64, 3)
	expectedY := make([]float64, 3)
	for i := 0; i < 3; i++ {
		expectedX[i] = c*x.Data[i] + s*y.Data[i]
		expectedY[i] = c*y.Data[i] - s*x.Data[i]
	}

	err := Drot(x, y, c, s)
	if err != nil {
		t.Fatalf("Drot failed: %v", err)
	}

	for i := uint(0); i < x.Size; i++ {
		if math.Abs(x.Data[i]-expectedX[i]) > 1e-15 {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expectedX[i], x.Data[i])
		}
		if math.Abs(y.Data[i]-expectedY[i]) > 1e-15 {
			t.Errorf("Expected y[%d] to be %f, but got %f", i, expectedY[i], y.Data[i])
		}
	}
}

func TestSrotmg(t *testing.T) {
	d1 := float32(1.0)
	d2 := float32(1.0)
	b1 := float32(1.0)
	b2 := float32(1.0)

	p := Srotmg(&d1, &d2, &b1, &b2)
	// Check for correctness of d1 and d2
	if !math.IsNaN(float64(d1)) {
		if d1 <= 0.0 {
			if p[0] != -1.0 {
				t.Errorf("Expected flag to be -1, but got %f", p[0])
			}
		}
	}

}

func TestDrotmg(t *testing.T) {
	d1 := 1.0
	d2 := 1.0
	b1 := 1.0
	b2 := 1.0

	p := Drotmg(&d1, &d2, &b1, &b2)
	if !math.IsNaN(d1) {
		if d1 <= 0.0 {
			if p[0] != -1.0 {
				t.Errorf("Expected flag to be -1, but got %f", p[0])
			}
		}
	}
}

func TestSrotm(t *testing.T) {
	x := newVectorFloat(3, 1, []float32{1.0, 2.0, 3.0})
	y := newVectorFloat(3, 1, []float32{4.0, 5.0, 6.0})

	// Test case 1: Flag = -1.0 (General modified Givens rotation)
	p1 := [5]float32{-1.0, 0.6, -0.8, 0.8, 0.6} // Example values
	xCopy1 := make([]float32, len(x.Data))
	copy(xCopy1, x.Data)
	yCopy1 := make([]float32, len(y.Data))
	copy(yCopy1, y.Data)
	Srotm(x, y, p1)

	for i := uint(0); i < x.Size; i++ {
		expectedX := p1[1]*xCopy1[i] + p1[3]*yCopy1[i]
		expectedY := p1[2]*xCopy1[i] + p1[4]*yCopy1[i]
		if x.Data[i] != expectedX {
			t.Errorf("Case 1: Expected x[%d] to be %f, but got %f", i, expectedX, x.Data[i])
		}
		if y.Data[i] != expectedY {
			t.Errorf("Case 1: Expected y[%d] to be %f, but got %f", i, expectedY, y.Data[i])
		}
	}
	// Test case 3: Flag = 1.0 (Scaled modified Givens rotation, H21 = -1, H12 = 1)
	x.Data = []float32{1.0, 2.0, 3.0}
	y.Data = []float32{4.0, 5.0, 6.0}
	xCopy1 = make([]float32, len(x.Data))
	copy(xCopy1, x.Data)
	yCopy1 = make([]float32, len(y.Data))
	copy(yCopy1, y.Data)
	p3 := [5]float32{1.0, 0.5, -1.0, 1.0, 0.7}
	Srotm(x, y, p3)
	for i := uint(0); i < x.Size; i++ {
		expectedX := p3[1]*xCopy1[i] + p3[3]*yCopy1[i]
		expectedY := p3[2]*xCopy1[i] + p3[4]*yCopy1[i]
		if x.Data[i] != expectedX {
			t.Errorf("Case 3: Expected x[%d] to be %f, but got %f", i, expectedX, x.Data[i])
		}
		if y.Data[i] != expectedY {
			t.Errorf("Case 3: Expected y[%d] to be %f, but got %f", i, expectedY, y.Data[i])
		}
	}
}

func TestDrotm(t *testing.T) {
	x := newVector(3, 1, []float64{1.0, 2.0, 3.0})
	y := newVector(3, 1, []float64{4.0, 5.0, 6.0})

	// Test case 1: Flag = -1.0 (General modified Givens rotation)
	p1 := [5]float64{-1.0, 0.6, -0.8, 0.8, 0.6} // Example values
	xCopy1 := make([]float64, len(x.Data))
	copy(xCopy1, x.Data)
	yCopy1 := make([]float64, len(y.Data))
	copy(yCopy1, y.Data)
	Drotm(x, y, p1)

	for i := uint(0); i < x.Size; i++ {
		expectedX := p1[1]*xCopy1[i] + p1[3]*yCopy1[i]
		expectedY := p1[2]*xCopy1[i] + p1[4]*yCopy1[i]
		if math.Abs(x.Data[i]-expectedX) > 1e-15 {
			t.Errorf("Case 1: Expected x[%d] to be %f, but got %f", i, expectedX, x.Data[i])
		}
		if math.Abs(y.Data[i]-expectedY) > 1e-15 {
			t.Errorf("Case 1: Expected y[%d] to be %f, but got %f", i, expectedY, y.Data[i])
		}
	}
	// Test case 3: Flag = 1.0 (Scaled modified Givens rotation, H21 = -1, H12 = 1)
	x.Data = []float64{1.0, 2.0, 3.0}
	y.Data = []float64{4.0, 5.0, 6.0}
	xCopy1 = make([]float64, len(x.Data))
	copy(xCopy1, x.Data)
	yCopy1 = make([]float64, len(y.Data))
	copy(yCopy1, y.Data)
	p3 := [5]float64{1.0, 0.5, -1.0, 1.0, 0.7}
	Drotm(x, y, p3)
	for i := uint(0); i < x.Size; i++ {
		expectedX := p3[1]*xCopy1[i] + p3[3]*yCopy1[i]
		expectedY := p3[2]*xCopy1[i] + p3[4]*yCopy1[i]
		if math.Abs(x.Data[i]-expectedX) > 1e-15 {
			t.Errorf("Case 3: Expected x[%d] to be %f, but got %f", i, expectedX, x.Data[i])
		}
		if math.Abs(y.Data[i]-expectedY) > 1e-15 {
			t.Errorf("Case 3: Expected y[%d] to be %f, but got %f", i, expectedY, y.Data[i])
		}
	}
}

func TestSscal(t *testing.T) {
	alpha := float32(2.5)
	x := newVectorFloat(3, 1, []float32{1.0, -2.0, 3.0})
	expected := []float32{2.5, -5.0, 7.5}
	Sscal(alpha, x)
	for i := uint(0); i < x.Size; i++ {
		if x.Data[i] != expected[i] {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expected[i], x.Data[i])
		}
	}
	alpha = float32(3.0)
	x = newVectorFloat(3, 2, []float32{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	expected = []float32{3.0, 0.0, 6.0, 0.0, 9.0, 0.0}
	Sscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i*uint(x.Stride)] != expected[i*2] {
			t.Errorf("Expected x[%d] to be %f, but got %f", i*2, expected[i*2], x.Data[i*uint(x.Stride)])
		}
	}

}

func TestDscal(t *testing.T) {
	alpha := 2.5
	x := newVector(3, 1, []float64{1.0, -2.0, 3.0})
	expected := []float64{2.5, -5.0, 7.5}
	Dscal(alpha, x)
	for i := uint(0); i < x.Size; i++ {
		if math.Abs(x.Data[i]-expected[i]) > 1e-14 {
			t.Errorf("Expected x[%d] to be %f, but got %f", i, expected[i], x.Data[i])
		}
	}
	alpha = 3.0
	x = newVector(3, 2, []float64{1.0, 0.0, 2.0, 0.0, 3.0, 0.0})
	expected = []float64{3.0, 0.0, 6.0, 0.0, 9.0, 0.0}
	Dscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if math.Abs(x.Data[i*uint(x.Stride)]-expected[i*2]) > 1e-14 {
			t.Errorf("Expected x[%d] to be %f, but got %f", i*2, expected[i*2], x.Data[i*uint(x.Stride)])
		}
	}
}

func TestCscal(t *testing.T) {
	alpha := GslComplexFloat{Data: [2]float32{2.0, 1.0}} // 2 + i
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})

	// Expected result:
	// x[0] = (2+i)*(1+2i) = (2 + 4i + i - 2) = 0 + 5i
	// x[1] = (2+i)*(3+4i) = (6 + 8i + 3i - 4) = 2 + 11i
	expectedX := []GslComplexFloat{
		{Data: [2]float32{0.0, 5.0}},
		{Data: [2]float32{2.0, 11.0}},
	}

	Cscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i].Data[0] != expectedX[i].Data[0] || x.Data[i].Data[1] != expectedX[i].Data[1] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
	}
}

func TestZscal(t *testing.T) {
	alpha := GslComplex{Data: [2]float64{2.0, 1.0}} // 2 + i
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})

	// Expected result:
	// x[0] = (2+i)*(1+2i) = (2 + 4i + i - 2) = 0 + 5i
	// x[1] = (2+i)*(3+4i) = (6 + 8i + 3i - 4) = 2 + 11i
	expectedX := []GslComplex{
		{Data: [2]float64{0.0, 5.0}},
		{Data: [2]float64{2.0, 11.0}},
	}

	Zscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i].Data[0] != expectedX[i].Data[0] || x.Data[i].Data[1] != expectedX[i].Data[1] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
	}
}

func TestCsscal(t *testing.T) {
	alpha := float32(2.5)
	x := newVectorComplexFloat(2, 1, []GslComplexFloat{
		{Data: [2]float32{1.0, 2.0}},
		{Data: [2]float32{3.0, 4.0}},
	})

	// Expected result:
	// x[0] =  2.5 * (1 + 2i) = 2.5 + 5i
	// x[1] = 2.5 * (3 + 4i) = 7.5 + 10i
	expectedX := []GslComplexFloat{
		{Data: [2]float32{2.5, 5.0}},
		{Data: [2]float32{7.5, 10.0}},
	}

	Csscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i].Data[0] != expectedX[i].Data[0] || x.Data[i].Data[1] != expectedX[i].Data[1] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
	}
}

func TestZdscal(t *testing.T) {
	alpha := 2.5
	x := newVectorComplex(2, 1, []GslComplex{
		{Data: [2]float64{1.0, 2.0}},
		{Data: [2]float64{3.0, 4.0}},
	})

	// Expected result:
	// x[0] =  2.5 * (1 + 2i) = 2.5 + 5i
	// x[1] = 2.5 * (3 + 4i) = 7.5 + 10i
	expectedX := []GslComplex{
		{Data: [2]float64{2.5, 5.0}},
		{Data: [2]float64{7.5, 10.0}},
	}

	Zdscal(alpha, x)

	for i := uint(0); i < x.Size; i++ {
		if x.Data[i].Data[0] != expectedX[i].Data[0] || x.Data[i].Data[1] != expectedX[i].Data[1] {
			t.Errorf("Expected x[%d] to be %v, but got %v", i, expectedX[i], x.Data[i])
		}
	}
}
