// Package blas provides Go implementations of BLAS (Basic Linear Algebra Subprograms).
//
// The code is organized into Level 1, Level 2, and Level 3 BLAS routines,
// following the standard BLAS categorization.
package blas

import (
	"fmt"
	"math"
)

// CblasTranspose represents different matrix transposition options.
type CblasTranspose int

// CblasUplo represents upper or lower triangular matrix storage.
type CblasUplo int

// CblasDiag represents if a triangular matrix is unit diagonal or not.
type CblasDiag int

// CblasSide represents whether a matrix is on the left or right side in an operation.
type CblasSide int

// Constants for transposition, upper/lower, diagonal, and side.
const (
	CblasNoTrans   CblasTranspose = 111
	CblasTrans     CblasTranspose = 112
	CblasConjTrans CblasTranspose = 113

	CblasUpper CblasUplo = 121
	CblasLower CblasUplo = 122

	CblasNonUnit CblasDiag = 131
	CblasUnit    CblasDiag = 132

	CblasLeft  CblasSide = 141
	CblasRight CblasSide = 142
)

// GslComplex represents a complex number with real and imaginary parts.
type GslComplex struct {
	Data [2]float64
}

// GslComplexFloat represents a complex number with float32 real and imaginary parts.
type GslComplexFloat struct {
	Data [2]float32
}

// Vector represents a vector with its data, size, and stride.
type Vector struct {
	Size   uint
	Stride int
	Data   []float64
}

// VectorFloat represents a float32 vector.
type VectorFloat struct {
	Size   uint
	Stride int
	Data   []float32
}

// VectorComplex represents a complex vector with GslComplex elements.
type VectorComplex struct {
	Size   uint
	Stride int
	Data   []GslComplex
}

// VectorComplexFloat represents a complex vector with GslComplexFloat elements.
type VectorComplexFloat struct {
	Size   uint
	Stride int
	Data   []GslComplexFloat
}

// Matrix represents a matrix with its data, dimensions, and leading dimension.
type Matrix struct {
	Size1 uint
	Size2 uint
	Tda   int
	Data  []float64
}

// MatrixFloat represents a float32 matrix.
type MatrixFloat struct {
	Size1 uint
	Size2 uint
	Tda   int
	Data  []float32
}

// MatrixComplex represents a complex matrix with GslComplex elements.
type MatrixComplex struct {
	Size1 uint
	Size2 uint
	Tda   int
	Data  []GslComplex
}

// MatrixComplexFloat represents a complex matrix with GslComplexFloat elements.
type MatrixComplexFloat struct {
	Size1 uint
	Size2 uint
	Tda   int
	Data  []GslComplexFloat
}

// Level 1

// Sdsdot computes the dot product of two float32 vectors with single precision accumulation, plus a scalar.
func Sdsdot(alpha float32, x *VectorFloat, y *VectorFloat) (float32, error) {
	if x.Size != y.Size {
		// FIXME: this is not needed, just set result to alpha
		// and return
		return 0, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}

	var result float32
	for i := uint(0); i < x.Size; i++ {
		result += x.Data[i*uint(x.Stride)] * y.Data[i*uint(y.Stride)]
	}
	result += alpha

	return result, nil
}

// Dsdot computes the dot product of two float32 vectors with double precision accumulation.
func Dsdot(x *VectorFloat, y *VectorFloat) (float64, error) {
	if x.Size != y.Size {
		return 0, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}

	var result float64
	for i := uint(0); i < x.Size; i++ {
		result += float64(x.Data[i*uint(x.Stride)]) * float64(y.Data[i*uint(y.Stride)])
	}
	return result, nil
}

// Sdot computes the dot product of two float32 vectors.
func Sdot(x *VectorFloat, y *VectorFloat) (float32, error) {
	if x.Size != y.Size {
		return 0, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var result float32
	for i := uint(0); i < x.Size; i++ {
		result += x.Data[i*uint(x.Stride)] * y.Data[i*uint(y.Stride)]
	}
	return result, nil
}

// Ddot computes the dot product of two vectors.
func Ddot(x *Vector, y *Vector) (float64, error) {
	if x.Size != y.Size {
		return 0, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var result float64
	for i := uint(0); i < x.Size; i++ {
		result += x.Data[i*uint(x.Stride)] * y.Data[i*uint(y.Stride)]
	}
	return result, nil
}

// Cdotu computes the dot product of two complex vectors (unconjugated).
func Cdotu(x *VectorComplexFloat, y *VectorComplexFloat) (GslComplexFloat, error) {
	if x.Size != y.Size {
		return GslComplexFloat{}, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var dotu GslComplexFloat
	for i := uint(0); i < x.Size; i++ {
		dotu.Data[0] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[0] - x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[1]
		dotu.Data[1] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[1] + x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[0]
	}
	return dotu, nil
}

// Cdotc computes the dot product of two complex vectors (conjugated).
func Cdotc(x *VectorComplexFloat, y *VectorComplexFloat) (GslComplexFloat, error) {
	if x.Size != y.Size {
		return GslComplexFloat{}, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var dotc GslComplexFloat
	for i := uint(0); i < x.Size; i++ {
		dotc.Data[0] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[1]
		dotc.Data[1] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[1] - x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[0]
	}
	return dotc, nil
}

// Zdotu computes the dot product of two complex vectors (unconjugated).
func Zdotu(x *VectorComplex, y *VectorComplex) (GslComplex, error) {
	if x.Size != y.Size {
		return GslComplex{}, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var dotu GslComplex
	for i := uint(0); i < x.Size; i++ {
		dotu.Data[0] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[0] - x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[1]
		dotu.Data[1] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[1] + x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[0]
	}
	return dotu, nil
}

// Zdotc computes the dot product of two complex vectors (conjugated).
func Zdotc(x *VectorComplex, y *VectorComplex) (GslComplex, error) {
	if x.Size != y.Size {
		return GslComplex{}, fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	var dotc GslComplex
	for i := uint(0); i < x.Size; i++ {
		dotc.Data[0] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[1]
		dotc.Data[1] += x.Data[i*uint(x.Stride)].Data[0]*y.Data[i*uint(y.Stride)].Data[1] - x.Data[i*uint(x.Stride)].Data[1]*y.Data[i*uint(y.Stride)].Data[0]
	}
	return dotc, nil
}

// Snrm2 computes the Euclidean norm of a float32 vector.
func Snrm2(x *VectorFloat) float32 {
	var nrm2 float32
	for i := uint(0); i < x.Size; i++ {
		nrm2 += x.Data[i*uint(x.Stride)] * x.Data[i*uint(x.Stride)]
	}
	return float32(math.Sqrt(float64(nrm2)))
}

// Sasum computes the sum of absolute values of a float32 vector.
func Sasum(x *VectorFloat) float32 {
	var asum float32
	for i := uint(0); i < x.Size; i++ {
		asum += float32(math.Abs(float64(x.Data[i*uint(x.Stride)])))
	}
	return asum
}

// Dnrm2 computes the Euclidean norm of a vector.
func Dnrm2(x *Vector) float64 {
	var nrm2 float64
	for i := uint(0); i < x.Size; i++ {
		nrm2 += x.Data[i*uint(x.Stride)] * x.Data[i*uint(x.Stride)]
	}
	return math.Sqrt(nrm2)
}

// Dasum computes the sum of absolute values of a vector.
func Dasum(x *Vector) float64 {
	var asum float64
	for i := uint(0); i < x.Size; i++ {
		asum += math.Abs(x.Data[i*uint(x.Stride)])
	}
	return asum
}

// Scnrm2 computes the Euclidean norm of a complex vector.
func Scnrm2(x *VectorComplexFloat) float32 {
	var nrm2 float32
	for i := uint(0); i < x.Size; i++ {
		nrm2 += x.Data[i*uint(x.Stride)].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
	}
	return float32(math.Sqrt(float64(nrm2)))
}

// Scasum computes the sum of absolute values of a complex vector.
func Scasum(x *VectorComplexFloat) float32 {
	var asum float32
	for i := uint(0); i < x.Size; i++ {
		asum += float32(math.Abs(float64(x.Data[i*uint(x.Stride)].Data[0]))) + float32(math.Abs(float64(x.Data[i*uint(x.Stride)].Data[1])))
	}
	return asum
}

// Dznrm2 computes the Euclidean norm of a complex vector.
func Dznrm2(x *VectorComplex) float64 {
	var nrm2 float64
	for i := uint(0); i < x.Size; i++ {
		nrm2 += x.Data[i*uint(x.Stride)].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
	}
	return math.Sqrt(nrm2)
}

// Dzasum computes the sum of absolute values of a complex vector.
func Dzasum(x *VectorComplex) float64 {
	var asum float64
	for i := uint(0); i < x.Size; i++ {
		asum += math.Abs(x.Data[i*uint(x.Stride)].Data[0]) + math.Abs(x.Data[i*uint(x.Stride)].Data[1])
	}
	return asum
}

// Isamax finds the index of the element with the largest absolute value in a float32 vector.
func Isamax(x *VectorFloat) int {
	if x.Size == 0 {
		return 0 // Return 0 for an empty vector, consistent with BLAS behavior
	}
	maxIndex := 0
	maxValue := float32(math.Abs(float64(x.Data[0])))
	for i := uint(1); i < x.Size; i++ {
		absValue := float32(math.Abs(float64(x.Data[i*uint(x.Stride)])))
		if absValue > maxValue {
			maxValue = absValue
			maxIndex = int(i)
		}
	}
	return maxIndex
}

// Idamax finds the index of the element with the largest absolute value in a vector.
func Idamax(x *Vector) int {
	if x.Size == 0 {
		return 0 // Return 0 for an empty vector, consistent with BLAS behavior.
	}

	maxIndex := 0
	maxValue := math.Abs(x.Data[0])
	for i := uint(1); i < x.Size; i++ {
		absValue := math.Abs(x.Data[i*uint(x.Stride)])
		if absValue > maxValue {
			maxValue = absValue
			maxIndex = int(i)
		}
	}
	return maxIndex
}

// Icamax finds the index of the element with the largest absolute value in a complex vector.
func Icamax(x *VectorComplexFloat) int {
	if x.Size == 0 {
		return 0 // Return 0 for an empty vector, consistent with BLAS behavior
	}
	maxIndex := 0
	maxValue := float32(math.Abs(float64(x.Data[0].Data[0]))) + float32(math.Abs(float64(x.Data[0].Data[1])))
	for i := uint(1); i < x.Size; i++ {
		absValue := float32(math.Abs(float64(x.Data[i*uint(x.Stride)].Data[0]))) + float32(math.Abs(float64(x.Data[i*uint(x.Stride)].Data[1])))
		if absValue > maxValue {
			maxValue = absValue
			maxIndex = int(i)
		}
	}
	return maxIndex
}

// Izamax finds the index of the element with the largest absolute value in a complex vector.
func Izamax(x *VectorComplex) int {
	if x.Size == 0 {
		return 0 // Return 0 for an empty vector, consistent with BLAS behavior.
	}
	maxIndex := 0
	maxValue := math.Abs(x.Data[0].Data[0]) + math.Abs(x.Data[0].Data[1])
	for i := uint(1); i < x.Size; i++ {
		absValue := math.Abs(x.Data[i*uint(x.Stride)].Data[0]) + math.Abs(x.Data[i*uint(x.Stride)].Data[1])
		if absValue > maxValue {
			maxValue = absValue
			maxIndex = int(i)
		}
	}
	return maxIndex
}

// Sswap swaps two float32 vectors.
func Sswap(x, y *VectorFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}

	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)], y.Data[i*uint(y.Stride)] = y.Data[i*uint(y.Stride)], x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Scopy copies a float32 vector to another float32 vector.
func Scopy(x, y *VectorFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] = x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Saxpy computes y = alpha * x + y for float32 vectors.
func Saxpy(alpha float32, x *VectorFloat, y *VectorFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] += alpha * x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Dswap swaps two vectors.
func Dswap(x, y *Vector) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)], y.Data[i*uint(y.Stride)] = y.Data[i*uint(y.Stride)], x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Dcopy copies a vector to another vector.
func Dcopy(x, y *Vector) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] = x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Daxpy computes y = alpha * x + y for vectors.
func Daxpy(alpha float64, x *Vector, y *Vector) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] += alpha * x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Cswap swaps two complex vectors.
func Cswap(x, y *VectorComplexFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)], y.Data[i*uint(y.Stride)] = y.Data[i*uint(y.Stride)], x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Ccopy copies a complex vector to another complex vector.
func Ccopy(x, y *VectorComplexFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] = x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Caxpy computes y = alpha * x + y for complex vectors.
func Caxpy(alpha GslComplexFloat, x, y *VectorComplexFloat) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1]
		y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0]
	}
	return nil
}

// Zswap swaps two complex vectors.
func Zswap(x, y *VectorComplex) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)], y.Data[i*uint(y.Stride)] = y.Data[i*uint(y.Stride)], x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Zcopy copies a complex vector to another complex vector.
func Zcopy(x, y *VectorComplex) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)] = x.Data[i*uint(x.Stride)]
	}
	return nil
}

// Zaxpy computes y = alpha * x + y for complex vectors.
func Zaxpy(alpha GslComplex, x, y *VectorComplex) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1]
		y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0]
	}
	return nil
}

// Srotg generates a float32 Givens rotation.
func Srotg(a, b float32) (c, s, r, z float32) {
	roe := b
	if math.Abs(float64(a)) > math.Abs(float64(b)) {
		roe = a
	}
	scale := float32(math.Abs(float64(a)) + math.Abs(float64(b)))
	if scale == 0.0 {
		c = 1.0
		s = 0.0
		r = 0.0
		z = 0.0
	} else {
		r = scale * float32(math.Sqrt(float64(a/scale*a/scale+b/scale*b/scale)))
		if roe < 0 {
			r = -r
		}
		c = a / r
		s = b / r
		if math.Abs(float64(c)) <= math.SmallestNonzeroFloat32 { // c near zero
			z = 1
		} else {
			if c != 0 {
				z = s
			} else {
				z = 1 / c
			}

		}

	}
	return c, s, r, z
}

// Srot applies a float32 Givens rotation to two float32 vectors.
func Srot(x, y *VectorFloat, c, s float32) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		xi := x.Data[i*uint(x.Stride)]
		yi := y.Data[i*uint(y.Stride)]
		x.Data[i*uint(x.Stride)] = c*xi + s*yi
		y.Data[i*uint(y.Stride)] = c*yi - s*xi
	}
	return nil
}

// Srotmg generates a modified float32 Givens rotation.
func Srotmg(d1, d2, b1, b2 *float32) (p [5]float32) {
	// Algorithm and notations from:
	// "A Generalized Vector Norm and an Algorithm for
	// Computing the Modified Givens Transformation"
	// Oz, O. and B. Philippe, INRIA research report, 1987

	// Original reference Fortran code:
	// http://www.netlib.org/blas/srotmg.f

	gam := float32(4096.0)
	gamsq := float32(16777216.0)
	rgamsq := float32(5.9604645e-8)

	if *d1 < 0.0 {
		// Flag the case d1 < 0, set all outputs to 0
		p[0] = -1.0
		*d1 = 0.0
		*d2 = 0.0
		*b1 = 0.0
		return
	}

	// Initialize
	h11 := float32(1.0)
	h12 := float32(1.0)
	h21 := float32(-1.0)
	h22 := float32(1.0)

	u := *b1

	if *d2 == 0.0 {
		// d2 is zero, return with d1 unchanged
		p[0] = -2.0 // Flag to indicate this case
		return
	}
	if *b1 == 0.0 {
		return
	}

	// Handle the general case
	temp := float32(1.0) + *b1**b2
	if temp == 0.0 { // Check for case where b1 * b2 = -1
		*d1 = *d1 + *d2*(*b1**b2) // Restore original d1
		*d2 = 0.0                 // Set d2 to 0
		p[0] = -2.0               // Return the correct flag
		return                    // Return early
	}

	// Original algorithm with adjustments to Go
	p2 := *d2 * *b2
	if p2 == 0.0 {
		p[0] = -1.0 // Flag to indicate this specific case
		return
	}

	p1 := *d1 * u
	q2 := p2 * u
	q1 := p1 * *b1

	if math.Abs(float64(q1)) > math.Abs(float64(q2)) {
		h21 = -(*b2) / *b1
		h12 = p2 / p1
		u = 1.0 + h12*h21
		if u > 0.0 {
			*d1 /= u
			*d2 /= u
			*b1 *= u
		}
	} else {
		if q2 < 0.0 {
			p[0] = -1.0
			return
		}
		h11 = p1 / p2
		h22 = *b1 / *b2
		u = 1.0 + h11*h22
		temp = *d1
		*d1 = *d2
		*d2 = temp
		*b1 = 1.0 / u
		*d1 *= *b1
		*d2 *= u
	}

	// Rescale if necessary
	for *d1 <= rgamsq || *d1 >= gamsq {
		if *d1 == 0.0 {
			h11 = 0.0
			h12 = 1.0
			h21 = -1.0
			h22 = 0.0
			break
		}
		if *d1 <= rgamsq {
			*d1 *= gam * gam
			*b1 /= gam
			h11 /= gam
			h12 /= gam
		} else {
			*d1 /= gam * gam
			*b1 *= gam
			h11 *= gam
			h12 *= gam
		}
	}

	for *d2 <= rgamsq || *d2 >= gamsq {
		if *d2 == 0.0 {
			h11 = 1.0
			h12 = 0.0
			h21 = 0.0
			h22 = 1.0
			break
		}
		if *d2 <= rgamsq {
			*d2 *= gam * gam
			h21 /= gam
			h22 /= gam
		} else {
			*d2 /= gam * gam
			h21 *= gam
			h22 *= gam
		}
	}

	if math.Abs(float64(u)) <= 1.0 {
		p[0] = h11
		p[1] = h21
		p[2] = h12
		p[3] = h22
	} else {
		if u >= 0.0 {
			p[0] = 1.0
		} else {
			p[0] = -1.0
		}
		p[1] = h21 / u
		p[2] = h12 / u
		p[3] = h22 / u
		*b1 = u
	}
	return
}

// Srotm applies a modified Givens rotation to two float32 vectors.
func Srotm(x, y *VectorFloat, p [5]float32) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}

	flag := p[0]
	var h11, h21, h12, h22 float32

	if flag == -2.0 { // Identity transformation, no-op
		return nil
	}

	if flag == -1.0 {
		h11 = p[1]
		h21 = p[2]
		h12 = p[3]
		h22 = p[4]
	} else if flag == 0.0 {
		h21 = p[2]
		h12 = p[3]
		h11 = 1.0
		h22 = 1.0
	} else if flag == 1.0 {
		h11 = p[1]
		h22 = p[4]
		h21 = -1.0
		h12 = 1.0
	} else {
		// Invalid flag value
		return fmt.Errorf("invalid flag value in Srotm: %f", flag)
	}

	for i := uint(0); i < x.Size; i++ {
		xi := x.Data[i*uint(x.Stride)]
		yi := y.Data[i*uint(y.Stride)]
		x.Data[i*uint(x.Stride)] = h11*xi + h12*yi
		y.Data[i*uint(y.Stride)] = h21*xi + h22*yi
	}

	return nil
}

// Drotg generates a Givens rotation.
func Drotg(a, b float64) (c, s, r, z float64) {
	roe := b
	if math.Abs(a) > math.Abs(b) {
		roe = a
	}
	scale := math.Abs(a) + math.Abs(b)
	if scale == 0.0 {
		c = 1.0
		s = 0.0
		r = 0.0
		z = 0.0
	} else {
		r = scale * math.Sqrt(a/scale*a/scale+b/scale*b/scale)
		if roe < 0 {
			r = -r
		}
		c = a / r
		s = b / r
		if math.Abs(c) <= math.SmallestNonzeroFloat64 {
			z = 1
		} else {
			if c != 0 {
				z = s
			} else {
				z = 1 / c
			}
		}

	}
	return c, s, r, z
}

// Drot applies a Givens rotation to two vectors.
func Drot(x, y *Vector, c, s float64) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}
	for i := uint(0); i < x.Size; i++ {
		xi := x.Data[i*uint(x.Stride)]
		yi := y.Data[i*uint(y.Stride)]
		x.Data[i*uint(x.Stride)] = c*xi + s*yi
		y.Data[i*uint(y.Stride)] = c*yi - s*xi
	}
	return nil
}

// Drotmg generates a modified Givens rotation.
func Drotmg(d1, d2, b1, b2 *float64) (p [5]float64) {
	// Algorithm and notations from:
	// "A Generalized Vector Norm and an Algorithm for
	// Computing the Modified Givens Transformation"
	// Oz, O. and B. Philippe, INRIA research report, 1987

	// Original reference Fortran code:
	// http://www.netlib.org/blas/drotmg.f

	gam := 4096.0
	gamsq := 16777216.0
	rgamsq := 5.9604645e-8

	if *d1 < 0.0 {
		// Flag the case d1 < 0, set all outputs to 0
		p[0] = -1.0
		*d1 = 0.0
		*d2 = 0.0
		*b1 = 0.0
		return
	}

	// Initialize
	h11 := 1.0
	h12 := 1.0
	h21 := -1.0
	h22 := 1.0

	u := *b1

	if *d2 == 0.0 {
		// d2 is zero, return with d1 unchanged
		p[0] = -2.0 // Flag to indicate this case
		return
	}
	if *b1 == 0.0 {
		return
	}

	// Handle the general case
	temp := 1.0 + *b1**b2
	if temp == 0.0 { // Check for case where b1 * b2 = -1
		*d1 = *d1 + *d2*(*b1**b2) // Restore original d1
		*d2 = 0.0                 // Set d2 to 0
		p[0] = -2.0               // Return the correct flag
		return                    // Return early
	}

	// Original algorithm with adjustments to Go
	p2 := *d2 * *b2
	if p2 == 0.0 {
		p[0] = -1.0 // Flag to indicate this specific case
		return
	}

	p1 := *d1 * u
	q2 := p2 * u
	q1 := p1 * *b1

	if math.Abs(q1) > math.Abs(q2) {
		h21 = -(*b2) / *b1
		h12 = p2 / p1
		u = 1.0 + h12*h21
		if u > 0.0 {
			*d1 /= u
			*d2 /= u
			*b1 *= u
		}
	} else {
		if q2 < 0.0 {
			p[0] = -1.0
			return
		}
		h11 = p1 / p2
		h22 = *b1 / *b2
		u = 1.0 + h11*h22
		temp = *d1
		*d1 = *d2
		*d2 = temp
		*b1 = 1.0 / u
		*d1 *= *b1
		*d2 *= u
	}

	// Rescale if necessary
	for *d1 <= rgamsq || *d1 >= gamsq {
		if *d1 == 0.0 {
			h11 = 0.0
			h12 = 1.0
			h21 = -1.0
			h22 = 0.0
			break
		}
		if *d1 <= rgamsq {
			*d1 *= gam * gam
			*b1 /= gam
			h11 /= gam
			h12 /= gam
		} else {
			*d1 /= gam * gam
			*b1 *= gam
			h11 *= gam
			h12 *= gam
		}
	}

	for *d2 <= rgamsq || *d2 >= gamsq {
		if *d2 == 0.0 {
			h11 = 1.0
			h12 = 0.0
			h21 = 0.0
			h22 = 1.0
			break
		}
		if *d2 <= rgamsq {
			*d2 *= gam * gam
			h21 /= gam
			h22 /= gam
		} else {
			*d2 /= gam * gam
			h21 *= gam
			h22 *= gam
		}
	}

	if math.Abs(u) <= 1.0 {
		p[0] = h11
		p[1] = h21
		p[2] = h12
		p[3] = h22
	} else {
		if u >= 0.0 {
			p[0] = 1.0
		} else {
			p[0] = -1.0
		}
		p[1] = h21 / u
		p[2] = h12 / u
		p[3] = h22 / u
		*b1 = u
	}
	return
}

// Drotm applies a modified Givens rotation to two vectors.
func Drotm(x, y *Vector, p [5]float64) error {
	if x.Size != y.Size {
		return fmt.Errorf("vectors have different sizes: %d != %d", x.Size, y.Size)
	}

	flag := p[0]
	var h11, h21, h12, h22 float64

	if flag == -2.0 { // Identity transformation, no-op
		return nil
	}

	if flag == -1.0 {
		h11 = p[1]
		h21 = p[2]
		h12 = p[3]
		h22 = p[4]
	} else if flag == 0.0 {
		h21 = p[2]
		h12 = p[3]
		h11 = 1.0
		h22 = 1.0
	} else if flag == 1.0 {
		h11 = p[1]
		h22 = p[4]
		h21 = -1.0
		h12 = 1.0
	} else {
		// Invalid flag value
		return fmt.Errorf("invalid flag value in Drotm: %f", flag)
	}

	for i := uint(0); i < x.Size; i++ {
		xi := x.Data[i*uint(x.Stride)]
		yi := y.Data[i*uint(y.Stride)]
		x.Data[i*uint(x.Stride)] = h11*xi + h12*yi
		y.Data[i*uint(y.Stride)] = h21*xi + h22*yi
	}

	return nil
}

// Sscal scales a float32 vector by a float32 scalar.
func Sscal(alpha float32, x *VectorFloat) {
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)] *= alpha
	}
}

// Dscal scales a vector by a scalar.
func Dscal(alpha float64, x *Vector) {
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)] *= alpha
	}
}

// Cscal scales a complex vector by a complex scalar.
func Cscal(alpha GslComplexFloat, x *VectorComplexFloat) {
	for i := uint(0); i < x.Size; i++ {
		a := x.Data[i*uint(x.Stride)].Data[0]
		b := x.Data[i*uint(x.Stride)].Data[1]
		x.Data[i*uint(x.Stride)].Data[0] = alpha.Data[0]*a - alpha.Data[1]*b
		x.Data[i*uint(x.Stride)].Data[1] = alpha.Data[0]*b + alpha.Data[1]*a
	}
}

// Zscal scales a complex vector by a complex scalar.
func Zscal(alpha GslComplex, x *VectorComplex) {
	for i := uint(0); i < x.Size; i++ {
		a := x.Data[i*uint(x.Stride)].Data[0]
		b := x.Data[i*uint(x.Stride)].Data[1]
		x.Data[i*uint(x.Stride)].Data[0] = alpha.Data[0]*a - alpha.Data[1]*b
		x.Data[i*uint(x.Stride)].Data[1] = alpha.Data[0]*b + alpha.Data[1]*a
	}
}

// Csscal scales a complex vector by a float32 scalar.
func Csscal(alpha float32, x *VectorComplexFloat) {
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)].Data[0] *= alpha
		x.Data[i*uint(x.Stride)].Data[1] *= alpha
	}
}

// Zdscal scales a complex vector by a float64 scalar.
func Zdscal(alpha float64, x *VectorComplex) {
	for i := uint(0); i < x.Size; i++ {
		x.Data[i*uint(x.Stride)].Data[0] *= alpha
		x.Data[i*uint(x.Stride)].Data[1] *= alpha
	}
}

// Level 2

// Sgemv computes y = alpha * A * x + beta * y  (general matrix-vector multiplication).
func Sgemv(transA CblasTranspose, alpha float32, a *MatrixFloat, x *VectorFloat, beta float32, y *VectorFloat) error {
	m := a.Size1
	n := a.Size2

	var rows, cols uint
	var xSize, ySize int
	if transA == CblasNoTrans {
		rows = m
		cols = n
		xSize = int(cols)
		ySize = int(rows)
	} else {
		rows = n
		cols = m
		xSize = int(cols)
		ySize = int(rows)
	}

	if x.Size != uint(xSize) {
		return fmt.Errorf("invalid x size: expected %d, got %d", xSize, x.Size)
	}
	if y.Size != uint(ySize) {
		return fmt.Errorf("invalid y size: expected %d, got %d", ySize, y.Size)
	}

	if transA == CblasNoTrans {
		for i := uint(0); i < m; i++ {
			var temp float32
			for j := uint(0); j < n; j++ {
				temp += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] = alpha*temp + beta*y.Data[i*uint(y.Stride)]
		}
	} else {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] *= beta
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)] += alpha * a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
			}
		}
	}

	return nil
}

// Strmv computes x = A * x (triangular matrix-vector multiplication).
func Strmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixFloat, x *VectorFloat) error {
	m := a.Size1
	n := a.Size2
	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
				}
				if i == 0 {
					break
				}
			}
		}
	} else { // CblasTrans or CblasConjTrans (same for real matrices)
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
			}
		}
	}

	return nil
}

// Strsv solves A * x = b for x, where A is a triangular matrix (triangular matrix-vector solve).
func Strsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixFloat, x *VectorFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
				}
				if i == 0 {
					break
				}

			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
				}
			}
		}
	} else { // CblasTrans or CblasConjTrans (same for real matrices)
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[j*uint(a.Tda)+i] * x.Data[i*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[j*uint(a.Tda)+i] * x.Data[i*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				if i == 0 {
					break
				}
			}
		}
	}

	return nil
}

// Dgemv computes y = alpha * A * x + beta * y (general matrix-vector multiplication).
func Dgemv(transA CblasTranspose, alpha float64, a *Matrix, x *Vector, beta float64, y *Vector) error {
	m := a.Size1
	n := a.Size2

	var rows, cols uint
	var xSize, ySize int

	if transA == CblasNoTrans {
		rows = m
		cols = n
		xSize = int(cols)
		ySize = int(rows)
	} else {
		rows = n
		cols = m
		xSize = int(cols)
		ySize = int(rows)
	}

	if x.Size != uint(xSize) {
		return fmt.Errorf("invalid x size: expected %d, got %d", xSize, x.Size)
	}

	if y.Size != uint(ySize) {
		return fmt.Errorf("invalid y size: expected %d, got %d", ySize, y.Size)
	}

	if transA == CblasNoTrans {
		for i := uint(0); i < m; i++ {
			var temp float64
			for j := uint(0); j < n; j++ {
				temp += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] = alpha*temp + beta*y.Data[i*uint(y.Stride)]
		}
	} else {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] *= beta
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)] += alpha * a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
			}
		}
	}
	return nil
}

// Dtrmv computes x = A * x (triangular matrix-vector multiplication).
func Dtrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *Matrix, x *Vector) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
				}
				if i == 0 {
					break
				}
			}
		}
	} else { // CblasTrans or CblasConjTrans (same for real matrices)
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)] += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] *= a.Data[i*uint(a.Tda)+i]
				}
			}
		}
	}
	return nil
}

// Dtrsv solves A * x = b for x, where A is a triangular matrix (triangular matrix-vector solve).
func Dtrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *Matrix, x *Vector) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[i*uint(a.Tda)+j] * x.Data[i*uint(x.Stride)]
				}
			}
		}
	} else { // CblasTrans
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[j*uint(a.Tda)+i] * x.Data[i*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)] -= a.Data[j*uint(a.Tda)+i] * x.Data[i*uint(x.Stride)]
				}
				if diag == CblasNonUnit {
					x.Data[i*uint(x.Stride)] /= a.Data[i*uint(a.Tda)+i]
				}
				if i == 0 {
					break
				}
			}
		}
	}
	return nil
}

// Cgemv computes y = alpha * A * x + beta * y  (general matrix-vector multiplication).
func Cgemv(transA CblasTranspose, alpha GslComplexFloat, a *MatrixComplexFloat, x *VectorComplexFloat, beta GslComplexFloat, y *VectorComplexFloat) error {
	m := a.Size1
	n := a.Size2

	var rows, cols uint
	var xSize, ySize int
	if transA == CblasNoTrans {
		rows = m
		cols = n
		xSize = int(cols)
		ySize = int(rows)
	} else if transA == CblasTrans || transA == CblasConjTrans {
		rows = n
		cols = m
		xSize = int(cols)
		ySize = int(rows)
	}

	if x.Size != uint(xSize) {
		return fmt.Errorf("invalid x size: expected %d, got %d", xSize, x.Size)
	}
	if y.Size != uint(ySize) {
		return fmt.Errorf("invalid y size: expected %d, got %d", ySize, y.Size)
	}
	if transA == CblasNoTrans {
		for i := uint(0); i < m; i++ {
			var temp GslComplexFloat
			for j := uint(0); j < n; j++ {
				temp.Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp.Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] = alpha.Data[0]*temp.Data[0] - alpha.Data[1]*temp.Data[1] + beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = alpha.Data[0]*temp.Data[1] + alpha.Data[1]*temp.Data[0] + beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
	} else if transA == CblasTrans {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] -
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] +
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0]
			}
		}
	} else { //CblasConjTrans
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] -
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] +
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0]
			}
		}
	}

	return nil
}

// Ctrmv computes x = A * x (triangular matrix-vector multiplication).
func Ctrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplexFloat, x *VectorComplexFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if i == 0 {
					break
				}
			}
		}
	} else if transA == CblasTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
			}
		}
	} else { //CblasConjTrans
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 + a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = -a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 + a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = -a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Ctrsv solves A * x = b for x, where A is a triangular matrix (triangular matrix-vector solve).
func Ctrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplexFloat, x *VectorComplexFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
			}
		}
	} else if transA == CblasTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				if i == 0 {
					break
				}
			}
		}
	} else { //CblasConjTrans
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] - t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (-t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] - t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (-t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				if i == 0 {
					break
				}
			}
		}
	}
	return nil
}

// Zgemv computes y = alpha * A * x + beta * y (general matrix-vector multiplication).
func Zgemv(transA CblasTranspose, alpha GslComplex, a *MatrixComplex, x *VectorComplex, beta GslComplex, y *VectorComplex) error {
	m := a.Size1
	n := a.Size2

	var rows, cols uint
	var xSize, ySize int
	if transA == CblasNoTrans {
		rows = m
		cols = n
		xSize = int(cols)
		ySize = int(rows)
	} else if transA == CblasTrans || transA == CblasConjTrans {
		rows = n
		cols = m
		xSize = int(cols)
		ySize = int(rows)
	}

	if x.Size != uint(xSize) {
		return fmt.Errorf("invalid x size: expected %d, got %d", xSize, x.Size)
	}
	if y.Size != uint(ySize) {
		return fmt.Errorf("invalid y size: expected %d, got %d", ySize, y.Size)
	}
	if transA == CblasNoTrans {
		for i := uint(0); i < m; i++ {
			var temp GslComplex
			for j := uint(0); j < n; j++ {
				temp.Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp.Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] = alpha.Data[0]*temp.Data[0] - alpha.Data[1]*temp.Data[1] + beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = alpha.Data[0]*temp.Data[1] + alpha.Data[1]*temp.Data[0] + beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
	} else if transA == CblasTrans {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] -
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] +
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0]
			}
		}
	} else { //CblasConjTrans
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[0] - beta.Data[1]*y.Data[i*uint(y.Stride)].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = beta.Data[0]*y.Data[i*uint(y.Stride)].Data[1] + beta.Data[1]*y.Data[i*uint(y.Stride)].Data[0]
		}
		for i := uint(0); i < m; i++ {
			for j := uint(0); j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] -
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] - alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1] +
					alpha.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0] + alpha.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0]
			}
		}
	}
	return nil
}

// Ztrmv computes x = A * x (triangular matrix-vector multiplication).
func Ztrmv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplex, x *VectorComplex) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if i == 0 {
					break
				}
			}
		}
	} else if transA == CblasTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 - a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
			}
		}
	} else { //CblasConjTrans
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				for j := i + 1; j < n; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 + a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = -a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				for j := uint(0); j < i; j++ {
					x.Data[i*uint(x.Stride)].Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[1] += -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = a.Data[i*uint(a.Tda)+i].Data[0]*t1 + a.Data[i*uint(a.Tda)+i].Data[1]*t2
					x.Data[i*uint(x.Stride)].Data[1] = -a.Data[i*uint(a.Tda)+i].Data[0]*t2 + a.Data[i*uint(a.Tda)+i].Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Ztrsv solves A * x = b for x, where A is a triangular matrix (triangular matrix-vector solve).
func Ztrsv(uplo CblasUplo, transA CblasTranspose, diag CblasDiag, a *MatrixComplex, x *VectorComplex) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if n != x.Size {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			for i := n - 1; ; i-- {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if i == 0 {
					break
				}
			}
		} else { // CblasLower
			for i := uint(0); i < n; i++ {
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
			}
		}
	} else if transA == CblasTrans {
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] + t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				if i == 0 {
					break
				}
			}
		}
	} else { //CblasConjTrans
		if uplo == CblasUpper {
			for i := uint(0); i < n; i++ {
				for j := i + 1; j < n; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] - t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (-t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
			}
		} else { // CblasLower
			for i := n - 1; ; i-- {
				for j := uint(0); j < i; j++ {
					x.Data[j*uint(x.Stride)].Data[0] -= a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[0] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[1]
					x.Data[j*uint(x.Stride)].Data[1] -= -a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[i*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[i*uint(x.Stride)].Data[0]
				}
				if diag == CblasNonUnit {
					t1 := x.Data[i*uint(x.Stride)].Data[0]
					t2 := x.Data[i*uint(x.Stride)].Data[1]
					x.Data[i*uint(x.Stride)].Data[0] = (t1*a.Data[i*uint(a.Tda)+i].Data[0] - t2*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
					x.Data[i*uint(x.Stride)].Data[1] = (-t2*a.Data[i*uint(a.Tda)+i].Data[0] - t1*a.Data[i*uint(a.Tda)+i].Data[1]) / (a.Data[i*uint(a.Tda)+i].Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] + a.Data[i*uint(a.Tda)+i].Data[1]*a.Data[i*uint(a.Tda)+i].Data[1])
				}
				if i == 0 {
					break
				}
			}
		}
	}
	return nil
}

// Sger computes A = alpha * x * y^T + A (rank-1 update).
func Sger(alpha float32, x *VectorFloat, y *VectorFloat, a *MatrixFloat) error {
	m := a.Size1
	n := a.Size2

	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}

	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * y.Data[j*uint(y.Stride)]
		}
	}

	return nil
}

// Dger computes A = alpha * x * y^T + A (rank-1 update).
func Dger(alpha float64, x *Vector, y *Vector, a *Matrix) error {
	m := a.Size1
	n := a.Size2

	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}

	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * y.Data[j*uint(y.Stride)]
		}
	}

	return nil
}

// Cgeru computes A = alpha * x * y^T + A (rank-1 update, unconjugated).
func Cgeru(alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error {
	m := a.Size1
	n := a.Size2
	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}
	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0])
			a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1])
		}
	}
	return nil
}

// Zgeru computes A = alpha * x * y^T + A (rank-1 update, unconjugated).
func Zgeru(alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error {
	m := a.Size1
	n := a.Size2
	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}
	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0])
			a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1])
		}
	}
	return nil
}

// Cgerc computes A = alpha * x * y^H + A (rank-1 update, conjugated).
func Cgerc(alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error {
	m := a.Size1
	n := a.Size2

	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}

	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0])
			a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1])
		}
	}

	return nil
}

// Zgerc computes A = alpha * x * y^H + A (rank-1 update, conjugated).
func Zgerc(alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error {
	m := a.Size1
	n := a.Size2

	if x.Size != m {
		return fmt.Errorf("invalid x size: expected %d, got %d", m, x.Size)
	}
	if y.Size != n {
		return fmt.Errorf("invalid y size: expected %d, got %d", n, y.Size)
	}

	for i := uint(0); i < m; i++ {
		for j := uint(0); j < n; j++ {
			a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0])
			a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
				alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1])
		}
	}

	return nil
}

// Cher computes A = alpha * x * x^H + A (Hermitian rank-1 update).
func Cher(uplo CblasUplo, alpha float32, x *VectorComplexFloat, a *MatrixComplexFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
				a.Data[i*uint(a.Tda)+j].Data[1] += alpha * (-x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
				a.Data[i*uint(a.Tda)+j].Data[1] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1] - x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])
			}
		}
	}
	return nil
}

// Zher computes A = alpha * x * x^H + A (Hermitian rank-1 update).
func Zher(uplo CblasUplo, alpha float64, x *VectorComplex, a *MatrixComplex) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
				a.Data[i*uint(a.Tda)+j].Data[1] += alpha * (-x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
				a.Data[i*uint(a.Tda)+j].Data[1] += alpha * (x.Data[i*uint(x.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1] - x.Data[i*uint(x.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])
			}
		}
	}
	return nil
}

// Cher2 computes A = alpha * x * y^H + conj(alpha) * y * x^H + A (Hermitian rank-2 update).
func Cher2(uplo CblasUplo, alpha GslComplexFloat, x *VectorComplexFloat, y *VectorComplexFloat, a *MatrixComplexFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n || y.Size != n {
		return fmt.Errorf("invalid vector sizes: matrix is %dx%d, vectors are %d and %d", n, n, x.Size, y.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[0]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1]) +
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]-y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])

				a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) +
					alpha.Data[0]*(-y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0]) -
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[0]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1]) +
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]-y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])

				a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) +
					alpha.Data[0]*(-y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0]) -
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
			}
		}
	}

	return nil
}

// Zher2 computes A = alpha * x * y^H + conj(alpha) * y * x^H + A (Hermitian rank-2 update).
func Zher2(uplo CblasUplo, alpha GslComplex, x *VectorComplex, y *VectorComplex, a *MatrixComplex) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n || y.Size != n {
		return fmt.Errorf("invalid vector sizes: matrix is %dx%d, vectors are %d and %d", n, n, x.Size, y.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[0]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1]) +
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]-y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])

				a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) +
					alpha.Data[0]*(-y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0]) -
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j].Data[0] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) -
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[0]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1]) +
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]-y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0])

				a.Data[i*uint(a.Tda)+j].Data[1] += alpha.Data[0]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[1]-x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[0]) +
					alpha.Data[1]*(x.Data[i*uint(x.Stride)].Data[0]*y.Data[j*uint(y.Stride)].Data[0]+x.Data[i*uint(x.Stride)].Data[1]*y.Data[j*uint(y.Stride)].Data[1]) +
					alpha.Data[0]*(-y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[1]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[0]) -
					alpha.Data[1]*(y.Data[i*uint(y.Stride)].Data[0]*x.Data[j*uint(x.Stride)].Data[0]+y.Data[i*uint(y.Stride)].Data[1]*x.Data[j*uint(x.Stride)].Data[1])
			}
		}
	}

	return nil
}

// Syr computes A = alpha * x * x^T + A (symmetric rank-1 update).
func Syr(uplo CblasUplo, alpha float32, x *VectorFloat, a *MatrixFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * x.Data[j*uint(x.Stride)]
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * x.Data[j*uint(x.Stride)]
			}
		}
	}
	return nil
}

// Dsyr computes A = alpha * x * x^T + A (symmetric rank-1 update).
func Dsyr(uplo CblasUplo, alpha float64, x *Vector, a *Matrix) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n {
		return fmt.Errorf("invalid vector size: matrix is %dx%d, vector is %d", n, n, x.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * x.Data[j*uint(x.Stride)]
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * x.Data[i*uint(x.Stride)] * x.Data[j*uint(x.Stride)]
			}
		}
	}
	return nil
}

// Syr2 computes A = alpha * x * y^T + alpha * y * x^T + A (symmetric rank-2 update).
func Syr2(uplo CblasUplo, alpha float32, x *VectorFloat, y *VectorFloat, a *MatrixFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n || y.Size != n {
		return fmt.Errorf("invalid vector sizes: matrix is %dx%d, vectors are %d and %d", n, n, x.Size, y.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * (x.Data[i*uint(x.Stride)]*y.Data[j*uint(y.Stride)] + y.Data[i*uint(y.Stride)]*x.Data[j*uint(x.Stride)])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * (x.Data[i*uint(x.Stride)]*y.Data[j*uint(y.Stride)] + y.Data[i*uint(y.Stride)]*x.Data[j*uint(x.Stride)])
			}
		}
	}

	return nil
}

// Dsyr2 computes A = alpha * x * y^T + alpha * y * x^T + A (symmetric rank-2 update).
func Dsyr2(uplo CblasUplo, alpha float64, x *Vector, y *Vector, a *Matrix) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix A must be square: %dx%d", m, n)
	}
	if x.Size != n || y.Size != n {
		return fmt.Errorf("invalid vector sizes: matrix is %dx%d, vectors are %d and %d", n, n, x.Size, y.Size)
	}

	if uplo == CblasUpper {
		for j := uint(0); j < n; j++ {
			for i := uint(0); i <= j; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * (x.Data[i*uint(x.Stride)]*y.Data[j*uint(y.Stride)] + y.Data[i*uint(y.Stride)]*x.Data[j*uint(x.Stride)])
			}
		}
	} else { // CblasLower
		for j := uint(0); j < n; j++ {
			for i := j; i < n; i++ {
				a.Data[i*uint(a.Tda)+j] += alpha * (x.Data[i*uint(x.Stride)]*y.Data[j*uint(y.Stride)] + y.Data[i*uint(y.Stride)]*x.Data[j*uint(x.Stride)])
			}
		}
	}

	return nil
}

// Ssymv  performs the matrix-vector  operation   y := alpha*A*x + beta*y,
func Ssymv(uplo CblasUplo, alpha float32, a *MatrixFloat, x *VectorFloat, beta float32, y *VectorFloat) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix must be square")
	}
	if n != x.Size || n != y.Size {
		return fmt.Errorf("invalid length")
	}
	if uplo == CblasUpper {
		for i := uint(0); i < n; i++ {
			temp1 := alpha * x.Data[i*uint(x.Stride)]
			temp2 := float32(0)
			y.Data[i*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+i]
			for j := i + 1; j < n; j++ {
				y.Data[j*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+j]
				temp2 += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] += alpha * temp2
		}

		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] = beta * y.Data[i*uint(y.Stride)]
		}

	} else {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] = beta * y.Data[i*uint(y.Stride)]
		}
		for i := uint(0); i < n; i++ {
			temp1 := alpha * x.Data[i*uint(x.Stride)]
			temp2 := float32(0)
			y.Data[i*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+i]
			for j := uint(0); j < i; j++ {
				y.Data[j*uint(y.Stride)] += temp1 * a.Data[j*uint(a.Tda)+i]
				temp2 += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] += alpha * temp2
		}

	}
	return nil
}

// Dsymv  performs the matrix-vector  operation   y := alpha*A*x + beta*y,
func Dsymv(uplo CblasUplo, alpha float64, a *Matrix, x *Vector, beta float64, y *Vector) error {
	m := a.Size1
	n := a.Size2

	if m != n {
		return fmt.Errorf("matrix must be square")
	}
	if n != x.Size || n != y.Size {
		return fmt.Errorf("invalid length")
	}
	if uplo == CblasUpper {
		for i := uint(0); i < n; i++ {
			temp1 := alpha * x.Data[i*uint(x.Stride)]
			temp2 := float64(0)
			y.Data[i*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+i]
			for j := i + 1; j < n; j++ {
				y.Data[j*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+j]
				temp2 += a.Data[i*uint(a.Tda)+j] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] += alpha * temp2
		}
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] = beta * y.Data[i*uint(y.Stride)]
		}

	} else {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)] = beta * y.Data[i*uint(y.Stride)]
		}
		for i := uint(0); i < n; i++ {
			temp1 := alpha * x.Data[i*uint(x.Stride)]
			temp2 := float64(0)
			y.Data[i*uint(y.Stride)] += temp1 * a.Data[i*uint(a.Tda)+i]
			for j := uint(0); j < i; j++ {
				y.Data[j*uint(y.Stride)] += temp1 * a.Data[j*uint(a.Tda)+i]
				temp2 += a.Data[j*uint(a.Tda)+i] * x.Data[j*uint(x.Stride)]
			}
			y.Data[i*uint(y.Stride)] += alpha * temp2
		}
	}
	return nil
}

// Chemv computes y = alpha * A * x + beta * y where A is hermitian (complex symmetric)
func Chemv(uplo CblasUplo, alpha GslComplexFloat, a *MatrixComplexFloat, x *VectorComplexFloat, beta GslComplexFloat, y *VectorComplexFloat) error {
	m := a.Size1
	n := a.Size2
	if m != n {
		return fmt.Errorf("matrix must be square")
	}
	if n != x.Size || n != y.Size {
		return fmt.Errorf("invalid length")
	}
	if uplo == CblasUpper {
		for i := uint(0); i < n; i++ {
			temp1 := GslComplexFloat{
				Data: [2]float32{
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1],
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0],
				},
			}
			temp2 := GslComplexFloat{}

			y.Data[i*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[0]

			for j := i + 1; j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]

				temp2.Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp2.Data[1] += -a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*temp2.Data[0] - alpha.Data[1]*temp2.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*temp2.Data[1] + alpha.Data[1]*temp2.Data[0]
		}

		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = y.Data[i*uint(y.Stride)].Data[0]*beta.Data[0] - y.Data[i*uint(y.Stride)].Data[1]*beta.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = y.Data[i*uint(y.Stride)].Data[1]*beta.Data[0] + y.Data[i*uint(y.Stride)].Data[0]*beta.Data[1]
		}
	} else { // CblasLower
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = y.Data[i*uint(y.Stride)].Data[0]*beta.Data[0] - y.Data[i*uint(y.Stride)].Data[1]*beta.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = y.Data[i*uint(y.Stride)].Data[1]*beta.Data[0] + y.Data[i*uint(y.Stride)].Data[0]*beta.Data[1]
		}
		for i := uint(0); i < n; i++ {
			temp1 := GslComplexFloat{
				Data: [2]float32{
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1],
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0],
				},
			}

			temp2 := GslComplexFloat{}

			y.Data[i*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[0]

			for j := uint(0); j < i; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[j*uint(a.Tda)+i].Data[0] + temp1.Data[1]*a.Data[j*uint(a.Tda)+i].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += -temp1.Data[0]*a.Data[j*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[j*uint(a.Tda)+i].Data[0]
				temp2.Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp2.Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*temp2.Data[0] - alpha.Data[1]*temp2.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*temp2.Data[1] + alpha.Data[1]*temp2.Data[0]
		}
	}
	return nil
}

// Zhemv computes y = alpha * A * x + beta * y where A is hermitian (complex symmetric)
func Zhemv(uplo CblasUplo, alpha GslComplex, a *MatrixComplex, x *VectorComplex, beta GslComplex, y *VectorComplex) error {
	m := a.Size1
	n := a.Size2
	if m != n {
		return fmt.Errorf("matrix must be square")
	}
	if n != x.Size || n != y.Size {
		return fmt.Errorf("invalid length")
	}

	if uplo == CblasUpper {
		for i := uint(0); i < n; i++ {
			temp1 := GslComplex{
				Data: [2]float64{
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1],
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0],
				},
			}
			temp2 := GslComplex{}

			y.Data[i*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[0]

			for j := i + 1; j < n; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+j].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+j].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+j].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+j].Data[0]

				temp2.Data[0] += a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[0] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp2.Data[1] += -a.Data[i*uint(a.Tda)+j].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[i*uint(a.Tda)+j].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*temp2.Data[0] - alpha.Data[1]*temp2.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*temp2.Data[1] + alpha.Data[1]*temp2.Data[0]
		}
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = y.Data[i*uint(y.Stride)].Data[0]*beta.Data[0] - y.Data[i*uint(y.Stride)].Data[1]*beta.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = y.Data[i*uint(y.Stride)].Data[1]*beta.Data[0] + y.Data[i*uint(y.Stride)].Data[0]*beta.Data[1]
		}

	} else {
		for i := uint(0); i < n; i++ {
			y.Data[i*uint(y.Stride)].Data[0] = y.Data[i*uint(y.Stride)].Data[0]*beta.Data[0] - y.Data[i*uint(y.Stride)].Data[1]*beta.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] = y.Data[i*uint(y.Stride)].Data[1]*beta.Data[0] + y.Data[i*uint(y.Stride)].Data[0]*beta.Data[1]
		}
		for i := uint(0); i < n; i++ {
			temp1 := GslComplex{
				Data: [2]float64{
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[0] - alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[1],
					alpha.Data[0]*x.Data[i*uint(x.Stride)].Data[1] + alpha.Data[1]*x.Data[i*uint(x.Stride)].Data[0],
				},
			}
			temp2 := GslComplex{}

			y.Data[i*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[0] - temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += temp1.Data[0]*a.Data[i*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[i*uint(a.Tda)+i].Data[0]

			for j := uint(0); j < i; j++ {
				y.Data[j*uint(y.Stride)].Data[0] += temp1.Data[0]*a.Data[j*uint(a.Tda)+i].Data[0] + temp1.Data[1]*a.Data[j*uint(a.Tda)+i].Data[1]
				y.Data[j*uint(y.Stride)].Data[1] += -temp1.Data[0]*a.Data[j*uint(a.Tda)+i].Data[1] + temp1.Data[1]*a.Data[j*uint(a.Tda)+i].Data[0]
				temp2.Data[0] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[0] - a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[1]
				temp2.Data[1] += a.Data[j*uint(a.Tda)+i].Data[0]*x.Data[j*uint(x.Stride)].Data[1] + a.Data[j*uint(a.Tda)+i].Data[1]*x.Data[j*uint(x.Stride)].Data[0]
			}
			y.Data[i*uint(y.Stride)].Data[0] += alpha.Data[0]*temp2.Data[0] - alpha.Data[1]*temp2.Data[1]
			y.Data[i*uint(y.Stride)].Data[1] += alpha.Data[0]*temp2.Data[1] + alpha.Data[1]*temp2.Data[0]
		}
	}
	return nil
}

// Level 3
// sgemm  performs one of the matrix-matrix operations
// C := alpha*op( A )*op( B ) + beta*C,
func Sgemm(transA, transB CblasTranspose, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error {
	var m, n, k int

	if transA == CblasNoTrans {
		m, k = int(a.Size1), int(a.Size2)
	} else {
		m, k = int(a.Size2), int(a.Size1)
	}
	if transB == CblasNoTrans {
		if int(b.Size1) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size2)
	} else {
		if int(b.Size2) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size1)
	}
	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("incompatible dimensions")
	}

	// CBLAS assumes column-major order, so we must adjust indices accordingly.
	if transA == CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a.Data[i*int(a.Tda)+l] * b.Data[l*int(b.Tda)+j]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else if transA != CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a.Data[l*int(a.Tda)+i] * b.Data[l*int(b.Tda)+j]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else if transA == CblasNoTrans && transB != CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a.Data[i*int(a.Tda)+l] * b.Data[j*int(b.Tda)+l]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else { // transA != CblasNoTrans && transB != CblasNoTrans
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for l := 0; l < k; l++ {
					sum += a.Data[l*int(a.Tda)+i] * b.Data[j*int(b.Tda)+l]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	}

	return nil
}

// Dgemm  performs one of the matrix-matrix operations
// C := alpha*op( A )*op( B ) + beta*C,
func Dgemm(transA, transB CblasTranspose, alpha float64, a, b *Matrix, beta float64, c *Matrix) error {
	var m, n, k int

	if transA == CblasNoTrans {
		m, k = int(a.Size1), int(a.Size2)
	} else {
		m, k = int(a.Size2), int(a.Size1)
	}
	if transB == CblasNoTrans {
		if int(b.Size1) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size2)
	} else {
		if int(b.Size2) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size1)
	}
	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("incompatible dimensions")
	}

	// CBLAS assumes column-major order, so we must adjust indices accordingly.
	if transA == CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float64(0)
				for l := 0; l < k; l++ {
					sum += a.Data[i*int(a.Tda)+l] * b.Data[l*int(b.Tda)+j]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else if transA != CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float64(0)
				for l := 0; l < k; l++ {
					sum += a.Data[l*int(a.Tda)+i] * b.Data[l*int(b.Tda)+j]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else if transA == CblasNoTrans && transB != CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float64(0)
				for l := 0; l < k; l++ {
					sum += a.Data[i*int(a.Tda)+l] * b.Data[j*int(b.Tda)+l]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else { // transA != CblasNoTrans && transB != CblasNoTrans
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float64(0)
				for l := 0; l < k; l++ {
					sum += a.Data[l*int(a.Tda)+i] * b.Data[j*int(b.Tda)+l]
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	}

	return nil
}

// Cgemm  performs one of the matrix-matrix operations
// C := alpha*op( A )*op( B ) + beta*C,
func Cgemm(transA, transB CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error {
	var m, n, k int

	if transA == CblasNoTrans {
		m, k = int(a.Size1), int(a.Size2)
	} else {
		m, k = int(a.Size2), int(a.Size1)
	}
	if transB == CblasNoTrans {
		if int(b.Size1) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size2)
	} else {
		if int(b.Size2) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size1)
	}
	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("incompatible dimensions")
	}
	if transA == CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplexFloat{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[l*int(b.Tda)+j].Data[1]
					sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[l*int(b.Tda)+j].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else if transA != CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplexFloat{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1]
					sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else if transA == CblasNoTrans && transB != CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplexFloat{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1]
					sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else { // transA != CblasNoTrans && transB != CblasNoTrans
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplexFloat{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[j*int(b.Tda)+l].Data[1]
					sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[j*int(b.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	}
	return nil
}

// Zgemm  performs one of the matrix-matrix operations
// C := alpha*op( A )*op( B ) + beta*C,
func Zgemm(transA, transB CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error {
	var m, n, k int

	if transA == CblasNoTrans {
		m, k = int(a.Size1), int(a.Size2)
	} else {
		m, k = int(a.Size2), int(a.Size1)
	}
	if transB == CblasNoTrans {
		if int(b.Size1) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size2)
	} else {
		if int(b.Size2) != k {
			return fmt.Errorf("incompatible dimensions")
		}
		n = int(b.Size1)
	}
	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("incompatible dimensions")
	}
	if transA == CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplex{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[l*int(b.Tda)+j].Data[1]
					sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[l*int(b.Tda)+j].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else if transA != CblasNoTrans && transB == CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplex{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1]
					sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else if transA == CblasNoTrans && transB != CblasNoTrans {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplex{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1]
					sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else { // transA != CblasNoTrans && transB != CblasNoTrans
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := GslComplex{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[j*int(b.Tda)+l].Data[1]
					sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[j*int(b.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	}
	return nil
}

// Ssymm performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Ssymm(side CblasSide, uplo CblasUplo, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := float32(0)
					temp2 := float32(0)
					for k := 0; k < i; k++ {
						temp1 += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[k*int(a.Tda)+i]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}
					for k := i + 1; k < m; k++ {
						temp1 += a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[i*int(a.Tda)+k]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)

					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp2)
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := float32(0)
					temp2 := float32(0)

					for k := 0; k < i; k++ {
						temp1 += a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[i*int(a.Tda)+k]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}

					for k := i + 1; k < m; k++ {
						temp1 += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[k*int(a.Tda)+i]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp2)
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := alpha * b.Data[i*int(b.Tda)+j]
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[k*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + temp*a.Data[j*int(a.Tda)+j]
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[j*int(a.Tda)+k]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := alpha * b.Data[i*int(b.Tda)+j]
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[j*int(a.Tda)+k]
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + temp*a.Data[j*int(a.Tda)+j]
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[k*int(a.Tda)+j]
					}
				}
			}
		}
	}

	return nil
}

// Dsymm performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Dsymm(side CblasSide, uplo CblasUplo, alpha float64, a, b *Matrix, beta float64, c *Matrix) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := float64(0)
					temp2 := float64(0)
					for k := 0; k < i; k++ {
						temp1 += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[k*int(a.Tda)+i]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}
					for k := i + 1; k < m; k++ {
						temp1 += a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[i*int(a.Tda)+k]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)

					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp2)
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := float64(0)
					temp2 := float64(0)

					for k := 0; k < i; k++ {
						temp1 += a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[i*int(a.Tda)+k]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}

					for k := i + 1; k < m; k++ {
						temp1 += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						temp2 += b.Data[k*int(b.Tda)+j] * a.Data[k*int(a.Tda)+i]
						c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp1)
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + alpha*(a.Data[i*int(a.Tda)+i]*b.Data[i*int(b.Tda)+j]+temp2)
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := alpha * b.Data[i*int(b.Tda)+j]
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[k*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + temp*a.Data[j*int(a.Tda)+j]
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[j*int(a.Tda)+k]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := alpha * b.Data[i*int(b.Tda)+j]
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[j*int(a.Tda)+k]
					}
					c.Data[i*int(c.Tda)+j] = beta*c.Data[i*int(c.Tda)+j] + temp*a.Data[j*int(a.Tda)+j]
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j] += temp * a.Data[k*int(a.Tda)+j]
					}
				}
			}
		}
	}

	return nil
}

// Csymm performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Csymm(side CblasSide, uplo CblasUplo, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplexFloat{}
					temp2 := GslComplexFloat{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplexFloat{}
					temp2 := GslComplexFloat{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplexFloat{
						Data: [2]float32{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] - temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] + temp.Data[0]*t2 + temp.Data[1]*t1

					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] - temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplexFloat{
						Data: [2]float32{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] - temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] + temp.Data[0]*t2 + temp.Data[1]*t1
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] - temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
				}
			}
		}
	}

	return nil
}

// Zsymm performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Zsymm(side CblasSide, uplo CblasUplo, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplex{}
					temp2 := GslComplex{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplex{}
					temp2 := GslComplex{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] - a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] - b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1-a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplex{
						Data: [2]float64{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] - temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] + temp.Data[0]*t2 + temp.Data[1]*t1

					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] - temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplex{
						Data: [2]float64{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] - temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] + temp.Data[0]*t2 + temp.Data[1]*t1
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] - temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
				}
			}
		}
	}

	return nil
}

// Chemm performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Chemm(side CblasSide, uplo CblasUplo, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplexFloat{}
					temp2 := GslComplexFloat{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(-a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2-a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplexFloat{}
					temp2 := GslComplexFloat{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(-a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2-a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplexFloat{
						Data: [2]float32{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] - temp.Data[0]*t2 - temp.Data[1]*t1

					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplexFloat{
						Data: [2]float32{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] - temp.Data[0]*t2 - temp.Data[1]*t1
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
				}
			}
		}
	}

	return nil
}

// Zhemm  performs one of the matrix-matrix operations   C := alpha*A*B + beta*C or  C := alpha*B*A + beta*C,
func Zhemm(side CblasSide, uplo CblasUplo, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(a.Size1), int(b.Size2)
		if int(a.Size2) != m {
			return fmt.Errorf("invalid dimensions")
		}
	} else {
		m, n = int(b.Size1), int(a.Size1)
		if int(a.Size2) != n {
			return fmt.Errorf("invalid dimensions")
		}
	}

	if int(b.Size1) != m || int(b.Size2) != n {
		return fmt.Errorf("invalid dimensions for B")
	}

	if int(c.Size1) != m || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if side == CblasLeft {
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplex{}
					temp2 := GslComplex{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(-a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2-a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp1 := GslComplex{}
					temp2 := GslComplex{}
					for k := 0; k < i; k++ {
						temp1.Data[0] += a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[i*int(a.Tda)+k].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[i*int(a.Tda)+k].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[i*int(a.Tda)+k].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[i*int(a.Tda)+k].Data[0]
					}
					for k := i + 1; k < m; k++ {
						temp1.Data[0] += a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[0] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[1]
						temp1.Data[1] += -a.Data[k*int(a.Tda)+i].Data[0]*b.Data[k*int(b.Tda)+j].Data[1] + a.Data[k*int(a.Tda)+i].Data[1]*b.Data[k*int(b.Tda)+j].Data[0]
						temp2.Data[0] += b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[0] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[1]
						temp2.Data[1] += -b.Data[k*int(b.Tda)+j].Data[0]*a.Data[k*int(a.Tda)+i].Data[1] + b.Data[k*int(b.Tda)+j].Data[1]*a.Data[k*int(a.Tda)+i].Data[0]
					}
					t1 := b.Data[i*int(b.Tda)+j].Data[0]
					t2 := b.Data[i*int(b.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0]) -
						alpha.Data[1]*(-a.Data[i*int(a.Tda)+i].Data[0]*t2+a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1])
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] +
						alpha.Data[0]*(a.Data[i*int(a.Tda)+i].Data[0]*t2-a.Data[i*int(a.Tda)+i].Data[1]*t1+temp1.Data[1]) +
						alpha.Data[1]*(a.Data[i*int(a.Tda)+i].Data[0]*t1+a.Data[i*int(a.Tda)+i].Data[1]*t2+temp1.Data[0])
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplex{
						Data: [2]float64{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] - temp.Data[0]*t2 - temp.Data[1]*t1

					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
				}
			}
		} else { // CblasLower
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					temp := GslComplex{
						Data: [2]float64{
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[0] - alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[1],
							alpha.Data[0]*b.Data[i*int(b.Tda)+j].Data[1] + alpha.Data[1]*b.Data[i*int(b.Tda)+j].Data[0],
						},
					}
					for k := 0; k < j; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[0] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[j*int(a.Tda)+k].Data[1] + temp.Data[1]*a.Data[j*int(a.Tda)+k].Data[0]
					}
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[0] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1] + temp.Data[0]*t1 - temp.Data[1]*t2
					c.Data[i*int(c.Tda)+j].Data[1] = beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0] - temp.Data[0]*t2 - temp.Data[1]*t1
					for k := j + 1; k < n; k++ {
						c.Data[i*int(c.Tda)+j].Data[0] += temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[0] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[1]
						c.Data[i*int(c.Tda)+j].Data[1] += -temp.Data[0]*a.Data[k*int(a.Tda)+j].Data[1] + temp.Data[1]*a.Data[k*int(a.Tda)+j].Data[0]
					}
				}
			}
		}
	}

	return nil
}

// Ssyrk performs one of the symmetric rank k operations  C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
func Ssyrk(uplo CblasUplo, trans CblasTranspose, alpha float32, a *MatrixFloat, beta float32, c *MatrixFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}
	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sum := float32(0.0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l] * a.Data[j*int(a.Tda)+l]
					}
				} else {
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i] * a.Data[l*int(a.Tda)+j]
					}
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sum := float32(0.0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l] * a.Data[j*int(a.Tda)+l]
					}
				} else {
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i] * a.Data[l*int(a.Tda)+j]
					}
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	}

	return nil
}

// Dsyrk performs one of the symmetric rank k operations  C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
func Dsyrk(uplo CblasUplo, trans CblasTranspose, alpha float64, a *Matrix, beta float64, c *Matrix) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}
	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sum := float64(0.0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l] * a.Data[j*int(a.Tda)+l]
					}
				} else {
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i] * a.Data[l*int(a.Tda)+j]
					}
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sum := float64(0.0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l] * a.Data[j*int(a.Tda)+l]
					}
				} else {
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i] * a.Data[l*int(a.Tda)+j]
					}
				}
				c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
			}
		}
	}

	return nil
}

// Csyrk performs one of the symmetric rank k operations  C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
func Csyrk(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}
	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sum := GslComplexFloat{}
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sum := GslComplexFloat{}
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	}

	return nil
}

// Zsyrk performs one of the symmetric rank k operations  C := alpha*A*A**T + beta*C or C := alpha*A**T*A + beta*C
func Zsyrk(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a *MatrixComplex, beta GslComplex, c *MatrixComplex) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}
	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sum := GslComplex{}
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sum := GslComplex{}
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else {
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	}

	return nil
}

// Cherk performs one of the hermitian rank k operations  C := alpha*A*A**H + beta*C or C := alpha*A**H*A + beta*C
func Cherk(uplo CblasUplo, trans CblasTranspose, alpha float32, a *MatrixComplexFloat, beta float32, c *MatrixComplexFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}
	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sumRe := float32(0)
				sumIm := float32(0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else { // CblasConjTrans
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[0]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[1]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sumRe := float32(0)
				sumIm := float32(0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else { // CblasConjTrans
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[0]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[1]
			}
		}
	}
	return nil
}

// Zherk performs one of the hermitian rank k operations  C := alpha*A*A**H + beta*C or C := alpha*A**H*A + beta*C
func Zherk(uplo CblasUplo, trans CblasTranspose, alpha float64, a *MatrixComplex, beta float64, c *MatrixComplex) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
	} else {
		n, k = int(a.Size2), int(a.Size1)
	}
	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				sumRe := float64(0)
				sumIm := float64(0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else { // CblasConjTrans
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[0]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[1]
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sumRe := float64(0)
				sumIm := float64(0)
				if trans == CblasNoTrans {
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
				} else { // CblasConjTrans
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[0]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[1]
			}
		}
	}
	return nil
}

// Ssyr2k performs one of the symmetric rank 2k operations C := alpha*A*B**T + alpha*B*A**T + beta*C or C := alpha*A**T*B + alpha*B**T*A + beta*C
func Ssyr2k(uplo CblasUplo, trans CblasTranspose, alpha float32, a, b *MatrixFloat, beta float32, c *MatrixFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}
	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := float32(0)
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l]*b.Data[j*int(b.Tda)+l] + b.Data[i*int(b.Tda)+l]*a.Data[j*int(a.Tda)+l]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := float32(0)
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i]*b.Data[l*int(b.Tda)+j] + b.Data[l*int(b.Tda)+i]*a.Data[l*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		}
	} else { // CblasLower
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := float32(0)
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l]*b.Data[j*int(b.Tda)+l] + b.Data[i*int(b.Tda)+l]*a.Data[j*int(a.Tda)+l]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := float32(0)
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i]*b.Data[l*int(b.Tda)+j] + b.Data[l*int(b.Tda)+i]*a.Data[l*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		}
	}
	return nil
}

// Dsyr2k performs one of the symmetric rank 2k operations C := alpha*A*B**T + alpha*B*A**T + beta*C or C := alpha*A**T*B + alpha*B**T*A + beta*C
func Dsyr2k(uplo CblasUplo, trans CblasTranspose, alpha float64, a, b *Matrix, beta float64, c *Matrix) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}
	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := float64(0)
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l]*b.Data[j*int(b.Tda)+l] + b.Data[i*int(b.Tda)+l]*a.Data[j*int(a.Tda)+l]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := float64(0)
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i]*b.Data[l*int(b.Tda)+j] + b.Data[l*int(b.Tda)+i]*a.Data[l*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		}
	} else { // CblasLower
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := float64(0)
					for l := 0; l < k; l++ {
						sum += a.Data[i*int(a.Tda)+l]*b.Data[j*int(b.Tda)+l] + b.Data[i*int(b.Tda)+l]*a.Data[j*int(a.Tda)+l]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := float64(0)
					for l := 0; l < k; l++ {
						sum += a.Data[l*int(a.Tda)+i]*b.Data[l*int(b.Tda)+j] + b.Data[l*int(b.Tda)+i]*a.Data[l*int(a.Tda)+j]
					}
					c.Data[i*int(c.Tda)+j] = alpha*sum + beta*c.Data[i*int(c.Tda)+j]
				}
			}
		}
	}
	return nil
}

// Csyr2k performs one of the symmetric rank 2k operations C := alpha*A*B**T + alpha*B*A**T + beta*C or C := alpha*A**T*B + alpha*B**T*A + beta*C
func Csyr2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta GslComplexFloat, c *MatrixComplexFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}
	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := GslComplexFloat{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := GslComplexFloat{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		}
	} else { // CblasLower
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := GslComplexFloat{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sum := GslComplexFloat{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		}
	}
	return nil
}

// Zsyr2k performs one of the symmetric rank 2k operations C := alpha*A*B**T + alpha*B*A**T + beta*C or C := alpha*A**T*B + alpha*B**T*A + beta*C
func Zsyr2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta GslComplex, c *MatrixComplex) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}
	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := GslComplex{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		} else { // CblasTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sum := GslComplex{}
					for l := 0; l < k; l++ {
						sum.Data[0] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] - b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sum.Data[1] += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
				}
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sum := GslComplex{}
				for l := 0; l < k; l++ {
					sum.Data[0] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] - a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
						b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] - b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
					sum.Data[1] += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] +
						b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sum.Data[0] - alpha.Data[1]*sum.Data[1] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[0] - beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[1]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sum.Data[1] + alpha.Data[1]*sum.Data[0] + beta.Data[0]*c.Data[i*int(c.Tda)+j].Data[1] + beta.Data[1]*c.Data[i*int(c.Tda)+j].Data[0]
			}
		}
	}
	return nil
}

// Cher2k performs one of the hermitian rank 2k operations  C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C   or   C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
func Cher2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplexFloat, a, b *MatrixComplexFloat, beta float32, c *MatrixComplexFloat) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}

	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sumRe := float32(0)
					sumIm := float32(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] -
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		} else { //CblasConjTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sumRe := float32(0)
					sumIm := float32(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] -
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		}
	} else { // CblasLower
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sumRe := float32(0)
					sumIm := float32(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] -
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		} else { //CblasConjTrans
			for i := 0; i < n; i++ {
				for j := 0; j <= i; j++ {
					sumRe := float32(0)
					sumIm := float32(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] -
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		}
	}
	return nil
}

// Zher2k performs one of the hermitian rank 2k operations  C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C   or   C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,
func Zher2k(uplo CblasUplo, trans CblasTranspose, alpha GslComplex, a, b *MatrixComplex, beta float64, c *MatrixComplex) error {
	var n, k int
	if trans == CblasNoTrans {
		n, k = int(a.Size1), int(a.Size2)
		if int(b.Size1) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	} else {
		n, k = int(a.Size2), int(a.Size1)
		if int(b.Size2) != n {
			return fmt.Errorf("matrix B has incompatible dimensions")
		}
	}
	if int(b.Size2) != k && trans == CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}
	if int(b.Size1) != k && trans != CblasNoTrans {
		return fmt.Errorf("matrix B has incompatible dimensions")
	}

	if int(c.Size1) != n || int(c.Size2) != n {
		return fmt.Errorf("invalid dimensions for C")
	}

	if uplo == CblasUpper {
		if trans == CblasNoTrans {
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sumRe := float64(0)
					sumIm := float64(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
						sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] -
							b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		} else { //CblasConjTrans
			for i := 0; i < n; i++ {
				for j := i; j < n; j++ {
					sumRe := float64(0)
					sumIm := float64(0)
					for l := 0; l < k; l++ {
						sumRe += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[0] + a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[1] +
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[0] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[1]
						sumIm += a.Data[l*int(a.Tda)+i].Data[0]*b.Data[l*int(b.Tda)+j].Data[1] - a.Data[l*int(a.Tda)+i].Data[1]*b.Data[l*int(b.Tda)+j].Data[0] -
							b.Data[l*int(b.Tda)+i].Data[0]*a.Data[l*int(a.Tda)+j].Data[1] + b.Data[l*int(b.Tda)+i].Data[1]*a.Data[l*int(a.Tda)+j].Data[0]
					}
					c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
					c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
				}
			}
		}
	} else { // CblasLower
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				sumRe := float64(0)
				sumIm := float64(0)
				for l := 0; l < k; l++ {
					sumRe += a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[0] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[1] +
						b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[0] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[1]
					sumIm += -a.Data[i*int(a.Tda)+l].Data[0]*b.Data[j*int(b.Tda)+l].Data[1] + a.Data[i*int(a.Tda)+l].Data[1]*b.Data[j*int(b.Tda)+l].Data[0] -
						b.Data[i*int(b.Tda)+l].Data[0]*a.Data[j*int(a.Tda)+l].Data[1] + b.Data[i*int(b.Tda)+l].Data[1]*a.Data[j*int(a.Tda)+l].Data[0]
				}
				c.Data[i*int(c.Tda)+j].Data[0] = alpha.Data[0]*sumRe - alpha.Data[1]*sumIm + beta*c.Data[i*int(c.Tda)+j].Data[0]
				c.Data[i*int(c.Tda)+j].Data[1] = alpha.Data[0]*sumIm + alpha.Data[1]*sumRe + beta*c.Data[i*int(c.Tda)+j].Data[1]
			}
		}
	}
	return nil
}

// Strmm performs one of the matrix-matrix operations  B := alpha*op( A )*B, or B := alpha*B*op( A )
func Strmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float32, a *MatrixFloat, b *MatrixFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}
	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := float32(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
						for k := i + 1; k < m; k++ {
							b.Data[i*int(b.Tda)+j] += alpha * a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := float32(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
						for k := 0; k < i; k++ {
							b.Data[i*int(b.Tda)+j] += alpha * a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						}
					}
				}
			}
		} else { // CblasTrans
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := float32(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						for k := i + 1; k < m; k++ {
							temp += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := float32(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						for k := 0; k < i; k++ {
							temp += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
					}
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] *= a.Data[j*int(a.Tda)+j]
					}
					for k := 0; k < j; k++ {
						b.Data[i*int(b.Tda)+j] += a.Data[k*int(a.Tda)+j] * b.Data[i*int(b.Tda)+k]
					}
					b.Data[i*int(b.Tda)+j] *= alpha
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] *= a.Data[j*int(a.Tda)+j]
					}
					for k := j + 1; k < n; k++ {
						b.Data[i*int(b.Tda)+j] += a.Data[k*int(a.Tda)+j] * b.Data[i*int(b.Tda)+k]
					}
					b.Data[i*int(b.Tda)+j] *= alpha
				}
			}
		}
	}

	return nil
}

// Dtrmm performs one of the matrix-matrix operations  B := alpha*op( A )*B, or B := alpha*B*op( A )
func Dtrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float64, a *Matrix, b *Matrix) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}
	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := float64(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
						for k := i + 1; k < m; k++ {
							b.Data[i*int(b.Tda)+j] += alpha * a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := float64(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
						for k := 0; k < i; k++ {
							b.Data[i*int(b.Tda)+j] += alpha * a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j]
						}
					}
				}
			}
		} else { // CblasTrans
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := float64(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						for k := i + 1; k < m; k++ {
							temp += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := float64(0)
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = 1
						}
						for k := 0; k < i; k++ {
							temp += a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j]
						}
						b.Data[i*int(b.Tda)+j] = alpha * temp * b.Data[i*int(b.Tda)+j]
					}
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] *= a.Data[j*int(a.Tda)+j]
					}
					for k := 0; k < j; k++ {
						b.Data[i*int(b.Tda)+j] += a.Data[k*int(a.Tda)+j] * b.Data[i*int(b.Tda)+k]
					}
					b.Data[i*int(b.Tda)+j] *= alpha
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] *= a.Data[j*int(a.Tda)+j]
					}
					for k := j + 1; k < n; k++ {
						b.Data[i*int(b.Tda)+j] += a.Data[k*int(a.Tda)+j] * b.Data[i*int(b.Tda)+k]
					}
					b.Data[i*int(b.Tda)+j] *= alpha
				}
			}
		}
	}

	return nil
}

// Ctrmm performs one of the matrix-matrix operations  B := alpha*op( A )*B, or B := alpha*B*op( A )
func Ctrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplexFloat, a *MatrixComplexFloat, b *MatrixComplexFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}
	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplexFloat{}
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = GslComplexFloat{Data: [2]float32{1, 0}} // 1 + 0i
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)

						for k := i + 1; k < m; k++ {
							t1 = b.Data[k*int(b.Tda)+j].Data[0]
							t2 = b.Data[k*int(b.Tda)+j].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2) -
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1)
							b.Data[i*int(b.Tda)+j].Data[1] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1) +
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2)
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplexFloat{}
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = GslComplexFloat{Data: [2]float32{1, 0}}
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)
						for k := 0; k < i; k++ {
							t1 = b.Data[k*int(b.Tda)+j].Data[0]
							t2 = b.Data[k*int(b.Tda)+j].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2) -
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1)
							b.Data[i*int(b.Tda)+j].Data[1] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1) +
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2)
						}
					}
				}
			}
		} else if transA == CblasTrans {
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplexFloat{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplexFloat{Data: [2]float32{t1, t2}}
						} else {
							temp = GslComplexFloat{Data: [2]float32{1, 0}}
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] - t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)

					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplexFloat{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplexFloat{Data: [2]float32{t1, t2}}
						} else {
							temp = GslComplexFloat{Data: [2]float32{1, 0}}
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] - t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)
					}
				}
			}
		}
	} else { //CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 + b.Data[i*int(b.Tda)+j].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = -b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := 0; k < j; k++ {
						t1 = a.Data[k*int(a.Tda)+j].Data[0]
						t2 = a.Data[k*int(a.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 + b.Data[i*int(b.Tda)+k].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] += -b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 + b.Data[i*int(b.Tda)+j].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = -b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := j + 1; k < n; k++ {
						t1 = a.Data[k*int(a.Tda)+j].Data[0]
						t2 = a.Data[k*int(a.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 + b.Data[i*int(b.Tda)+k].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] += -b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Ztrmm performs one of the matrix-matrix operations  B := alpha*op( A )*B, or B := alpha*B*op( A )
func Ztrmm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplex, a *MatrixComplex, b *MatrixComplex) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}
	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}} // 1 + 0i
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)

						for k := i + 1; k < m; k++ {
							t1 = b.Data[k*int(b.Tda)+j].Data[0]
							t2 = b.Data[k*int(b.Tda)+j].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2) -
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1)
							b.Data[i*int(b.Tda)+j].Data[1] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1) +
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2)
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							temp = a.Data[i*int(a.Tda)+i]
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}}
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)
						for k := 0; k < i; k++ {
							t1 = b.Data[k*int(b.Tda)+j].Data[0]
							t2 = b.Data[k*int(b.Tda)+j].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2) -
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1)
							b.Data[i*int(b.Tda)+j].Data[1] += alpha.Data[0]*(a.Data[i*int(a.Tda)+k].Data[0]*t2+a.Data[i*int(a.Tda)+k].Data[1]*t1) +
								alpha.Data[1]*(a.Data[i*int(a.Tda)+k].Data[0]*t1-a.Data[i*int(a.Tda)+k].Data[1]*t2)
						}
					}
				}
			}
		} else if transA == CblasTrans {
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplex{Data: [2]float64{t1, t2}}
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}}
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] - t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)

					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplex{Data: [2]float64{t1, t2}}
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}}
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] - t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)
					}
				}
			}
		} else if transA == CblasConjTrans {
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplex{Data: [2]float64{t1, -t2}}
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}} // 1 + 0i
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] + t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += -t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)

					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp = GslComplex{Data: [2]float64{t1, -t2}}
						} else {
							temp = GslComplex{Data: [2]float64{1, 0}}
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							temp.Data[0] += t1*b.Data[k*int(b.Tda)+j].Data[0] + t2*b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[1] += -t1*b.Data[k*int(b.Tda)+j].Data[1] + t2*b.Data[k*int(b.Tda)+j].Data[0]
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*(temp.Data[0]*t1-temp.Data[1]*t2) - alpha.Data[1]*(temp.Data[0]*t2+temp.Data[1]*t1)
						b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*(temp.Data[0]*t2+temp.Data[1]*t1) + alpha.Data[1]*(temp.Data[0]*t1-temp.Data[1]*t2)
					}
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 - b.Data[i*int(b.Tda)+j].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] = b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := 0; k < j; k++ {
						t1 = a.Data[k*int(a.Tda)+j].Data[0]
						t2 = a.Data[k*int(a.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 - b.Data[i*int(b.Tda)+k].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] += b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 - b.Data[i*int(b.Tda)+j].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] = b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := j + 1; k < n; k++ {
						t1 = a.Data[k*int(a.Tda)+j].Data[0]
						t2 = a.Data[k*int(a.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 - b.Data[i*int(b.Tda)+k].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] += b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Strsm solves one of the matrix equations   op( A )*X = alpha*B, or X*op( A ) = alpha*B
func Strsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float32, a *MatrixFloat, b *MatrixFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := alpha
						for k := 0; k < i; k++ {
							temp -= a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := alpha
						for k := i + 1; k < m; k++ {
							temp -= a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		} else { // CblasTrans
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := alpha
						for k := i + 1; k < m; k++ {
							temp -= a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := alpha
						for k := 0; k < i; k++ {
							temp -= a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		}
	} else { //CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					temp := alpha
					for k := 0; k < j; k++ {
						temp -= a.Data[j*int(a.Tda)+k] * b.Data[i*int(b.Tda)+k] / alpha
					}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] = temp / a.Data[j*int(a.Tda)+j]
					} else {
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					temp := alpha
					for k := j + 1; k < n; k++ {
						temp -= a.Data[j*int(a.Tda)+k] * b.Data[i*int(b.Tda)+k] / alpha
					}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] = temp / a.Data[j*int(a.Tda)+j]
					} else {
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		}
	}
	return nil
}

// Dtrsm solves one of the matrix equations   op( A )*X = alpha*B, or X*op( A ) = alpha*B
func Dtrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha float64, a *Matrix, b *Matrix) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}
	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := alpha
						for k := 0; k < i; k++ {
							temp -= a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := alpha
						for k := i + 1; k < m; k++ {
							temp -= a.Data[k*int(a.Tda)+i] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		} else { // CblasTrans
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := alpha
						for k := i + 1; k < m; k++ {
							temp -= a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := alpha
						for k := 0; k < i; k++ {
							temp -= a.Data[i*int(a.Tda)+k] * b.Data[k*int(b.Tda)+j] / alpha
						}
						if diag == CblasNonUnit {
							b.Data[i*int(b.Tda)+j] = temp / a.Data[i*int(a.Tda)+i]
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		}
	} else { //CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					temp := alpha
					for k := 0; k < j; k++ {
						temp -= a.Data[j*int(a.Tda)+k] * b.Data[i*int(b.Tda)+k] / alpha
					}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] = temp / a.Data[j*int(a.Tda)+j]
					} else {
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					temp := alpha
					for k := j + 1; k < n; k++ {
						temp -= a.Data[j*int(a.Tda)+k] * b.Data[i*int(b.Tda)+k] / alpha
					}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j] = temp / a.Data[j*int(a.Tda)+j]
					} else {
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		}
	}
	return nil
}

// Ctrsm solves one of the matrix equations   op( A )*X = alpha*B, or X*op( A ) = alpha*B
func Ctrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplexFloat, a *MatrixComplexFloat, b *MatrixComplexFloat) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						if diag == CblasNonUnit {
							t3 := a.Data[i*int(a.Tda)+i].Data[0]
							t4 := a.Data[i*int(a.Tda)+i].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] = (temp.Data[0]*t3 + temp.Data[1]*t4) / (t3*t3 + t4*t4)
							b.Data[i*int(b.Tda)+j].Data[1] = (temp.Data[1]*t3 - temp.Data[0]*t4) / (t3*t3 + t4*t4)
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						if diag == CblasNonUnit {
							t3 := a.Data[i*int(a.Tda)+i].Data[0]
							t4 := a.Data[i*int(a.Tda)+i].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] = (temp.Data[0]*t3 + temp.Data[1]*t4) / (t3*t3 + t4*t4)
							b.Data[i*int(b.Tda)+j].Data[1] = (temp.Data[1]*t3 - temp.Data[0]*t4) / (t3*t3 + t4*t4)
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		} else if transA == CblasTrans {
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
							temp.Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[i*int(a.Tda)+k].Data[0]
							t2 := a.Data[i*int(a.Tda)+k].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						b.Data[i*int(b.Tda)+j] = temp

					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}

						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
							temp.Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[i*int(a.Tda)+k].Data[0]
							t2 := a.Data[i*int(a.Tda)+k].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		}
	} else { //CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 + b.Data[i*int(b.Tda)+j].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = -b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := 0; k < j; k++ {
						t1 = a.Data[j*int(a.Tda)+k].Data[0]
						t2 = a.Data[j*int(a.Tda)+k].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 + b.Data[i*int(b.Tda)+k].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] += -b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					//temp := GslComplexFloat{Data: [2]float32{alpha.Data[0], alpha.Data[1]}}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 + b.Data[i*int(b.Tda)+j].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = -b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := j + 1; k < n; k++ {
						t1 = a.Data[j*int(a.Tda)+k].Data[0]
						t2 = a.Data[j*int(a.Tda)+k].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 + b.Data[i*int(b.Tda)+k].Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] += -b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Ztrsm solves one of the matrix equations   op( A )*X = alpha*B, or X*op( A ) = alpha*B
func Ztrsm(side CblasSide, uplo CblasUplo, transA CblasTranspose, diag CblasDiag, alpha GslComplex, a *MatrixComplex, b *MatrixComplex) error {
	var m, n int
	if side == CblasLeft {
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != m || int(a.Size2) != m {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	} else { // CblasRight
		m, n = int(b.Size1), int(b.Size2)
		if int(a.Size1) != n || int(a.Size2) != n {
			return fmt.Errorf("matrix A must be square and its size must match the corresponding dimension of B")
		}
	}

	if transA == CblasNoTrans {
		if uplo == CblasUpper {
			if side == CblasLeft {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						if diag == CblasNonUnit {
							t3 := a.Data[i*int(a.Tda)+i].Data[0]
							t4 := a.Data[i*int(a.Tda)+i].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] = (temp.Data[0]*t3 + temp.Data[1]*t4) / (t3*t3 + t4*t4)
							b.Data[i*int(b.Tda)+j].Data[1] = (temp.Data[1]*t3 - temp.Data[0]*t4) / (t3*t3 + t4*t4)
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			} else { // CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						if diag == CblasNonUnit {
							t3 := a.Data[i*int(a.Tda)+i].Data[0]
							t4 := a.Data[i*int(a.Tda)+i].Data[1]
							b.Data[i*int(b.Tda)+j].Data[0] = (temp.Data[0]*t3 + temp.Data[1]*t4) / (t3*t3 + t4*t4)
							b.Data[i*int(b.Tda)+j].Data[1] = (temp.Data[1]*t3 - temp.Data[0]*t4) / (t3*t3 + t4*t4)
						} else {
							b.Data[i*int(b.Tda)+j] = temp
						}
					}
				}
			}
		} else if transA == CblasTrans {
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
							temp.Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[i*int(a.Tda)+k].Data[0]
							t2 := a.Data[i*int(a.Tda)+k].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						b.Data[i*int(b.Tda)+j] = temp

					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
							temp.Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[i*int(a.Tda)+k].Data[0]
							t2 := a.Data[i*int(a.Tda)+k].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 - t2*t4
							temp.Data[1] -= t1*t4 + t2*t3
						}
						b.Data[i*int(b.Tda)+j] = temp
					}
				}
			}
		} else { //CblasConjTrans
			if uplo == CblasUpper {
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 + temp.Data[1]*t2
							temp.Data[1] = -temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := i + 1; k < m; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 + t2*t4
							temp.Data[1] -= -t1*t4 + t2*t3
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
					}
				}
			} else { //CblasLower
				for j := 0; j < n; j++ {
					for i := m - 1; i >= 0; i-- {
						temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
						if diag == CblasNonUnit {
							t1 := a.Data[i*int(a.Tda)+i].Data[0]
							t2 := a.Data[i*int(a.Tda)+i].Data[1]
							temp.Data[0] = temp.Data[0]*t1 + temp.Data[1]*t2
							temp.Data[1] = -temp.Data[0]*t2 + temp.Data[1]*t1
						}
						for k := 0; k < i; k++ {
							t1 := a.Data[k*int(a.Tda)+i].Data[0]
							t2 := a.Data[k*int(a.Tda)+i].Data[1]
							t3 := b.Data[k*int(b.Tda)+j].Data[0]
							t4 := b.Data[k*int(b.Tda)+j].Data[1]
							temp.Data[0] -= t1*t3 + t2*t4
							temp.Data[1] -= -t1*t4 + t2*t3
						}
						t1 := b.Data[i*int(b.Tda)+j].Data[0]
						t2 := b.Data[i*int(b.Tda)+j].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
						b.Data[i*int(b.Tda)+j].Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
					}
				}
			}
		}
	} else { // CblasRight
		if uplo == CblasUpper {
			for j := n - 1; j >= 0; j-- {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 - b.Data[i*int(b.Tda)+j].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] = b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := 0; k < j; k++ {
						t1 = a.Data[j*int(a.Tda)+k].Data[0]
						t2 = a.Data[j*int(a.Tda)+k].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 - b.Data[i*int(b.Tda)+k].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] += b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = temp.Data[0]*t1 - temp.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = temp.Data[0]*t2 + temp.Data[1]*t1
				}
			}
		} else { // CblasLower
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					t1 := a.Data[j*int(a.Tda)+j].Data[0]
					t2 := a.Data[j*int(a.Tda)+j].Data[1]
					//temp := GslComplex{Data: [2]float64{alpha.Data[0], alpha.Data[1]}}
					if diag == CblasNonUnit {
						b.Data[i*int(b.Tda)+j].Data[0] = b.Data[i*int(b.Tda)+j].Data[0]*t1 - b.Data[i*int(b.Tda)+j].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] = b.Data[i*int(b.Tda)+j].Data[0]*t2 + b.Data[i*int(b.Tda)+j].Data[1]*t1
					}
					for k := j + 1; k < n; k++ {
						t1 = a.Data[j*int(a.Tda)+k].Data[0]
						t2 = a.Data[j*int(a.Tda)+k].Data[1]
						b.Data[i*int(b.Tda)+j].Data[0] += b.Data[i*int(b.Tda)+k].Data[0]*t1 - b.Data[i*int(b.Tda)+k].Data[1]*(-t2)
						b.Data[i*int(b.Tda)+j].Data[1] += b.Data[i*int(b.Tda)+k].Data[0]*t2 + b.Data[i*int(b.Tda)+k].Data[1]*t1
					}
					t1 = b.Data[i*int(b.Tda)+j].Data[0]
					t2 = b.Data[i*int(b.Tda)+j].Data[1]
					b.Data[i*int(b.Tda)+j].Data[0] = alpha.Data[0]*t1 - alpha.Data[1]*t2
					b.Data[i*int(b.Tda)+j].Data[1] = alpha.Data[0]*t2 + alpha.Data[1]*t1
				}
			}
		}
	}
	return nil
}

// Unit Tests

// Helper function to create a new Vector.
func newVector(size uint, stride int, data []float64) *Vector {
	if data == nil {
		data = make([]float64, size*uint(stride))
	}
	return &Vector{Size: size, Stride: stride, Data: data}
}

// Helper function to create a new VectorFloat.
func newVectorFloat(size uint, stride int, data []float32) *VectorFloat {
	if data == nil {
		data = make([]float32, size*uint(stride))
	}
	return &VectorFloat{Size: size, Stride: stride, Data: data}
}

// Helper function to create a new VectorComplex.
func newVectorComplex(size uint, stride int, data []GslComplex) *VectorComplex {
	if data == nil {
		data = make([]GslComplex, size*uint(stride))
	}
	return &VectorComplex{Size: size, Stride: stride, Data: data}
}

// Helper function to create a new VectorComplexFloat.
func newVectorComplexFloat(size uint, stride int, data []GslComplexFloat) *VectorComplexFloat {
	if data == nil {
		data = make([]GslComplexFloat, size*uint(stride))
	}
	return &VectorComplexFloat{Size: size, Stride: stride, Data: data}
}

// Helper function to create a new Matrix.
func newMatrix(size1, size2 uint, tda int, data []float64) *Matrix {
	if data == nil {
		data = make([]float64, size1*uint(tda))
	}
	return &Matrix{Size1: size1, Size2: size2, Tda: tda, Data: data}
}

// Helper function to create a new MatrixFloat.
func newMatrixFloat(size1, size2 uint, tda int, data []float32) *MatrixFloat {
	if data == nil {
		data = make([]float32, size1*uint(tda))
	}
	return &MatrixFloat{Size1: size1, Size2: size2, Tda: tda, Data: data}
}

// Helper function to create a new MatrixComplex.
func newMatrixComplex(size1, size2 uint, tda int, data []GslComplex) *MatrixComplex {
	if data == nil {
		data = make([]GslComplex, size1*uint(tda))
	}
	return &MatrixComplex{Size1: size1, Size2: size2, Tda: tda, Data: data}
}

// Helper function to create a new MatrixComplexFloat.
func newMatrixComplexFloat(size1, size2 uint, tda int, data []GslComplexFloat) *MatrixComplexFloat {
	if data == nil {
		data = make([]GslComplexFloat, size1*uint(tda))
	}
	return &MatrixComplexFloat{Size1: size1, Size2: size2, Tda: tda, Data: data}
}
